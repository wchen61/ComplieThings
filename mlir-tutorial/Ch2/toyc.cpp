#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
}
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

namespace {
    enum Action { None, DumpAST, DumpMLIR };
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = 
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }
    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);
    return parser.parseModule();
}

int dumpMLIR() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::toy::ToyDialect>();

    if (inputType != InputType::MLIR &&
        !llvm::StringRef(inputFilename).ends_with(".mlir")) {
        auto moduleAST = parseInputFile(inputFilename);
        if (!moduleAST)
            return 6;
        mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
        if (!module)
            return 1;
        module->dump();
        return 0;
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    module->dump();
    return 0;
}

int dumpAST() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
        return 5;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
        return 1;
    dump(*moduleAST);
    return 0;
}

int main(int argc, char **argv) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
    switch(emitAction) {
    case Action::DumpAST:
        return dumpAST();
    case Action::DumpMLIR:
        return dumpMLIR();
    default:
        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }
    return 0;
}