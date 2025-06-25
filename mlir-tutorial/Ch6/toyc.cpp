#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/IR/Diagnostics.h"
#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
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
    enum Action {
        None,
        DumpAST,
        DumpMLIR,
        DumpMLIRAffine,
        DumpMLIRLLVM,
        DumpLLVMIR,
        RunJIT
    };
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(RunJIT, "jit", "JIT the code and run it by invoking the main function"))
);

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

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

int loadMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
    if (inputType != InputType::MLIR && !llvm::StringRef(inputFilename).ends_with(".mlir")) {
        auto moduleAST = parseInputFile(inputFilename);
        if (!moduleAST)
            return 6;
        module = mlirGen(context, *moduleAST);
        return !module ? 1 : 0;
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file" << inputFilename << "\n";
        return 3;
    }
    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
    if (int error = loadMLIR(context, module))
        return error;

    mlir::PassManager pm(module.get()->getName());
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
        return 4;

    bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
    bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

    if (enableOpt || isLoweringToAffine) {
        pm.addPass(mlir::createInlinerPass());
        mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
        optPM.addPass(mlir::toy::createShapeInferencePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
    }

    if (isLoweringToAffine) {
        pm.addPass(mlir::toy::createLowerToAffinePass());
        mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
        if (enableOpt) {
            optPM.addPass(mlir::affine::createLoopFusionPass());
            optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
        }
    }

    if (isLoweringToLLVM) {
        pm.addPass(mlir::toy::createLowerToLLVMPass());
        pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
    }

    if (mlir::failed(pm.run(*module)))
        return 4;
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

int dumpLLVMIR(mlir::ModuleOp module) {
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        llvm::errs() << "Could not create JITTargetMachineBuilder\n";
        return -1;
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
        llvm::errs() << "Could not create TargetMachine\n";
        return -1;
    }
    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(), tmOrError.get().get());
    auto optPipeline = mlir::makeOptimizingTransformer(enableOpt ? 3 : 0, 0, nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Failed to optimize LLVM IR" << err << "\n";
        return -1;
    }
    llvm::errs() << *llvmModule << "\n";
    return 0;
}

int runJit(mlir::ModuleOp module) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    auto optPipeline = mlir::makeOptimizingTransformer(enableOpt ? 3 : 0, 0, nullptr);
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;

    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
    if (emitAction == Action::DumpAST)
        return dumpAST();

    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);
    mlir::LLVM::registerInlinerInterface(registry);

    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::toy::ToyDialect>();
    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (int error = loadAndProcessMLIR(context, module))
        return error;
    
    bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
    if (isOutputingMLIR) {
        module->dump();
        return 0;
    }

    if (emitAction == Action::DumpLLVMIR)
        return dumpLLVMIR(*module);

    if (emitAction == Action::RunJIT)
        return runJit(*module);

    llvm::errs() << "No action specified (parsing only?), use -emit=<action>";
    return -1;
}