#include "toy/AST.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy;

namespace {
struct Indent {
    Indent(int &level) : level(level) { ++level; }
    ~Indent() { --level; }
    int &level;
};

class ASTDumper {
public:
    void dump(ModuleAST *node);

private:
    void dump(const VarType &type);
    void dump(ExprAST *expr);
    void dump(VarDeclExprAST *varDecl);
    void dump(ExprASTList *exprList);
    void dump(NumberExprAST *num);
    void dump(LiteralExprAST *node);
    void dump(VariableExprAST *node);
    void dump(ReturnExprAST *node);
    void dump(BinaryExprAST *node);
    void dump(CallExprAST *node);
    void dump(PrintExprAST *node);
    void dump(PrototypeAST *node);
    void dump(FunctionAST *node);

    void indent() {
        for (int i = 0; i < curIndent; i++)
            llvm::errs() << "    ";
    }
    int curIndent = 0;
};
}

template <typename T>
static std::string loc(T *node) {
    const auto &loc = node->loc();
    return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" + llvm::Twine(loc.col)).str();
}

#define INDENT()                                    \
    Indent level_(curIndent);                       \
    indent()

void ASTDumper::dump(ExprAST *expr) {
    llvm::TypeSwitch<ExprAST *>(expr)
        .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
              PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableExprAST>(
                [&](auto *node) {this->dump(node); })
        .Default([&](ExprAST *){
            INDENT();
            llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
        });
}

void ASTDumper::dump(VarDeclExprAST *varDecl) {
    INDENT();
    llvm::errs() << "VarDecl " << varDecl->getName();
    dump(varDecl->getType());
    llvm::errs() << " " << loc(varDecl) << "\n";
    dump(varDecl->getInitVal());
}

void ASTDumper::dump(ExprASTList *exprList) {
    INDENT();
    llvm::errs() << "Block {\n";
    for (auto &expr : *exprList)
        dump(expr.get());
    indent();
    llvm::errs() << "} // Block\n";
}

void ASTDumper::dump(NumberExprAST *num) {
    INDENT();
    llvm::errs() << num->getValue() << " " << loc(num) << "\n";
}

void printLitHelper(ExprAST *litOrNum) {
    if (auto *num = llvm::dyn_cast<NumberExprAST>(litOrNum)) {
        llvm::errs() << num->getValue();
        return;
    }
    auto *literal = llvm::cast<LiteralExprAST>(litOrNum);

    llvm::errs() << "<";
    llvm::interleaveComma(literal->getDims(), llvm::errs());
    llvm::errs() << ">";

    llvm::errs() << "[";
    llvm::interleaveComma(literal->getValues(), llvm::errs(),
                         [&](auto &elt) { printLitHelper(elt.get()); });
    llvm::errs() << "]";
}

void ASTDumper::dump(LiteralExprAST *node) {
    INDENT();
    llvm::errs() << "Literal: ";
    printLitHelper(node);
    llvm::errs() << " " << loc(node) << "\n";
}

void ASTDumper::dump(VariableExprAST *node) {
    INDENT();
    llvm::errs() << "var: " << node->getName() << " " << loc(node) << "\n";
}

void ASTDumper::dump(ReturnExprAST *node) {
    INDENT();
    llvm::errs() << "Return\n";
    if (node->getExpr().has_value())
        return dump(*node->getExpr());
    {
        INDENT();
        llvm::errs() << "(void)\n";
    }
}

void ASTDumper::dump(BinaryExprAST *node) {
    INDENT();
    llvm::errs() << "BinOp: " << node->getOp() << " " << loc(node) << "\n";
    dump(node->getLHS());
    dump(node->getRHS());
}

void ASTDumper::dump(CallExprAST *node) {
    INDENT();
    llvm::errs() << "Call '" << node->getCallee() << "' [ " << loc(node) << "\n";
    for (auto &arg : node->getArgs())
        dump(arg.get());
    indent();
    llvm::errs() << "]\n";
}

void ASTDumper::dump(PrintExprAST *node) {
    INDENT();
    llvm::errs() << "Print [" << loc(node) << "\n";
    dump(node->getArg());
    indent();
    llvm::errs() << "]\n";
}

void ASTDumper::dump(const VarType &type) {
    llvm::errs() << "<";
    llvm::interleaveComma(type.shape, llvm::errs());
    llvm::errs() << ">";
}

void ASTDumper::dump(PrototypeAST *node) {
    INDENT();
    llvm::errs() << "Proto '" << node->getName() << "' " << loc(node) << "\n";
    indent();
    llvm::errs() << "Params: [";
    llvm::interleaveComma(node->getArgs(), llvm::errs(),
                            [](auto &arg) { llvm::errs() << arg->getName(); });
    llvm::errs() << "]\n";
}

void ASTDumper::dump(FunctionAST *node) {
    INDENT();
    llvm::errs() << "Function \n";
    dump(node->getProto());
    dump(node->getBody());
}

void ASTDumper::dump(ModuleAST *node) {
    INDENT();
    llvm::errs() << "Module:\n";
    for (auto &f : *node)
        dump(&f);
}

namespace toy {
void dump(ModuleAST &module) { ASTDumper().dump(&module); }
}