#ifndef TOY_MLIRGEN_H
#define TOY_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
}

namespace toy {
class ModuleAST;

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);

}

#endif