#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "toy/Dialect.h"

using namespace mlir;
using namespace toy;

namespace {
#include "ToyCombine.inc"
}

// c++ rewrite pattern for the TransposeOp. transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    SimplifyRedundantTranspose(mlir::MLIRContext *context)
        : OpRewritePattern<TransposeOp>(context, 1) {}

    llvm::LogicalResult matchAndRewrite(TransposeOp op, mlir::PatternRewriter &rewriter) const override {
        mlir::Value transposeInput = op.getOperand();
        TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();
        if (!transposeInputOp)
            return failure();
        rewriter.replaceOp(op, {transposeInputOp.getOperand()});
        return success();
    }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<SimplifyRedundantTranspose>(context);
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
                FoldConstantReshapeOptPattern>(context);
}