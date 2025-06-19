#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

using namespace mlir;

static MemRefType convertTensorToMemRef(RankedTensorType type) {
    return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter &rewriter) {
    auto alloc = rewriter.create<memref::AllocOp>(loc, type);
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc; 
}

using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands, PatternRewriter &rewriter, LoopIterationFn processIteration) {
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    auto loc = op->getLoc();

    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
    affine::buildAffineLoopNest(
        rewriter, loc, lowerBounds, tensorType.getShape(), steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            Value valueToStore = processIteration(nestedBuilder, operands, ivs);
            nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc, ivs);
        }
    );
    rewriter.replaceOp(op, alloc);
}

namespace {

template<typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
    BinaryOpLowering(MLIRContext *ctx)
        : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}
    
    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
                    typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);
                    auto loadedLhs = builder.create<affine::AffineLoadOp>(
                        loc, binaryAdaptor.getLhs(), loopIvs);
                    auto loadedRhs = builder.create<affine::AffineLoadOp>(
                        loc, binaryAdaptor.getRhs(), loopIvs);
                    return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
                });
        return success();
    }
};

using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
    using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;
    LogicalResult 
    matchAndRewrite(toy::ConstantOp op, PatternRewriter &rewriter) const final {
        DenseElementsAttr constantValue = op.getValue();
        Location loc = op.getLoc();

        auto tensorType = llvm::cast<RankedTensorType>(op.getType());
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        auto valueShape = memRefType.getShape();
        SmallVector<Value, 8> constantIndices;

        if (!valueShape.empty()) {
            for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
                constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
        } else {
            constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        }

        SmallVector<Value, 2> indices;
        auto valueIt = constantValue.value_begin<FloatAttr>();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
            if (dimension == valueShape.size()) {
                rewriter.create<affine::AffineStoreOp>(
                    loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
                    llvm::ArrayRef(indices));
                return;
            }

            for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
                indices.push_back(constantIndices[i]);
                storeElements(dimension + 1);
                indices.pop_back();
            }
        };

        storeElements(0);
        rewriter.replaceOp(op, alloc);
        return success();
    }
};

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
    using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
        if (op.getName() != "main")
            return failure();
        if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
            return rewriter.notifyMatchFailure(op, [](Diagnostic &diag){
                diag << "expected 'main' to have 0 inputs and 0 results";
            });
        }

        auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), op.getFunctionType());
        rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
        rewriter.eraseOp(op);
        return success();
    }
};

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
    using OpConversionPattern<toy::PrintOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
        rewriter.modifyOpInPlace(op, [&]{ op->setOperands(adaptor.getOperands()); });
        return success();
    }
};

struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
    using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;
    LogicalResult 
    matchAndRewrite(toy::ReturnOp op, PatternRewriter &rewriter) const final {
        if (op.hasOperand())
            return failure();
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return success();
    }
};

struct TransposeOpLowering : public ConversionPattern {
    TransposeOpLowering(MLIRContext *ctx)
        : ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}
    
    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                        [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs){
                            toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                            Value input = transposeAdaptor.getInput();
                            SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                            return builder.create<affine::AffineLoadOp>(loc, input, reverseIvs);
                        });
        return success();
    }
};
}

namespace {
struct ToyToAffineLoweringPass
        : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)
    StringRef getArgument() const override { return "toy-to-affine"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<affine::AffineDialect, func::FuncDialect, memref::MemRefDialect>();
    }

    void runOnOperation() final;
};
}

void ToyToAffineLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                           arith::ArithDialect, func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<toy::ToyDialect>();
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op){
        return llvm::none_of(op->getOperandTypes(), [](Type type) { return llvm::isa<TensorType>(type); });
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
                 PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(&getContext());
    
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
    return std::make_unique<ToyToAffineLoweringPass>();
}