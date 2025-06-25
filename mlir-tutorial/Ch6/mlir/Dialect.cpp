#include "toy/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <string>
#include <cassert>
#include <cstdint>

using namespace mlir;
using namespace mlir::toy;

#include "toy/Dialect.cpp.inc"

struct ToyInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final {
        return true;
    }

    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
        return true;
    }

    bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
        return true;
    }

    void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
        auto returnOp = cast<ReturnOp>(op);
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }

    Operation *materializeCallConversion(OpBuilder &builder, Value input, Type resultType, Location conversionLoc) const final {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
};

void ToyDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
        >();
    addInterfaces<ToyInlinerInterface>();
}

static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    SMLoc operandsLoc = parser.getCurrentLocation();
    Type type;
    if (parser.parseOperandList(operands, 2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
        return mlir::failure();
    
    if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc, result.operands))
            return mlir::failure();
        result.addTypes(funcType.getResults());
        return mlir::success();
    }

    if (parser.resolveOperands(operands, type, result.operands))
        return mlir::failure();
    result.addTypes(type);
    return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(),
                        [=](Type type) { return type == resultType; })) {
        printer << resultType;
        return;
    }

    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
    auto dataType = RankedTensorType::get({}, builder.getF64Type());
    auto dataAttribute = DenseElementsAttr::get(dataType, value);
    ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    mlir::DenseElementsAttr value;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes))
        return failure();
    result.addTypes(value.getType());
    return mlir::success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
    printer << getValue();
}

llvm::LogicalResult ConstantOp::verify() {
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
    if (!resultType)
        return success();

    auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
    if (attrType.getRank() != resultType.getRank()) {
        return emitOpError("return type must match the one of the attached value attribute: ")
             << attrType.getRank() << " != " << resultType.getRank();
    }

    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
        if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
            return emitOpError("return type shape mismatches its attribute at dimension ")
                    << dim << " : " << attrType.getShape()[dim] << " != " << resultType.getShape()[dim];
        }
    }
    return mlir::success();
}

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) {
    printBinaryOp(p, *this);
}

void AddOp::inferShapes() {
    getResult().setType(getLhs().getType());
}

void CastOp::inferShapes() {
    getResult().setType(getInput().getType());
}

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;

    TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
    TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
    if (!input || !output || input.getElementType() != output.getElementType())
        return false;
    return !input.hasRank() || !output.hasRank() || input == output;
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name, mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    auto buildFuncType = [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
                           llvm::ArrayRef<mlir::Type> results, mlir::function_interface_impl::VariadicFlag, std::string&) {
        return builder.getFunctionType(argTypes, results);
    };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, false, getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
    mlir::function_interface_impl::printFunctionOp(
        p, *this, false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName()
    );
}

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, StringRef callee, ArrayRef<mlir::Value> arguments) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(arguments);
    state.addAttribute("callee", mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

CallInterfaceCallable GenericCallOp::getCallableForCallee() {
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
    (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

Operation::operand_range GenericCallOp::getArgOperands() {
    return getInputs();
}

MutableOperandRange GenericCallOp::getArgOperandsMutable() {
    return getInputsMutable();
}

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) {
    printBinaryOp(p, *this);
}

void MulOp::inferShapes() {
    getResult().setType(getLhs().getType());
}

llvm::LogicalResult ReturnOp::verify() {
    auto function = cast<FuncOp>((*this)->getParentOp());
    if (getNumOperands() > 1)
        return emitOpError() << "expects at most 1 return operand";

    const auto &results = function.getFunctionType().getResults();
    if (getNumOperands() != results.size())
        return emitOpError() << "does not return the same number of values ("
                             << getNumOperands() << ") as the enclosing function (" 
                             << results.size() << ")";
    
    if (!hasOperand())
        return mlir::success();

    auto inputType = *operand_type_begin();
    auto resultType = results.front();

    if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
            llvm::isa<mlir::UnrankedTensorType>(resultType))
        return mlir::success();
    
    return emitError() << "type of return operand (" << inputType
                       << ") doesn't match function result type (" << resultType
                       << ")";
}

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value value) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(value);
}

void TransposeOp::inferShapes() {
    auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
    SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
    getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

llvm::LogicalResult TransposeOp::verify() {
    auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
    auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
    if (!inputType || !resultType)
        return mlir::success();

    auto inputShape = inputType.getShape();
    if (!std::equal(inputShape.begin(), inputShape.end(), resultType.getShape().rbegin())) {
        return emitError()
                << "expected result shape to be a transpose of the input";
    }
    return mlir::success();
}

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"


