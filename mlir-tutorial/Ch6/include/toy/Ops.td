#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "toy/ShapeInferenceInterface.td"

def Toy_Dialect : Dialect {
    let name = "toy";
    let cppNamespace = "::mlir::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

def ConstantOp : Toy_Op<"constant", [Pure]> {
    let summary = "constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached to the
        operation as an attribute. For example:

        ```mlir
            %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                                : tensor<2x3xf64>
        ```
    }];

    let arguments = (ins F64ElementsAttr:$value);

    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "DenseElementsAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,

        OpBuilder<(ins "double":$value)>
    ];

    let hasVerifier = 1;
}

def AddOp : Toy_Op<"add",
        [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two tensors.
        The shapes of the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def CastOp : Toy_Op<"cast", [
        DeclareOpInterfaceMethods<CastOpInterface>,
        DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
        Pure,
        SameOperandsAndResultShape
    ]> {
    let summary = "shape cast operation";
    let description = [{
        The "cast" operation converts a tensor from one type to an equivalent type
        without changing any data elements. The source and destination types must
        both be tensor types with the same element type. If both are ranked, then
        shape is required to match. The operation is invalid if converting to a
        mismatching constant dimension.
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$output);
    let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

}

def FuncOp : Toy_Op<"func", [
        FunctionOpInterface, IsolatedFromAbove
    ]> {
    let summary = "user defined function operation";
    let description = [{
        The "toy.func" operation represents a user defined function. These are callable SSA-region
        operations that contain toy computations.

        Example:

        ```mlir
        toy.func @main() {
            %0 = toy.constant dense<5.5> : tensor<f64>
            %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
            toy.print %1 : tensor<2x2xf64>
            toy.return
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs 
    );

    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];

    let extraClassDeclaration = [{
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
        Region *getCallableRegion() { return &getBody(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that needs to be specialized for the shape of its arguments.
        The callee name is attached as a symbol reference via an attribute. The arguments list must match the arguments
        expected by the callee. For example:
        ```mlir
            %4 = toy.generic_call @my_func(%1, %3)
                    : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```
        This is only valid if a function named "my_func" exists and takes two arguments.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<F64Tensor>:$inputs,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let results = (outs F64Tensor);

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    let builders = [
        OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
    ];
}

def MulOp : Toy_Op<"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "element-wise multiplication operation";
    let description = [{
        The "mul" operation performs element-wise multiplication between two tensors. The shapes of
        the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def PrintOp : Toy_Op<"print"> {
    let summary = "print operation";
    let description = [{
        The "print" bultin operation prints a given input tensor, and produces no results.
    }];

    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReshapeOp : Toy_Op<"reshape", [Pure]> {
    let summary = "tensor reshape operation";
    let description = [{
        Reshape operation is transforming its input tensor into a new tensor with the same
        number of elements but different shapes. For example:

        ```mlir
            %0 = toy.reshpae (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
        ```
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs StaticShapeTensorOf<[F64]>);
    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let hasCanonicalizer = 1;
}

def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
    let summary = "return operation";
    let description = [{
        The "return operation represents a return operation within a function.
        The operation takes an optional tensor operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation. For example:

        ```mlir
            toy.func @foo() -> tensor<2xf64> {
                ...
                toy.return %0 : tensor<2xf64>
            }
        ```
    }];

    let arguments = (ins Variadic<F64Tensor>:$input);
    let assemblyFormat = "($input^ `:` type($input))? attr-dict";

    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    let hasVerifier = 1;
}

def TransposeOp : Toy_Op<"transpose",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "transpose operation";
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];

    let hasVerifier = 1;
}

#endif // TOY_OPS