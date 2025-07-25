if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
    return()
endif()

include_directories(include)
add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh6CombineIncGen)

add_executable(toy-ch6
    toyc.cpp
    parser/AST.cpp
    mlir/MLIRGen.cpp
    mlir/Dialect.cpp
    mlir/LowerToAffineLoops.cpp
    mlir/LowerToLLVM.cpp
    mlir/ShapeInferencePass.cpp
    mlir/ToyCombine.cpp
)
add_dependencies(toy-ch6 ToyCh6OpsIncGen ToyCh6CombineIncGen ToyCh6ShapeInferenceInterfaceIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(toy-ch6
    PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIRSupport
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRCastInterfaces
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIRFunctionInterfaces
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTransforms
        MLIRTargetLLVMIRExport
)