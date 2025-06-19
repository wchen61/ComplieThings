include_directories(include)
add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh5CombineIncGen)

add_executable(toy-ch5
    toyc.cpp
    parser/AST.cpp
    mlir/MLIRGen.cpp
    mlir/Dialect.cpp
    mlir/LowerToAffineLoops.cpp
    mlir/ShapeInferencePass.cpp
    mlir/ToyCombine.cpp
)
add_dependencies(toy-ch5 ToyCh5OpsIncGen ToyCh5CombineIncGen ToyCh5ShapeInferenceInterfaceIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(toy-ch5
    PRIVATE
        ${dialect_libs}
        ${extension_libs}
        MLIRSupport
        MLIRAnalysis
        MLIRCastInterfaces
        MLIRCallInterfaces
        MLIRFunctionInterfaces
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTransforms
)