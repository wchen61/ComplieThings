include_directories(include)
add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh4CombineIncGen)

add_executable(toy-ch4
    toyc.cpp
    parser/AST.cpp
    mlir/MLIRGen.cpp
    mlir/Dialect.cpp
    mlir/ShapeInferencePass.cpp
    mlir/ToyCombine.cpp
)
add_dependencies(toy-ch4 ToyCh4OpsIncGen ToyCh4CombineIncGen ToyCh4ShapeInferenceInterfaceIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

target_link_libraries(toy-ch4
    PRIVATE
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