include_directories(include)
add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh3CombineIncGen)

add_executable(toy-ch3
    toyc.cpp
    parser/AST.cpp
    mlir/MLIRGen.cpp
    mlir/Dialect.cpp
    mlir/ToyCombine.cpp
)
add_dependencies(toy-ch3 ToyCh3OpsIncGen ToyCh3CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

target_link_libraries(toy-ch3
    PRIVATE
        MLIRSupport
        MLIRAnalysis
        MLIRFunctionInterfaces
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRTransforms
)