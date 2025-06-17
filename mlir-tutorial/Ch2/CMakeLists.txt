add_subdirectory(include)

add_executable(toy-ch2
    toyc.cpp
    parser/AST.cpp
    mlir/MLIRGen.cpp
    mlir/Dialect.cpp
)
add_dependencies(toy-ch2 ToyCh2OpsIncGen)

include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

target_link_libraries(toy-ch2
    PRIVATE
        MLIRSupport
        MLIRAnalysis
        MLIRFunctionInterfaces
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRTransforms
)