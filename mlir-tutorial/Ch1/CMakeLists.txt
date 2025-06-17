include_directories(include/)
add_executable(toy-ch1 toyc.cpp parser/AST.cpp)
target_link_libraries(
    toy-ch1 PRIVATE
    MLIRSupport
)