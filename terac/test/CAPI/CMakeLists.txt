# The test links the shared library through its header, the way a caller does,
# so it is built as C and not put through `llvm_update_compile_flags` -- that
# would compile it as C++ and stop it from proving what it is here to prove.
add_llvm_executable(tera-capi-test
        PARTIAL_SOURCES_INTENDED
        execution.c
        )
target_link_libraries(tera-capi-test PRIVATE tera-capi)
