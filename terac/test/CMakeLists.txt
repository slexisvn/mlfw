configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_subdirectory(CAPI)

set(TERA_TEST_DEPENDS
        FileCheck count not
        tera-capi-test
        tera-gradcheck
        tera-opt
        tera-runner
        )

add_lit_testsuite(check-tera "Running the tera regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TERA_TEST_DEPENDS}
        )
set_target_properties(check-tera PROPERTIES FOLDER "Tests")

add_lit_testsuites(TERA ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TERA_TEST_DEPENDS})
