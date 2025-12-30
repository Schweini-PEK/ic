find_package(GTest REQUIRED)

function(add_unit_test name)
    add_executable(${name} unit/${name}.cpp)

    target_link_libraries(${name}
        PRIVATE
            ichol
            GTest::gtest_main
    )

    target_include_directories(${name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/test
    )

    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_unit_test(test_io)
add_unit_test(test_fact)
add_unit_test(test_cg)
add_unit_test(test_ic)
add_unit_test(test_symbolic)
add_unit_test(test_supernodal)
