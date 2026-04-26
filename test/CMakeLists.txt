find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

function(add_unit_test name)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/unit/${name}.cu")
        add_executable(${name} "${CMAKE_CURRENT_LIST_DIR}/unit/${name}.cu")
    else()
        add_executable(${name} "${CMAKE_CURRENT_LIST_DIR}/unit/${name}.cpp")
    endif()

    # These tests provide their own main() to parse extra CLI flags or initialize PETSc.
    if(${name} STREQUAL "test_supernodal" OR ${name} STREQUAL "test_supernode_amalgamation_experiment")
        target_link_libraries(${name}
            PRIVATE
                ichol
                GTest::gtest
                OpenMP::OpenMP_CXX
        )
    else()
        target_link_libraries(${name}
            PRIVATE
                ichol
                GTest::gtest_main
                OpenMP::OpenMP_CXX
        )
    endif()

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
add_unit_test(test_supernode_storage)
add_unit_test(test_supernode_solve)
add_unit_test(test_supernode_solve_gpu)
add_unit_test(test_supernode_amalgamation_experiment)
add_unit_test(test_sptrsv)
add_unit_test(test_mpcg)
