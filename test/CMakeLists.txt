find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CHOLMOD REQUIRED)


function(add_unit_test name)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/unit/${name}.cu")
        add_executable(${name} "${CMAKE_CURRENT_LIST_DIR}/unit/${name}.cu")
    else()
        add_executable(${name} "${CMAKE_CURRENT_LIST_DIR}/unit/${name}.cpp")
    endif()

    target_link_libraries(${name}
        PRIVATE
            ichol
            GTest::gtest_main
            OpenMP::OpenMP_CXX
    )

    target_include_directories(${name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/test
    )

    add_test(NAME ${name} COMMAND ${name})

    # --- add CHOLMOD to tests ---
    if(TARGET CHOLMOD::CHOLMOD)
        target_link_libraries(${name} PRIVATE CHOLMOD::CHOLMOD)
    elseif(TARGET SuiteSparse::CHOLMOD)
        target_link_libraries(${name} PRIVATE SuiteSparse::CHOLMOD)
    else()
        # fallback to variables (depends on the FindCHOLMOD.cmake you have)
        target_include_directories(${name} PRIVATE ${CHOLMOD_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE ${CHOLMOD_LIBRARIES})
    endif()

endfunction()

add_unit_test(test_io)
add_unit_test(test_fact)
add_unit_test(test_cg)
add_unit_test(test_ic)
add_unit_test(test_symbolic)
add_unit_test(test_supernodal)
add_unit_test(test_sptrsv)
