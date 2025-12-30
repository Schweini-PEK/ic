file(GLOB_RECURSE ICHOL_SRC CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")

list(FILTER ICHOL_SRC EXCLUDE REGEX ".*/apps/.*")

add_library(ichol STATIC ${ICHOL_SRC})

target_include_directories(ichol PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)

target_compile_options(ichol PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(AMD REQUIRED IMPORTED_TARGET amd)
pkg_check_modules(SUITESPARSECONFIG REQUIRED IMPORTED_TARGET suitesparseconfig)

target_link_libraries(ichol PUBLIC
    PkgConfig::AMD
    PkgConfig::SUITESPARSECONFIG
)

target_link_libraries(ichol PUBLIC
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
)

add_executable(ict_pcg
    apps/ict_pcg_main.cpp
)

target_link_libraries(ict_pcg PRIVATE ichol)