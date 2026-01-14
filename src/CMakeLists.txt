file(GLOB_RECURSE ICHOL_SRC CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")

list(FILTER ICHOL_SRC EXCLUDE REGEX ".*/apps/.*")

add_library(ichol STATIC ${ICHOL_SRC}
        ../include/ichol/supernode.hpp
)

target_include_directories(ichol PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)

target_compile_options(ichol PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(CUDAToolkit REQUIRED)

find_package(MPI REQUIRED)

target_link_libraries(ichol PUBLIC
  MPI::MPI_CXX
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CHOLMOD REQUIRED IMPORTED_TARGET CHOLMOD)

# Prefer the AMD that matches the loaded SuiteSparse module
find_library(AMD_LIB amd
    HINTS "$ENV{EBROOTSUITESPARSE}/lib64" "$ENV{EBROOTSUITESPARSE}/lib"
    REQUIRED
)

pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET petsc)
if (NOT TARGET PkgConfig::PETSC)
  pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
endif()

target_link_libraries(ichol PUBLIC
    PkgConfig::CHOLMOD
    PkgConfig::PETSC
    ${AMD_LIB}
)

find_library(SPQR_CFG_LIB suitesparseconfig
    HINTS "$ENV{EBROOTSUITESPARSE}/lib64" "$ENV{EBROOTSUITESPARSE}/lib"
    REQUIRED
)

target_link_libraries(ichol PUBLIC
    ${SPQR_CFG_LIB}
)

target_link_libraries(ichol PUBLIC
    PkgConfig::CHOLMOD
)

target_link_libraries(ichol PUBLIC
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
)

target_link_libraries(ichol PUBLIC PkgConfig::PETSC)

add_executable(ict_pcg
    apps/ict_pcg_main.cpp
)

target_link_libraries(ict_pcg PRIVATE ichol)

