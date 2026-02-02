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

# CUDA basic options
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

# CUDA libs
target_link_libraries(ichol PUBLIC
        CUDA::cudart
        CUDA::cusparse
        CUDA::cublas
        CUDA::cusolver
)

option(ICHOL_USE_OPENMP "Enable OpenMP parallelism for supernodal numeric" ON)
option(ICHOL_USE_BLAS   "Use BLAS/LAPACK (dpotrf/dtrsm/dsyrk) for supernodal numeric" ON)

if (ICHOL_USE_OPENMP)
    # 必须找到 OpenMP，否则直接配置失败（避免运行时退化 streams=1）
    find_package(OpenMP REQUIRED)

    # 1) C++ 编译/链接启用 OpenMP
    target_link_libraries(ichol PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ichol PUBLIC ICHOL_USE_OPENMP=1)

    # 2) 关键：让 NVCC 编译 .cu 时把 -fopenmp 传给 host compiler
    target_compile_options(ichol PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )

    message(STATUS "ICHOL: OpenMP enabled for ichol (CUDA host compile gets -fopenmp)")
else()
    message(STATUS "ICHOL: OpenMP disabled for ichol (ICHOL_USE_OPENMP=OFF)")
endif()

# Ensure that BLAS and LAPACK are found if enabled
if (ICHOL_USE_BLAS)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    if (BLAS_FOUND AND LAPACK_FOUND)
        target_link_libraries(ichol PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        target_compile_definitions(ichol PUBLIC ICHOL_USE_BLAS=1)
    else()
        message(FATAL_ERROR "BLAS and LAPACK are required but were not found.")
    endif()
else()
    message(STATUS "ICHOL: BLAS is disabled for ichol (ICHOL_USE_BLAS=OFF)")
endif()

add_executable(ict_pcg
        apps/ict_pcg_main.cpp
)

target_link_libraries(ict_pcg PRIVATE ichol)
