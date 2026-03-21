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
    CUDA::cusolver
)

target_link_libraries(ichol PUBLIC PkgConfig::PETSC)

option(ICHOL_USE_OPENMP "Enable OpenMP parallelism for supernodal numeric" ON)
option(ICHOL_USE_BLAS   "Use BLAS/LAPACK (dpotrf/dtrsm/dsyrk) for supernodal numeric" ON)

if (ICHOL_USE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(ichol PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(ichol PUBLIC ICHOL_USE_OPENMP=1)
    endif()
endif()

if (ICHOL_USE_BLAS)
    find_package(BLAS)
    find_package(LAPACK)
    if (BLAS_FOUND AND LAPACK_FOUND)
        target_link_libraries(ichol PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        target_compile_definitions(ichol PUBLIC ICHOL_USE_BLAS=1)
    endif()
endif()

add_executable(ict_pcg
    apps/ict_pcg_main.cpp
)

target_link_libraries(ict_pcg PRIVATE ichol)

add_executable(mpcg_pcg_compare
    apps/mpcg_pcg_compare_main.cpp
)

target_link_libraries(mpcg_pcg_compare PRIVATE ichol)

find_path(CUDSS_INCLUDE_DIR cudss.h
    HINTS "$ENV{EBROOTCUDSS}/include" "$ENV{CUDA_HOME}/include" "$ENV{EBROOTCUDA}/include")
find_library(CUDSS_LIB NAMES cudss
    HINTS "$ENV{EBROOTCUDSS}/lib64" "$ENV{EBROOTCUDSS}/lib" "$ENV{CUDA_HOME}/lib64" "$ENV{CUDA_HOME}/lib" "$ENV{EBROOTCUDA}/lib64" "$ENV{EBROOTCUDA}/lib")

if(CUDSS_INCLUDE_DIR AND CUDSS_LIB)
    target_include_directories(ichol PRIVATE ${CUDSS_INCLUDE_DIR})
    target_link_libraries(ichol PUBLIC ${CUDSS_LIB})
    target_compile_definitions(ichol PUBLIC ICHOL_HAVE_CUDSS=1)
else()
    message(WARNING "cuDSS not found: exact subdomain preconditioner backend will be disabled")
    target_compile_definitions(ichol PUBLIC ICHOL_HAVE_CUDSS=0)
endif()
