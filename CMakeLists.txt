cmake_minimum_required(VERSION 3.8)
project(hmatrix)

set(CMAKE_CXX_STANDARD 14)

set(IL_BLAS 1)
set(IL_MKL 1)
set(IL_PARALLEL_MKL 0)
set(IL_PARALLEL 0)

if (IL_MKL)
    if (UNIX)
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -DIL_MKL -DIL_BLAS")
            if (IL_PARALLEL_MKL)
                set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 dl pthread m)
            else()
                set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core dl pthread m)
            endif()
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -DIL_MKL -DIL_BLAS")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
            if (IL_PARALLEL_MKL)
                set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 dl pthread m)
            else()
                set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core dl pthread m)
            endif()
            link_directories($ENV{MKLROOT}/lib/intel64_lin)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel -DIL_MKL -DIL_BLAS")
        endif()
    endif()
elseif (IL_OPENBLAS)
    if (UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_OPENBLAS -DIL_BLAS")
        set(CMAKE_OPENBLAS_LIBRARIES "openblas")
    elseif(WIN32)
        if (MINGW)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_OPENBLAS -DIL_BLAS")
            set(CMAKE_OPENBLAS_LIBRARIES "openblas")
        endif()
    elseif (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_MKL -DIL_OPENBLAS -DIL_BLAS")
        set(CMAKE_OPENBLAS_LIBRARIES "openblas")
    endif()
endif()

if (IL_PARALLEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DIL_PARALLEL")
endif()

add_executable(HMatrix
        src/core/Utilities.cpp
        src/elasticity/Simplified3D.cpp
        cluster/cluster.h
        cluster/cluster.cpp
        compression/adaptiveCrossApproximation.h
        compression/routines.h
        hmatrix/HMatrix.h
        hmatrix/HMatrixType.h
        hmatrix/HMatrixNode.h
        hmatrix/HMatrixUtils.h
        hmatrix/LowRank.h
        linearAlgebra/factorization/luDecomposition.h
        linearAlgebra/factorization/lowRankApproximation.h
        linearAlgebra/factorization/lowRankApproximation.cpp
        arrayFunctor/MatrixGenerator.h
        main.cpp
        Matrix.h
        arrayFunctor/ArrayFunctor.h
        linearAlgebra/blas/hsolve.h
        linearAlgebra/blas/hblas.h
        arrayFunctor/FullMatrix.h)

target_include_directories(HMatrix PRIVATE ${CMAKE_SOURCE_DIR})
if (IL_PARALLEL)
    target_link_libraries(HMatrix ${CMAKE_MKL_LIBRARIES} tbb)
else()
    target_link_libraries(HMatrix ${CMAKE_MKL_LIBRARIES})
endif()


################################################################################
# For Unit Tests
################################################################################

set(UNIT_TEST_FILES
    cluster/cluster.cpp
        linearAlgebra/factorization/lowRankApproximation.cpp
    hmatrix/_test/HMatrixUtils_test.cpp
    compression/_test/adaptiveCrossApproximation_test.cpp
    linearAlgebra/factorization/_test/luDecomposition_test.cpp
    gtest/src/gtest-all.cc
    )

add_executable(HMatrixUnitTest ${SOURCE_FILES} ${UNIT_TEST_FILES} test.cpp)
target_include_directories(HMatrixUnitTest PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gtest)

if (IL_MKL)
    if (IL_PARALLEL)
        target_link_libraries(HMatrixUnitTest pthread ${CMAKE_MKL_LIBRARIES} tbb)
    else()
        target_link_libraries(HMatrixUnitTest pthread ${CMAKE_MKL_LIBRARIES})
    endif()
else()
    target_link_libraries(HMatrixUnitTest pthread)
endif()

if (APPLE)
    if (IL_MKL)
        if (IL_PARALLEL_MKL)
            add_custom_command(TARGET HMatrixUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_intel_lp64.dylib $ENV{MKLROOT}/lib/libmkl_intel_lp64.dylib $<TARGET_FILE:HMatrixUnitTest>)
            add_custom_command(TARGET HMatrixUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_core.dylib $ENV{MKLROOT}/lib/libmkl_core.dylib $<TARGET_FILE:HMatrixUnitTest>)
            add_custom_command(TARGET HMatrixUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_intel_thread.dylib $ENV{MKLROOT}/lib/libmkl_intel_thread.dylib $<TARGET_FILE:HMatrixUnitTest>)
            add_custom_command(TARGET HMatrixUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libiomp5.dylib $ENV{MKLROOT}/../compiler/lib/libiomp5.dylib $<TARGET_FILE:HMatrixUnitTest>)
        else()
            add_custom_command(TARGET HMatrixUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_intel_lp64.dylib $ENV{MKLROOT}/lib/libmkl_intel_lp64.dylib $<TARGET_FILE:HMatrixUnitTest>)
            add_custom_command(TARGET HMatrixUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_core.dylib $ENV{MKLROOT}/lib/libmkl_core.dylib $<TARGET_FILE:HMatrixUnitTest>)
            add_custom_command(TARGET HMatrixUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libmkl_sequential.dylib $ENV{MKLROOT}/lib/libmkl_sequential.dylib $<TARGET_FILE:HMatrixUnitTest>)
        endif()
        if (IL_PARALLEL)
            add_custom_command(TARGET HMatrixUnitTest POST_BUILD COMMAND /usr/bin/install_name_tool -change @rpath/libtbb.dylib /opt/intel/compilers_and_libraries_2018.2.164/mac/tbb/lib/libtbb.dylib $<TARGET_FILE:HMatrixUnitTest>)
        endif()
    endif()
endif()