cmake_minimum_required(VERSION 3.8)
project(hmatrix)

set(CMAKE_CXX_STANDARD 14)

set(IL_BLAS 1)
set(IL_MKL 1)

if (IL_MKL)
    if (UNIX)
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -DIL_MKL -DIL_BLAS")
            set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 dl pthread m)
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -DIL_MKL -DIL_BLAS")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
            set(CMAKE_MKL_LIBRARIES mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 dl pthread m)
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

add_executable(HMatrix
        /home/fayard/Documents/Projects/InsideLoop/InsideLoop/il/linearAlgebra/matrixFree/solver/Gmres.cpp
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
        linearAlgebra/blas/OldHMatrix.h
        linearAlgebra/factorization/LU.h
        linearAlgebra/factorization/LU.cpp
        linearAlgebra/factorization/old/LUHMatrix.h
        linearAlgebra/factorization/old/LUHMatrixNode.h
        linearAlgebra/factorization/old/lu.h
        matrixFunctor/MatrixFunctor.h
        main.cpp
        Matrix.h
        )

target_include_directories(HMatrix PRIVATE /home/fayard/Documents/Projects/InsideLoop/InsideLoop /home/fayard/Documents/Projects/EPFL/HMatrix)
target_link_libraries(HMatrix ${CMAKE_MKL_LIBRARIES})


################################################################################
# For Unit Tests
################################################################################

set(UNIT_TEST_FILES
    gtest/src/gtest-all.cc
    hmatrix/_test/HMatrixUtils_test.cpp
    compression/_test/adaptiveCrossApproximation_test.cpp
    )

add_executable(HMatrixUnitTest ${SOURCE_FILES} ${UNIT_TEST_FILES} test.cpp)
target_include_directories(HMatrixUnitTest PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gtest)

if (IL_MKL)
    target_link_libraries(HMatrixUnitTest pthread ${CMAKE_MKL_LIBRARIES})
else()
    target_link_libraries(HMatrixUnitTest pthread)
endif()
