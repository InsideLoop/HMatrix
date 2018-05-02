#pragma once

#ifdef IL_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#define IL_CBLAS_INT MKL_INT
#define IL_CBLAS_LAYOUT CBLAS_LAYOUT
#elif IL_OPENBLAS
#include <OpenBLAS/cblas.h>
#include <OpenBLAS/lapacke.h>
#define IL_CBLAS_INT int
#define IL_CBLAS_LAYOUT CBLAS_ORDER
#endif

#include <Matrix.h>
#include <il/Array2DView.h>
#include <il/ArrayView.h>
#include <il/linearAlgebra/Matrix.h>

#include <hmatrix/HMatrix.h>
#include <luhmatrix/LuHMatrix.h>

namespace il {

void solve(const il::LuHMatrix<double, int>& lu, il::spot_t s, il::io_t,
           il::ArrayEdit<double> x);

void solveLower(const il::LuHMatrix<double, int>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x);
void solveLower(const il::LuHMatrix<double, int>& lu, il::spot_t s, il::io_t,
                il::Array2DEdit<double> A);
void solveLower(const il::LuHMatrix<double, int>& lu, il::spot_t slu,
                il::spot_t s, il::io_t, il::LuHMatrix<double, int>& A);

void solveUpper(const il::LuHMatrix<double, int>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x);
void solveUpper(const il::LuHMatrix<double, int>& lu, il::spot_t s, il::io_t,
                il::Array2DEdit<double> A);
void solveUpper(const il::LuHMatrix<double, int>& lu, il::spot_t slu,
                il::spot_t s, il::io_t, il::LuHMatrix<double, int>& A);


void solve(il::ArrayView<int> pivot, il::Array2DView<double> A,
           il::MatrixType type, il::io_t, il::ArrayEdit<double> x);
void solve(il::ArrayView<int> pivot, il::Array2DView<double> A,
           il::MatrixType type, il::io_t, il::Array2DEdit<double> B);
void solve(il::Array2DView<double> A, il::MatrixType type, il::io_t,
           il::ArrayEdit<double> x);
void solve(il::Array2DView<double> A, il::MatrixType type, il::io_t,
           il::Array2DEdit<double> B);

}  // namespace il