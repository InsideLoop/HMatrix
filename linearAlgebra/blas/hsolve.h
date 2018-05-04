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

namespace il {

void solve(const il::HMatrix<double>& lu, il::MatrixType type, il::io_t,
           il::ArrayEdit<double> xy);
void solve(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
           il::ArrayEdit<double> x);

void solveLower(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x);
void solveLower(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::Array2DEdit<double> A);
void solveLower(const il::HMatrix<double>& lu, il::spot_t slu, il::spot_t s,
                il::io_t, il::HMatrix<double>& A);

void solveUpper(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x);
void solveUpper(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::Array2DEdit<double> A);
void solveUpper(const il::HMatrix<double>& lu, il::spot_t slu, il::spot_t s,
                il::io_t, il::HMatrix<double>& A);

void solveUpperTranspose(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                         il::Array2DEdit<double> A);
void solveUpperTranspose(const il::HMatrix<double>& lu, il::spot_t slu,
                         il::spot_t s, il::io_t, il::HMatrix<double>& A);

void solveUpperRight(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                     il::Array2DEdit<double> A);
void solveUpperRight(const il::HMatrix<double>& lu, il::spot_t slu,
                     il::spot_t s, il::io_t, il::HMatrix<double>& A);

}  // namespace il