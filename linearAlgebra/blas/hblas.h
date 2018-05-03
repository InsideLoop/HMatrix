#pragma once

#include <hmatrix/HMatrix.h>

namespace il {

void blas(double alpha, const il::HMatrix<double>& A, il::spot_t sa,
          const il::HMatrix<double>& B, il::spot_t sb, double beta,
          il::spot_t sc, il::io_t, il::HMatrix<double>& C);

void blas(double alpha, const il::HMatrix<double>& A, il::spot_t s,
          il::Array2DView<double> B, double beta, il::io_t,
          il::Array2DEdit<double> C);
void blas(double alpha, const il::HMatrix<double>& A, il::spot_t s, il::Dot op,
          il::Array2DView<double> B, double beta, il::io_t,
          il::Array2DEdit<double> C);
void blas(double alpha, il::Array2DView<double> A, const il::HMatrix<double>& B,
          il::spot_t s, double beta, il::io_t, il::Array2DEdit<double> C);

// Adds the Low Rank matrix A.B^T to the Hierachical matrix C
void blasLowRank(double alpha, il::Array2DView<double> A,
                 il::Array2DView<double> B, double beta, il::spot_t s, il::io_t,
                 il::HMatrix<double>& C);

void blas_rec(double alpha, const il::HMatrix<double>& A, il::spot_t s,
              il::MatrixType type, il::ArrayView<double> x, double beta,
              il::io_t, il::ArrayEdit<double> y);

void blas(double alpha, const il::HMatrix<double>& lu, il::spot_t s,
          il::MatrixType type, il::ArrayView<double> x, double beta, il::io_t,
          il::ArrayEdit<double> y);
void blas_rec(double alpha, const il::HMatrix<double>& A, il::spot_t s,
              il::MatrixType type, il::Array2DView<double> B, double beta,
              il::io_t, il::Array2DEdit<double> C);
void blas(double alpha, const il::HMatrix<double>& lu, il::spot_t s,
          il::MatrixType type, il::Array2DView<double> A, double beta, il::io_t,
          il::Array2DEdit<double> B);

}  // namespace il