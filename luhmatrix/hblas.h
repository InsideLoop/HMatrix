#pragma once

#include <luhmatrix/LuHMatrix.h>

namespace il {

void blas(double alpha, const il::LuHMatrix<double, int>& A, il::spot_t sa,
          const il::LuHMatrix<double, int>& B, il::spot_t sb, double beta,
          il::spot_t sc, il::io_t, il::LuHMatrix<double, int>& C);

void blas(double alpha, const il::LuHMatrix<double, int>& A, il::spot_t s,
          il::Array2DView<double> B, double beta, il::io_t,
          il::Array2DEdit<double> C);
void blas(double alpha, const il::LuHMatrix<double, int>& A, il::spot_t s,
          il::Dot op, il::Array2DView<double> B, double beta, il::io_t,
          il::Array2DEdit<double> C);
void blas(double alpha, il::Array2DView<double> A,
          const il::LuHMatrix<double, int>& B, il::spot_t s, double beta,
          il::io_t, il::Array2DEdit<double> C);

// Adds the Low Rank matrix A.B^T to the Hierachical matrix C
void blasLowRank(double alpha, il::Array2DView<double> A,
                 il::Array2DView<double> B, double beta, il::spot_t s, il::io_t,
                 il::LuHMatrix<double, int>& C);

}  // namespace il