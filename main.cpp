#include <hmatrix/HMatrix.h>
#include <linearAlgebra/blas/hsolve.h>
#include <linearAlgebra/factorization/lowRankAddition.h>
#include <linearAlgebra/factorization/luDecomposition.h>

#include <iostream>

int main() {
  il::Array2D<double> A{4, 1, 1.0};
  il::Array2D<double> B{4, 1, 1.0};

  const double eps = 0.1;
  il::LowRank<double> lr =
      il::lowRankAddition(eps, 1.0, A.view(), B.view(), 1.0, A.view(), B.view());

  return 0;
}
