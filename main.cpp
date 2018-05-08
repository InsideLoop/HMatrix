#include <iostream>

#include <Matrix.h>
#include <cluster/cluster.h>
#include <compression/toHMatrix.h>
#include <hmatrix/HMatrix.h>
#include <hmatrix/HMatrixUtils.h>
#include <linearAlgebra/blas/hdot.h>
#include <linearAlgebra/factorization/luDecomposition.h>

int main() {
  const il::int_t n = 8192;
  const il::int_t dim = 2;
  const il::int_t leaf_max_size = 32;

  const double radius = 1.0;
  il::Array2D<double> point{n, dim};
  for (il::int_t i = 0; i < n; ++i) {
    point(i, 0) = radius * std::cos((il::pi * (i + 0.5)) / n);
    point(i, 1) = radius * std::sin((il::pi * (i + 0.5)) / n);
  }
  const il::Cluster cluster = il::cluster(leaf_max_size, il::io, point);

  const double eta = 0.3;
  const il::Tree<il::SubHMatrix, 4> hmatrix_tree =
      il::hmatrixTree(point, cluster.partition, eta);

  const double alpha = 1.0;
  const double beta = 0.1 / n;
  const il::Matrix<std::complex<double>> M{point, alpha, beta};
  const double epsilon = 1.0e-1;
  il::HMatrix<std::complex<double>> h = il::toHMatrix(M, hmatrix_tree, epsilon);

  std::cout << "Compression ratio: " << il::compressionRatio(h) << std::endl;

  il::Array<std::complex<double>> x{h.size(0), 1.0};
  il::Array<std::complex<double>> y = il::dot(h, x);

  const double epsilon_lu = 1.0e-10;
  il::luDecomposition(epsilon_lu, il::io, h);

  std::cout << "Compression ratio of HLU-decomposition: "
            << il::compressionRatio(h) << std::endl;

  il::solve(h, il::io, y.Edit());

  double relative_error = 0.0;
  for (il::int_t i = 0; i < y.size(); ++i) {
    const double re = il::abs(y[i] - 1.0);
    if (re > relative_error) {
      relative_error = re;
    }
  }

  std::cout << "Relative error: " << relative_error << std::endl;

  return 0;
}
