#include <iostream>
#include <memory>
#include <random>

#include <il/Timer.h>

#include "build.h"
#include "HMatrix.h"
#include "adaptiveCrossApproximation.h"
#include "cluster.h"

int main() {
  const il::int_t p = 1;
  const il::int_t dim = 1;
  const il::int_t n = 1024 * 256;
  const il::int_t leaf_max_size = 128;

  // We construct the positions of the points.
  il::Array2D<double> node{n, dim};
  for (il::int_t i = 0; i < n; ++i) {
    node(i, 0) = i * (1.0 / (n - 1));
  }

  // We construct the clustering of the points.
  const hmat::Reordering reordering =
      hmat::clustering(leaf_max_size, il::io, node);

  // And then, the clustering of the matrix, only from the geomtry of the
  // points.
  const double eta = 1.0;
  const hmat::QuadTree quad_tree =
      hmat::matrixClustering(node, reordering.partition, eta);

  // We construct the matrix we need to compress
  const double lambda = 1.0;
  const hmat::Matrix<1> M{node, lambda};

  // We build the H-Matrix
  const double epsilon = 0.1;
  const hmat::HMatrix<double> H = hmat::build(M, quad_tree, epsilon);

  const double cr = H.compressionRatio();

  //  const il::int_t n0 = 10;
  //  const il::int_t n1 = 10;
  //  const double x0 = 0.0;
  //  const double x1 = 5.0;
  //  hmat::Matrix<p> M{n0, n1, x0, x1};
  //
  //  const double epsilon = 0.01;
  //  hmat::SmallRank<double> approx = hmat::adaptiveCrossApproximation(M,
  //  epsilon);

  //  const il::int_t n = 10;
  //  il::Array2D<double> A{n, n};
  //  for (il::int_t i1 = 0; i1 < n; ++i1) {
  //    for (il::int_t i0 = 0; i0 < n; ++i0) {
  //      A(i0, i1) = (i0 == i1) ? 1.0 : 0.0;
  //    }
  //  }
  //  std::unique_ptr<hmat::HMatrix<double>> M00 =
  //      std::make_unique<hmat::HMatrix<double>>(A);
  //  std::unique_ptr<hmat::HMatrix<double>> M01 =
  //      std::make_unique<hmat::HMatrix<double>>(A);
  //  std::unique_ptr<hmat::HMatrix<double>> M10 =
  //      std::make_unique<hmat::HMatrix<double>>(A);
  //  std::unique_ptr<hmat::HMatrix<double>> M11 =
  //      std::make_unique<hmat::HMatrix<double>>(A);
  //
  //  hmat::HMatrix<double> H{std::move(M00), std::move(M10), std::move(M01),
  //                          std::move(M11)};
  //  const double r = H.compressionRatio();
  //
  //  il::Array<double> x{2 * n, 1.0};
  //  il::Array<double> y = H.dot(x);

  std::cout << "Compression ratio: " << cr << std::endl;

  return 0;
}