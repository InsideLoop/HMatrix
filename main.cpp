#include <il/MatrixFreeGmres.h>
#include <il/Tree.h>
#include <il/math.h>

#include <arrayFunctor/ArrayFunctor.h>
#include <cluster/cluster.h>
#include <hmatrix/HMatrix.h>
#include <hmatrix/HMatrixType.h>
#include <hmatrix/HMatrixUtils.h>

#include <Matrix.h>
#include <compression/toHMatrix.h>
#include <linearAlgebra/blas/dot.h>

int main() {
  const il::int_t n = 8;
  const il::int_t dim = 2;
  const il::int_t leaf_max_size = 2;

  //////////////////////////////////////////////////////////////////////////////
  // On this part of the code, we setup the points and we do the clustering
  //////////////////////////////////////////////////////////////////////////////
  const double radius = 1.0;
  il::Array2D<double> node{n + 1, dim};
  for (il::int_t i = 0; i < n + 1; ++i) {
    node(i, 0) = radius * std::cos((il::pi * i) / n);
    node(i, 1) = radius * std::sin((il::pi * i) / n);
  }
  il::Array2D<double> collocation{n, dim};
  for (il::int_t i = 0; i < n; ++i) {
    collocation(i, 0) = radius * std::cos((il::pi * (i + 0.5)) / n);
    collocation(i, 1) = radius * std::sin((il::pi * (i + 0.5)) / n);
  }
  const il::Cluster cluster = il::cluster(leaf_max_size, il::io, collocation);

  //////////////////////////////////////////////////////////////////////////////
  // Here we prepare the compression scheme of the matrix depending upon the
  // clustering and the relative diameter and distance in between the clusters
  //////////////////////////////////////////////////////////////////////////////
  //
  // And then, the clustering of the matrix, only from the geometry of the
  // points.
  // - Use eta = 0 for no compression
  // - Use eta = 1 for moderate compression
  // - Use eta = 10 for large compression
  // We have compression when Max(diam0, diam1) <= eta * distance
  // Use a large value for eta when you want everything to be Low-Rank when
  // you are outside the diagonal
  const double eta = 10.0;
  const il::Tree<il::SubHMatrix, 4> hmatrix_tree =
      il::hmatrixTree(collocation, cluster.partition, eta);

  //////////////////////////////////////////////////////////////////////////////
  // We build the H-Matrix
  //////////////////////////////////////////////////////////////////////////////
  const double alpha = 1.0;
  const il::Matrix M{collocation, alpha};
  const double epsilon = 1.0;
  const il::HMatrix<double> h = il::toHMatrix(M, hmatrix_tree, epsilon);

  //////////////////////////////////////////////////////////////////////////////
  // Let's play with it
  //////////////////////////////////////////////////////////////////////////////
  // First, we compute the compression ratio
  const double cr = il::compressionRatio(h);
  std::cout << "Compression ratio: " << cr << std::endl;

  // Then we compare the hmatrix to the original one
  const il::Array2D<double> h0 = il::toArray2D(h);
  const il::Array2D<double> h1 = il::toArray2D(M);
  il::Array2D<double> diff{n, n};
  for (il::int_t i1 = 0; i1 < n; ++i1) {
    for (il::int_t i0 = 0; i0 < n; ++i0) {
      diff(i0, i1) = h0(i0, i1) - h1(i0, i1);
    }
  }

  // Then we compare the result of a dot product with the hmatrix and with the
  // original matrix
  il::Array<double> x{n, 1.0};
  il::Array<double> yh = il::dot(h, x);
  il::Array<double> y1 = il::dot(h1, x);

  //////////////////////////////////////////////////////////////////////////////
  // Use an iterative solver
  //////////////////////////////////////////////////////////////////////////////
  il::HMatrixGenerator<double> matrix_free_h{h};
  const double relative_precision = 1.0e-3;
  const il::int_t max_nb_iterations = 100;
  const il::int_t restart_iteration = 20;
  il::MatrixFreeGmres<il::HMatrixGenerator<double>> gmres{
      relative_precision, max_nb_iterations, restart_iteration};

  il::Array<double> y{n, 1.0};
  il::Array<double> sol = gmres.Solve(matrix_free_h, y);
  const il::int_t nb_iterations = gmres.nbIterations();

  il::Array<double> check_sol = il::dot(h, sol);

  std::cout << "Number of iterations: " << nb_iterations << std::endl;

  return 0;
}
