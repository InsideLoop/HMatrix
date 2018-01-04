#include <iostream>
#include <memory>
#include <random>

#include <il/Timer.h>
#include <il/linear_algebra/dense/factorization/LU.h>
#include <il/linear_algebra/matrixFree/solver/MatrixFreeGmres.h>

#include "HMatrix.h"
#include "adaptiveCrossApproximation.h"
#include "build.h"
#include "cluster.h"
#include "routines.h"

double relativeError(const il::Array<double> &a, const il::Array<double> &b) {
  IL_EXPECT_FAST(a.size() == b.size());

  const il::int_t n = a.size();
  double relative_error = 0.0;
  for (il::int_t i = 0; i < n; ++i) {
    const double re = il::abs((a[i] - b[i]) / b[i]);
    if (re > relative_error) {
      relative_error = re;
    }
  }
  return relative_error;
}

class DenseMatrix {
 private:
  il::Array2D<double> a_;

 public:
  DenseMatrix(il::Array2D<double> a) : a_{std::move(a)} {};
  il::int_t size(il::int_t d) const { return a_.size(d); };
  void dot(const il::ArrayView<double> &x, il::io_t,
           il::ArrayEdit<double> &y) const {
    IL_EXPECT_FAST(y.size() == a_.size(0));
    IL_EXPECT_FAST(x.size() == a_.size(1));

    for (il::int_t i = 0; i < y.size(); ++i) {
      y[i] = 0.0;
    }

    il::blas(1.0, a_.view(), x, 0.0, il::io, y);
  }
};

int main() {
  const il::int_t p = 1;
  const il::int_t dim = 1;
  //  const il::int_t n = 1024 * 256;
  const il::int_t n = 16 * 1024;
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
  // - Use eta = 0 for no compression
  // - Use eta = 1 for moderate compression
  const double eta = 1.0;
  const hmat::QuadTree quad_tree =
      hmat::matrixClustering(node, reordering.partition, eta);

  // We construct the matrix we need to compress
  const double lambda = 2.0 * il::ipow<2>(static_cast<double>(n - 1));
  const hmat::Matrix<1> M{node, lambda};

  // We build the H-Matrix
  const double epsilon = 0.0;
  const hmat::HMatrix<double> h = hmat::build(M, quad_tree, epsilon);
  const double cr = h.compressionRatio();

  const il::Array2D<double> h_full =
      hmat::fullMatrix(M, il::Range{0, n}, il::Range{0, n});
  il::Array2D<double> diff = h.denseMatrix();
  il::blas(-1.0, h_full, 1.0, il::io, diff);
  const double norm_diff = il::norm(diff, il::Norm::Linf);

  std::cout << "Compression ratio: " << cr << std::endl;
  std::cout << "Error between the  matrices: " << norm_diff << std::endl;

  il::Status status{};
  il::LU<il::Array2D<double>> lu{h_full, il::io, status};
  status.abortOnError();
  const double norm_h = il::norm(h_full, il::Norm::Linf);
  const double condition_number = lu.conditionNumber(il::Norm::Linf, norm_h);

  const il::Array<double> x{n, 1.0};
  const il::Array<double> y_full = il::dot(h_full, x);
  const il::Array<double> y_h = hmat::dot(h, x);
  const double relative_error_y = relativeError(y_full, y_h);

  const double relative_precision_solver = 1.0e-14;
  const il::int_t max_nb_iterations = 100;
  const il::int_t restart_iteration = 20;

  // This object should be const
  il::MatrixFreeGmres<DenseMatrix> solver_f{
      relative_precision_solver, max_nb_iterations, restart_iteration};
  const il::Array<double> x_solution_f = solver_f.solve(h_full, y_full);
  const il::int_t nb_interations_f = solver_f.nbIterations();

  // This object should be const
  il::MatrixFreeGmres<hmat::HMatrix<double>> solver_h{
      relative_precision_solver, max_nb_iterations, restart_iteration};
  const il::Array<double> x_solution_h = solver_h.solve(h, y_full);
  const il::int_t nb_interations_h = solver_h.nbIterations();

  double relative_error = 0.0;
  for (il::int_t i = 0; i < p * n; ++i) {
    const double re = il::abs((x_solution_f[i] - x[i]) / x[i]);
    if (re > relative_error) {
      relative_error = re;
    }
  }
  std::cout << "Relative error solution: " << relative_error << std::endl;
  std::cout << "Number of iterations Full: " << nb_interations_f << std::endl;
  std::cout << "Number of iterations HMatrix: " << nb_interations_h
            << std::endl;

  return 0;
}