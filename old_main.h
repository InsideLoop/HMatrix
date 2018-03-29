#include <iostream>
#include <memory>
#include <random>

#include <il/Timer.h>
#include <il/linearAlgebra/dense/factorization/LU.h>
#include <il/linearAlgebra/matrixFree/solver/Gmres.h>
#include <il/linearAlgebra/matrixFree/solver/MatrixFreeGmres.h>

#include "HMatrix/adaptiveCrossApproximation.h"
#include "HMatrix/toHMatrix.h"
#include "cluster/cluster.h"
#include "HMatrix/routines.h"

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

class FedericoMatrix : public il::ArrayFunctor<double> {
 private:
  il::int_t nb_elements_;
  il::Array2D<double> collocation_;
  hfp2d::ElasticProperties elastic_properties_;
  bool diagonal_;

 public:
  FedericoMatrix(const il::Array2D<double> &collocation,
                 const hfp2d::ElasticProperties &elastic_properties,
                 bool diagonal) {
    IL_EXPECT_FAST(collocation.size(1) == 2);

    nb_elements_ = collocation.size(0);
    collocation_ = collocation;
    elastic_properties_ = elastic_properties;
    diagonal_ = diagonal;
  };
  il::int_t size(il::int_t k) const { return nb_elements_; }
  il::int_t sizeInput() const override { return 2 * nb_elements_; }
  il::int_t sizeOutput() const override { return 2 * nb_elements_; }
  hfp2d::SegmentData segmentData(il::int_t i) const {
    il::StaticArray2D<double, 2, 2> Xs{};
    const double t = (il::pi / nb_elements_) / 2;
    const double cost = std::cos(t);
    const double sint = std::sin(t);

    Xs(0, 0) = collocation_(i, 0) * cost + collocation_(i, 1) * sint;
    Xs(0, 1) = -collocation_(i, 0) * sint + collocation_(i, 1) * cost;
    Xs(1, 0) = collocation_(i, 0) * cost - collocation_(i, 1) * sint;
    Xs(1, 1) = collocation_(i, 0) * sint + collocation_(i, 1) * cost;

    const il::int_t interpolation_order = 0;
    return hfp2d::SegmentData(Xs, interpolation_order);
  };
  il::StaticArray2D<double, 2, 2> operator()(il::int_t i0, il::int_t i1) const {
    IL_EXPECT_FAST(static_cast<std::size_t>(i0) <
        static_cast<std::size_t>(nb_elements_));
    IL_EXPECT_FAST(static_cast<std::size_t>(i1) <
        static_cast<std::size_t>(nb_elements_));

    if (i0 != i1 && diagonal_) {
      return il::StaticArray2D<double, 2, 2>{0.0};
    }
    const double ker_options = 1.0;
    return hfp2d::normal_shear_stress_kernel_s3d_dp0_dd_nodal(
        segmentData(i1), segmentData(i0), 0, 0, elastic_properties_,
        ker_options);
  }
  virtual void operator()(il::ArrayView<double> x, il::io_t,
                          il::ArrayEdit<double> y) const override {
    IL_EXPECT_FAST(x.size() == 2 * nb_elements_);
    IL_EXPECT_FAST(y.size() == 2 * nb_elements_);

    for (il::int_t i0 = 0; i0 < nb_elements_; ++i0) {
      y[2 * i0] = 0.0;
      y[2 * i0 + 1] = 0.0;
      for (il::int_t i1 = 0; i1 < nb_elements_; ++i1) {
        il::StaticArray2D<double, 2, 2> matrix = (*this)(i0, i1);
        y[2 * i0] += matrix(0, 0) * x[2 * i1] + matrix(0, 1) * x[2 * i1 + 1];
        y[2 * i0 + 1] +=
            matrix(1, 0) * x[2 * i1] + matrix(1, 1) * x[2 * i1 + 1];
      }
    }
  }
};

int main() {
  const il::int_t p = 2;
  const il::int_t dim = 2;
  const il::int_t n = 100;
  const il::int_t leaf_max_size = 2;
  const double radius = 1.0;

  // We construct the positions of the points.
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

  const double young_modulus = 1.0;
  const double poisson_ratio = 0.1;
  hfp2d::ElasticProperties elastic_properties{young_modulus, poisson_ratio};

  const FedericoMatrix m{collocation, elastic_properties, false};
  const FedericoMatrix preconditionner{collocation, elastic_properties, true};

  const double relative_precision = 1.0e-1;
  const il::int_t max_nb_iterations = 100;
  const il::int_t restart_iteration = 10;
  il::Gmres gmres_solver{relative_precision, max_nb_iterations,
                         restart_iteration};
  const il::Array<double> y{p * n, 1.0};
  il::Array<double> x{p * n};

  const bool use_preconditionner = true;
  const bool use_x_as_initial_value = false;
  gmres_solver.Solve(m, preconditionner, y.view(), use_preconditionner,
                     use_x_as_initial_value, il::io, x.Edit());

  std::cout << "Number of iterations: " << gmres_solver.nbIterations()
            << std::endl;

  /*
  // We construct the clustering of the points.
  const il::Clustering reordering =
      il::clustering(leaf_max_size, il::io, collocation);

  // And then, the clustering of the matrix, only from the geomtry of the
  // points.
  // - Use eta = 0 for no compression
  // - Use eta = 1 for moderate compression
  const double eta = 1.0;
  const il::QuadTree quad_tree =
      il::matrixClustering(node, reordering.partition, eta);

  // We construct the matrix we need to compress
  const double lambda = 2.0 * il::ipow<2>(static_cast<double>(n - 1));
  const il::Matrix<2> M{collocation, elastic_properties};

  // We build the H-Matrix
  const double epsilon = 0.0;
  const il::OldHMatrix<double> h = il::build(M, quad_tree, epsilon);
  const double cr = h.compressionRatio();

  const il::Array2D<double> h_full =
      il::fullMatrix(M, il::Range{0, n}, il::Range{0, n});
  il::Array2D<double> diff = h.denseMatrix();
  il::blas(-1.0, h_full, 1.0, il::io, diff);
  const double norm_diff = il::norm(diff, il::Norm::Linf);

  std::cout << "Compression ratio: " << cr << std::endl;
  std::cout << "Error between the  matrices: " << norm_diff << std::endl;

  il::Status status{};
  il::LU<il::Array2D<double>> lu{h_full, il::io, status};
  status.AbortOnError();
  const double norm_h = il::norm(h_full, il::Norm::Linf);
  const double condition_number = lu.conditionNumber(il::Norm::Linf, norm_h);

  const il::Array<double> x{p * n, 1.0};
  const il::Array<double> y_full = il::dot(h_full, x);
  const il::Array<double> y_h = il::dot(h, x);
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
  il::MatrixFreeGmres<il::OldHMatrix<double>> solver_h{
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
  std::cout << "Number of iterations OldHMatrix: " << nb_interations_h
            << std::endl;
            */

  return 0;
}
