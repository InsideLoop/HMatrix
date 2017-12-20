#include <iostream>
#include <random>

#include <il/Array2D.h>
#include <il/StaticArray.h>
#include <il/StaticArray2D.h>
#include <il/Timer.h>
#include <il/algorithmArray.h>

#include <il/linear_algebra/dense/blas/blas.h>
#include <il/linear_algebra/dense/blas/dot.h>
#include <il/linear_algebra/dense/factorization/LU.h>
#include <il/linear_algebra/dense/factorization/Singular.h>

class Random {
 private:
  double x;

 public:
  Random() { x = 0.12345678987654321; }
  double next() {
    x = 4 * x * (1 - x);
    return x;
  }
};

template <il::int_t p>
class Matrix {
 private:
  il::int_t n0_;
  il::int_t n1_;
  double x0_;
  double x1_;

 public:
  Matrix(il::int_t n0, il::int_t n1, double x0, double x1) {
    n0_ = n0;
    n1_ = n1;
  }
  il::int_t size(il::int_t k) const {
    IL_EXPECT_FAST(k >= 0 && k <= 2);
    return (k == 0) ? n0_ : n1_;
  }
  il::StaticArray2D<double, p, p> operator()(il::int_t i0, il::int_t i1) {
    IL_EXPECT_FAST(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(n0_));
    IL_EXPECT_FAST(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(n1_));

    const double xi = x0_ + i0 * (1.0 / n0_);
    const double xj = x1_ + i1 * (1.0 / n1_);
    const double u = std::exp(-il::abs(xi - xj));
    return il::StaticArray2D<double, p, p>{
        il::value, {{u, 0.0, 0.0}, {0.0, 2 * u, 0.0}, {0.0, 0.0, 3 * u}}};
  }
};

template <il::int_t p>
il::StaticArray2D<double, p, p> residual(const Matrix<p>& M,
                                         const il::Array2D<double>& A,
                                         const il::Array2D<double>& B,
                                         il::int_t i0, il::int_t i1,
                                         il::int_t r) {
  il::StaticArray2D<double, p, p> matrix = M(i0, i1);
  il::Array2DEdit<double> reference_matrix = matrix.edit();
  il::blas(-1.0,
           A.view(il::Range{i0 * p, (i0 + 1) * p}, il::Range{0, r * p}),
           B.view(il::Range{0, r * p}, il::Range{i1 * p, (i1 + 1) * p}),
           1.0, il::io, reference_matrix);

  return matrix;
};

template <il::int_t p>
il::int_t searchI1(const Matrix<p>& M, const il::Array2D<double>& A,
                   const il::Array2D<double>& B, il::int_t i0_search,
                   const il::Array<il::int_t>& i1_used) {
  const il::int_t n0 = M.size(0);
  const il::int_t n1 = M.size(1);
  const il::int_t rank = i1_used.size();

  il::int_t i1_search = -1;
  double largest_singular_value = 0.0;
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    bool already_searched = false;
    for (il::int_t k = 0; k < i1_used.size(); ++k) {
      if (i1 == i1_used[k]) {
        already_searched = true;
      }
    }
    if (!already_searched) {
      // To optimize: We don't need to compute the full list of singular
      // values. Only the smallest singular value needs to be computed
      il::StaticArray2D<double, p, p> matrix =
          residual(M, A, B, i0_search, i1, rank);

      il::Status status{};
      il::StaticArray<double, p> singular_values =
          il::singularValues(matrix, il::io, status);
      status.abortOnError();

      il::sort(il::io, singular_values);

      if (singular_values[0] > largest_singular_value) {
        i1_search = i1;
        largest_singular_value = singular_values[0];
      }
    }
  }
  return i1_search;
}

template <il::int_t p>
il::int_t searchI0(const Matrix<p>& M, const il::Array2D<double>& A,
                   const il::Array<il::int_t> i0_used, il::int_t i1) {
  const il::int_t n0 = M.size(0);
  const il::int_t n1 = M.size(1);
  const il::int_t rank = i0_used.size();

  il::int_t i0_search = -1;
  double largest_singular_value = 0.0;
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    bool already_searched = false;
    for (il::int_t k = 0; k < i0_used.size(); ++k) {
      if (i0 == i0_used[k]) {
        already_searched = true;
      }
    }
    if (!already_searched) {
      // To optimize: We don't need to compute the full list of singular
      // values. Only the smallest singular value needs to be computed
      il::StaticArray2D<double, p, p> matrix{};
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          matrix(b0, b1) = A(i0 * p + b0, rank * p + b1);
        }
      }

      il::Status status{};
      il::StaticArray<double, p> singular_values =
          il::singularValues(matrix, il::io, status);
      status.abortOnError();

      il::sort(il::io, singular_values);

      if (singular_values[0] > largest_singular_value) {
        i0_search = i0;
        largest_singular_value = singular_values[0];
      }
    }
  }
  return i0_search;
}

int main() {
  const il::int_t p = 3;
  const il::int_t n0 = 5;
  const il::int_t n1 = 5;
  const double x0 = 0.0;
  const double x1 = 5.0;

  Matrix<p> M{n0, n1, x0, x1};
  il::Array2D<double> A{n0 * p, 0};
  il::Array2D<double> B{0, n1 * p};

  il::Array<il::int_t> i0_used{};
  il::Array<il::int_t> i1_used{};

  il::int_t rank = 0;
  il::int_t i0_search = 0;
  double frobenius_low_rank = 0.0;

  while (true) {
    // In the Row i0_search of the matrix M - Sum_{k = 0}^rank Ak cross Bk
    // we search for the p x p matrix whose lowest singular value is the highest
    //
    const il::int_t i1_search = searchI1(M, A, B, i0_search, i1_used);
    if (i1_search == -1) {
      // We don't have any pivot
      IL_UNREACHABLE;
    }
    i0_used.append(i0_search);
    i1_used.append(i1_search);

    // Now, we compute the inverse for the pxp-matrix which has the largest
    // smallest singular value. The matrix we have has to be nonsingular.
    // Otherwise, we would have gotten i1_search == -1
    //
    il::StaticArray2D<double, p, p> matrix =
        residual(M, A, B, i0_search, i1_search, rank);
    il::Status status{};
    il::LU<il::StaticArray2D<double, p, p>> lu{matrix, il::io, status};
    status.abortOnError();
    il::StaticArray2D<double, p, p> gamma = lu.inverse();
    // Just to check
    il::StaticArray2D<double, p, p> check_identity = il::dot(gamma, matrix);

    // Update the Matrices A and B to take into account the new ranks
    A.resize(n0 * p, (rank + 1) * p);
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      il::StaticArray2D<double, p, p> matrix =
          residual(M, A, B, i0, i1_search, rank);
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          A(i0 * p + b0, rank * p + b1) = matrix(b0, b1);
        }
      }
    }
    B.resize((rank + 1) * p, n1 * p);
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      il::StaticArray2D<double, p, p> matrix =
          residual(M, A, B, i0_search, i1, rank);
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          B(rank * p + b0, i1 * p + b1) = matrix(b0, b1);
        }
      }
    }

    // New value for the norm of A
    il::StaticArray2D<double, p, p> frobenius_A{0.0};
    il::blas(
        1.0,
        A.view(il::Range{0, n0 * p}, il::Range{rank * p, (rank + 1) * p}),
        il::Blas::Transpose,
        A.view(il::Range{0, n0 * p}, il::Range{rank * p, (rank + 1) * p}),
        0.0, frobenius_A.edit());
    // New value for the norm of B
    il::StaticArray2D<double, p, p> frobenius_B{0.0};
    il::blas(
        1.0,
        B.view(il::Range{rank * p, (rank + 1) * p}, il::Range{0, n1 * p}),
        B.view(il::Range{rank * p, (rank + 1) * p}, il::Range{0, n1 * p}),
        il::Blas::Transpose, 0.0, frobenius_B.edit());

    il::int_t i0_search = searchI0(M, A, i0_used, i1_search);
    ++rank;

    if (rank == il::min(n0, n1)) {
      break;
    }
  }

  // Compute the difference in between the original matrix and A.B
  il::Array2D<double> diff{n0 * p, n1 * p};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      const il::StaticArray2D<double, p, p> matrix = M(i0, i1);
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          diff(i0 * p + b0, i1 * p + b1) = matrix(b0, b1);
        }
      }
    }
  }
  il::blas(-1.0, A, B, 1.0, il::io, diff);

  return 0;
}