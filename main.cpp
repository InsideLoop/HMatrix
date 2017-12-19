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
  il::int_t n_;

 public:
  Matrix(il::int_t n) { n_ = n; }
  il::StaticArray2D<double, p, p> operator()(il::int_t i, il::int_t j) {
    IL_EXPECT_FAST(static_cast<std::size_t>(i) < static_cast<std::size_t>(n_));
    IL_EXPECT_FAST(static_cast<std::size_t>(j) < static_cast<std::size_t>(n_));

    const double xi = 0.0 + i * (1.0 / n_);
    const double xj = 5.0 + j * (1.0 / n_);
    const double u = std::exp(-il::abs(xi - xj));

    return il::StaticArray2D<double, 3, 3>{
        il::value, {{u, 0.0, 0.0}, {0.0, 2 * u, 0.0}, {0.0, 0.0, 3 * u}}};
  }
};

int main() {
  const il::int_t p = 3;
  const il::int_t n = 5;

  Matrix<p> M{n};
  il::Array2D<il::StaticArray2D<double, p, p>> A{n, 0};
  il::Array2D<il::StaticArray2D<double, p, p>> B{0, n};

  il::Array<il::int_t> list_i{};
  il::Array<il::int_t> list_j{};

  il::int_t rank = 0;
  il::int_t i_search = 0;
  double frobenius = 0.0;
  while (true) {
    il::int_t j_search = -1;
    double largest_singular_value = -1.0;
    for (il::int_t j = 0; j < n; ++j) {
      bool already_searched = false;
      for (il::int_t k = 0; k < list_j.size(); ++k) {
        if (j == list_j[k]) {
          already_searched = true;
        }
      }
      if (!already_searched) {
        // To optimize: We don't need to compute the full list of singular
        // values. Only the smallest singular value needs to be computed
        il::StaticArray2D<double, p, p> matrix = M(i_search, j);
        for (il::int_t k = 0; k < rank; ++k) {
          il::blas(-1.0, A(i_search, k), B(k, j), 1.0, il::io, matrix);
        }

        il::Status status{};
        il::StaticArray<double, p> singular_values =
            il::singularValues(matrix, il::io, status);
        status.abortOnError();

        il::sort(il::io, singular_values);

        if (singular_values[0] > largest_singular_value) {
          j_search = j;
          largest_singular_value = singular_values[0];
        }
      }
    }
    if (largest_singular_value == 0.0) {
      break;
    }

    list_i.append(i_search);
    list_j.append(j_search);

    // We have found the largest singular value
    il::StaticArray2D<double, p, p> matrix = M(i_search, j_search);
    for (il::int_t k = 0; k < rank; ++k) {
      il::blas(-1.0, A(i_search, k), B(k, j_search), 1.0, il::io, matrix);
    }
    il::Status status{};
    il::LU<il::StaticArray2D<double, p, p>> lu{matrix, il::io, status};
    status.abortOnError();

    il::StaticArray2D<double, p, p> gamma = lu.inverse();

    // Just to check
    il::StaticArray2D<double, p, p> check_identity = il::dot(gamma, matrix);

    A.resize(n, rank + 1);
    for (il::int_t i = 0; i < n; ++i) {
      il::StaticArray2D<double, p, p> matrix = M(i, j_search);
      for (il::int_t k = 0; k < rank; ++k) {
        il::blas(-1.0, A(i, k), B(k, j_search), 1.0, il::io, matrix);
      }
      A(i, rank) = matrix;
    }
    B.resize(rank + 1, n);
    for (il::int_t j = 0; j < n; ++j) {
      il::StaticArray2D<double, p, p> matrix = M(i_search, j);
      for (il::int_t k = 0; k < rank; ++k) {
        il::blas(-1.0, A(i_search, k), B(k, j), 1.0, il::io, matrix);
      }
      B(rank, j) = il::dot(gamma, matrix);
    }

    // Compute the Frobenius norms
    for (il::int_t j = 0; j < n; ++j) {
      for (il::int_t bj = 0; bj < p; ++bj) {
        for (il::int_t bi = 0; bi < p; ++bi) {
          const double value = B(rank, j)(bi, bj);
          frobenius_B += value * value;
        }
      }
    }
    // Update Frobenius general
    il::StaticArray2D<double, p, p> A_norm{};
    for (il::int_t b0 = 0; b0 < p; ++b0) {
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        double sum = 0.0;
        for (il::int_t i = 0; i < n; ++i) {
          for (il::int_t b = 0; b < p; ++b) {
            sum += A(i, rank)(b, b0) * A(i, rank)(b ,b1);
          }
        }
        A_norm(b0, b1) = sum;
      }
    }
    il::StaticArray2D<double, p, p> B_norm{};
    for (il::int_t b0 = 0; b0 < p; ++b0) {
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        double sum = 0.0;
        for (il::int_t j = 0; j < n; ++j) {
          for (il::int_t b = 0; b < p; ++b) {
            sum += B(rank, j)(b0, b) * B(rank, j)(b1, b);
          }
        }
        B_norm(b0, b1) = sum;
      }
    }
//    double frobenius_A = 0.0;
//    for (il::int_t b = 0; b < p; ++b) {
//      frobenius_A = A_norm(b, b);
//    }
//    double frobenius_B = 0.0;
//    for (il::int_t b = 0; b < p; ++b) {
//      frobenius_B = B_norm(b, b);
//    }
    for (il::int_t bi = 0; bi < p; ++bi) {
      for (il::int_t bj = 0; bj < p; ++bj) {
        frobenius += A_norm(bi, bj) * B_norm(bi, bj);
      }
    }
    il::Array2D<double> AAB{p, n, 0.0};
    for (il::int_t bi = 0; bi < p; ++bi) {
      for (il::int_t j = 0; j < n; ++j) {
        for (il::int_t bj = 0; bj < p; ++bj) {
          AAB(bi, j) += A_norm(bi, bj) * B(rank, j)(bj, );
        }
      }
    }






    // Find the next row
    i_search = -1;
    largest_singular_value = -1.0;
    for (il::int_t i = 0; i < n; ++i) {
      // Look if i has already been searched for
      bool already_searched = false;
      for (il::int_t k = 0; k < list_i.size(); ++k) {
        if (i == list_i[k]) {
          already_searched = true;
        }
      }
      if (!already_searched) {
        // To optimize: We don't need to compute the full list of singular
        // values. Only the smallest singular value needs to be computed
        il::Status status{};
        il::StaticArray<double, p> singular_values =
            il::singularValues(A(i, rank), il::io, status);
        status.abortOnError();

        il::sort(il::io, singular_values);

        if (singular_values[0] > largest_singular_value) {
          i_search = i;
          largest_singular_value = singular_values[0];
        }
      }
    }
    ++rank;

    if (rank == n) {
      break;
    }
  }

  // Compute the difference in between the original matrix and A.B
  il::Array2D<il::StaticArray2D<double, p, p>> diff{n, n};
  for (il::int_t j = 0; j < n; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      il::StaticArray2D<double, p, p> matrix = M(i, j);
      for (il::int_t k = 0; k < rank; ++k) {
        il::blas(-1.0, A(i, k), B(k, j), 1.0, il::io, matrix);
      }
      diff(i, j) = matrix;
    }
  }

  return 0;
}