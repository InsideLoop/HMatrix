#pragma once

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/algorithmArray.h>
#include <il/linearAlgebra/dense/blas/blas.h>
#include <il/linearAlgebra/dense/blas/dot.h>
#include <il/linearAlgebra/dense/factorization/LU.h>
#include <il/linearAlgebra/dense/factorization/Singular.h>

#include "Matrix.h"

namespace il {

double frobeniusNorm(const il::Array2D<double> &A) {
  double ans = 0.0;
  for (il::int_t i1 = 0; i1 < A.size(1); ++i1) {
    for (il::int_t i0 = 0; i0 < A.size(0); ++i0) {
      ans += A(i0, i1) * A(i0, i1);
    }
  }
  return ans;
}

template <il::int_t p>
il::StaticArray2D<double, p, p> residual(const il::Matrix<double, p> &M,
                                         const il::Array2D<double> &A,
                                         const il::Array2D<double> &B,
                                         il::Range range0, il::Range range1,
                                         il::int_t i0, il::int_t i1,
                                         il::int_t r) {
  IL_EXPECT_MEDIUM(range0.begin <= i0);
  IL_EXPECT_MEDIUM(i0 < range0.end);
  IL_EXPECT_MEDIUM(range1.begin <= i1);
  IL_EXPECT_MEDIUM(i1 < range1.end);

  il::StaticArray2D<double, p, p> matrix = M(i0, i1);
  if (r >= 1) {
    il::Array2DEdit<double> reference_matrix = matrix.Edit();
    il::blas(
        -1.0,
        A.view(
            il::Range{(i0 - range0.begin) * p, ((i0 - range0.begin) + 1) * p},
            il::Range{0, r * p}),
        B.view(il::Range{0, r * p}, il::Range{(i1 - range1.begin) * p,
                                              ((i1 - range1.begin) + 1) * p}),
        1.0, il::io, reference_matrix);
  }
  return matrix;
};

template <il::int_t p>
il::StaticArray2D<double, p, p> lowRankSubmatrix(const il::Matrix<double, p> &M,
                                                 const il::Array2D<double> &A,
                                                 const il::Array2D<double> &B,
                                                 il::int_t i0, il::int_t i1,
                                                 il::int_t r) {
  il::StaticArray2D<double, p, p> matrix{0.0};
  if (r >= 1) {
    il::Array2DEdit<double> reference_matrix = matrix.Edit();
    il::blas(1.0, A.view(il::Range{i0 * p, (i0 + 1) * p}, il::Range{0, r * p}),
             B.view(il::Range{0, r * p}, il::Range{i1 * p, (i1 + 1) * p}), 0.0,
             il::io, reference_matrix);
  }
  return matrix;
};

template <il::int_t p>
il::int_t searchI1(const il::Matrix<double, p> &M, const il::Array2D<double> &A,
                   const il::Array2D<double> &B, il::Range range0,
                   il::Range range1, il::int_t i0_search,
                   const il::Array<il::int_t> &i1_used) {
  const il::int_t n0 = range0.end - range0.begin;
  const il::int_t n1 = range1.end - range1.begin;
  const il::int_t rank = i1_used.size();

  il::int_t i1_search = -1;
  double largest_singular_value = 0.0;
  for (il::int_t i1 = range1.begin; i1 < range1.end; ++i1) {
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
          residual(M, A, B, range0, range1, i0_search, i1, rank);

      il::Status status{};
      il::StaticArray<double, p> singular_values =
          il::singularValues(matrix, il::io, status);
      status.AbortOnError();

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
il::int_t searchI0(const il::Matrix<double, p> &M, const il::Array2D<double> &A,
                   il::Range range0, il::Range range1,
                   const il::Array<il::int_t> i0_used, il::int_t i1,
                   il::int_t rank) {
  const il::int_t n0 = range0.end - range0.begin;
  const il::int_t n1 = range1.end - range0.begin;

  il::int_t i0_search = -1;
  double largest_singular_value = 0.0;
  for (il::int_t i0 = range0.begin; i0 < range0.end; ++i0) {
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
          matrix(b0, b1) = A((i0 - range0.begin) * p + b0, rank * p + b1);
        }
      }

      il::Status status{};
      il::StaticArray<double, p> singular_values =
          il::singularValues(matrix, il::io, status);
      status.AbortOnError();

      il::sort(il::io, singular_values);

      if (singular_values[0] > largest_singular_value) {
        i0_search = i0;
        largest_singular_value = singular_values[0];
      }
    }
  }
  return i0_search;
}

template <il::int_t p>
il::Array2D<double> fullMatrix(const il::Matrix<double, p> &M, il::Range range0,
                               il::Range range1) {
  const il::int_t n0 = range0.end - range0.begin;
  const il::int_t n1 = range1.end - range1.begin;
  il::Array2D<double> ans{n0 * p, n1 * p};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      il::StaticArray2D<double, p, p> local =
          M(range0.begin + i0, range1.begin + i1);
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          ans(i0 * p + b0, i1 * p + b1) = local(b0, b1);
        }
      }
    }
  }
  return ans;
}

template <il::int_t p>
il::Array2D<double> lowRankApproximation(const il::Matrix<double, p> &M,
                                         il::Range range0, il::Range range1,
                                         const il::Array2D<double> &A,
                                         const il::Array2D<double> &B,
                                         il::int_t r) {
  const il::int_t n0 = range0.end - range0.begin;
  const il::int_t n1 = range1.end - range1.begin;
  il::Array2D<double> ans{n0 * p, n1 * p};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      il::StaticArray2D<double, p, p> local =
          il::lowRankSubmatrix(M, A, B, i0, i1, r);
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          ans(i0 * p + b0, i1 * p + b1) = local(b0, b1);
        }
      }
    }
  }
  return ans;
}

template <il::int_t p>
il::Array2D<double> fullDifference(const il::Matrix<double, p> &M,
                                   il::Range range0, il::Range range1,
                                   const il::Array2D<double> &A,
                                   const il::Array2D<double> &B, il::int_t r) {
  const il::int_t n0 = range0.end - range0.begin;
  const il::int_t n1 = range1.end - range1.begin;
  il::Array2D<double> ans{n0 * p, n1 * p};
  for (il::int_t i1 = range1.begin; i1 < range1.end; ++i1) {
    for (il::int_t i0 = range0.begin; i0 < range0.end; ++i0) {
      il::StaticArray2D<double, p, p> local =
          residual(M, A, B, range0, range1, i0, i1, r);
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          ans((i0 - range0.begin) * p + b0, (i1 - range1.begin) * p + b1) =
              local(b0, b1);
        }
      }
    }
  }
  return ans;
}

}  // namespace il
