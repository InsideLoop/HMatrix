#pragma once

// int main() {
//  const il::int_t p = 2;
//  const il::int_t n0 = 10;
//  const il::int_t n1 = 10;
//  const double x0 = 0.0;
//  const double x1 = 5.0;
//
//  hmat::Matrix<p> M{n0, n1, x0, x1};
//  il::Array2D<double> A{n0 * p, 0};
//  il::Array2D<double> B{0, n1 * p};
//
//  il::Array<il::int_t> i0_used{};
//  il::Array<il::int_t> i1_used{};
//
//  il::int_t rank = 0;
//  il::int_t i0_search = 0;
//  double frobenius_low_rank = 0.0;
//  const double epsilon = 0.01;
//}
#include <il/Timer.h>

#include "routines.h"

namespace hmat {

template <typename T>
struct SmallRank {
  il::Array2D<T> A;
  il::Array2D<T> B;
};

template <il::int_t p>
SmallRank<double> adaptiveCrossApproximation(const hmat::Matrix<p> &M,
                                             il::Range range0, il::Range range1,
                                             double epsilon) {
  const il::int_t n0 = range0.end - range0.begin;
  const il::int_t n1 = range1.end - range1.begin;

  il::Array2D<double> A{n0 * p, 0};
  il::Array2D<double> B{0, n1 * p};

  il::Array<il::int_t> i0_used{};
  il::Array<il::int_t> i1_used{};

  il::int_t rank = 0;
  il::int_t i0_search = range0.begin;
  double frobenius_low_rank = 0.0;
  double frobenius_norm_difference = -1.0;

  while (true) {
    // In the Row i0_search of the matrix M - Sum_{k = 0}^rank Ak cross Bk
    // we search for the p x p matrix whose lowest singular value is the highest
    //
    const il::int_t i1_search =
        hmat::searchI1(M, A, B, range0, range1, i0_search, i1_used);
    if (i1_search == -1) {
      // We don't have any pivot
      break;
    }
    i0_used.Append(i0_search);
    i1_used.Append(i1_search);

    // Now, we compute the inverse for the pxp-matrix which has the largest
    // smallest singular value. The matrix we have has to be nonsingular.
    // Otherwise, we would have gotten i1_search == -1
    //
    il::StaticArray2D<double, p, p> pivot_matrix =
        hmat::residual(M, A, B, range0, range1, i0_search, i1_search, rank);
    il::Status status{};
    il::LU<il::StaticArray2D<double, p, p>> lu{pivot_matrix, il::io, status};
    status.AbortOnError();
    il::StaticArray2D<double, p, p> gamma = lu.inverse();

    // Check if we have a finite matrix
    // Be careful, as the error can raise only at the last minute with
    // denormalized number
    bool is_finite = true;
    for (il::int_t i0 = 0; i0 < p; ++i0) {
      for (il::int_t i1 = 0; i1 < p; ++i1) {
        if (!std::isfinite(gamma(i0, i1))) {
          is_finite = false;
        }
      }
    }
    if (!is_finite) {
      break;
    }

    // Just to check
    il::StaticArray2D<double, p, p> check_identity =
        il::dot(gamma, pivot_matrix);

    // Update the Matrices A and B to take into account the new ranks
    A.Resize(n0 * p, (rank + 1) * p);
    for (il::int_t i0 = range0.begin; i0 < range0.end; ++i0) {
      il::StaticArray2D<double, p, p> matrix =
          hmat::residual(M, A, B, range0, range1, i0, i1_search, rank);
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          IL_EXPECT_MEDIUM(std::isfinite(matrix(b0, b1)));
          A((i0 - range0.begin) * p + b0, rank * p + b1) = matrix(b0, b1);
        }
      }
    }
    B.Resize((rank + 1) * p, n1 * p);
    for (il::int_t i1 = range1.begin; i1 < range1.end; ++i1) {
      il::StaticArray2D<double, p, p> matrix =
          hmat::residual(M, A, B, range0, range1, i0_search, i1, rank);
      matrix = il::dot(gamma, matrix);
      for (il::int_t b1 = 0; b1 < p; ++b1) {
        for (il::int_t b0 = 0; b0 < p; ++b0) {
          IL_EXPECT_MEDIUM(std::isfinite(matrix(b0, b1)));
          B(rank * p + b0, (i1 - range1.begin) * p + b1) = matrix(b0, b1);
        }
      }
    }

    // New value for the norm of A
    il::StaticArray2D<double, p, p> frobenius_A{0.0};
    il::blas(1.0,
             A.view(il::Range{0, n0 * p}, il::Range{rank * p, (rank + 1) * p}),
             il::MatrixOperator::Tranpose,
             A.view(il::Range{0, n0 * p}, il::Range{rank * p, (rank + 1) * p}),
             0.0, il::io, frobenius_A.Edit());
    // New value for the norm of B
    il::StaticArray2D<double, p, p> frobenius_B{0.0};
    il::blas(1.0,
             B.view(il::Range{rank * p, (rank + 1) * p}, il::Range{0, n1 * p}),
             B.view(il::Range{rank * p, (rank + 1) * p}, il::Range{0, n1 * p}),
             il::MatrixOperator::Tranpose, 0.0, il::io, frobenius_B.Edit());
    // compute ||A_k B_k||^2
    double frobenius_norm_ab = 0.0;
    for (il::int_t b1 = 0; b1 < p; ++b1) {
      for (il::int_t b0 = 0; b0 < p; ++b0) {
        frobenius_norm_ab += frobenius_A(b0, b1) * frobenius_B(b0, b1);
      }
    }
    // Compute the double scalar product
    double scalar_product = 0.0;
    for (il::int_t r = 0; r < rank; ++r) {
      // Compute Ar*.Arank
      il::StaticArray2D<double, p, p> ars_arank{0.0};
      il::Array2DEdit<double> ref_ars_arank = ars_arank.Edit();
      il::blas(
          1.0, A.view(il::Range{0, n0 * p}, il::Range{r * p, (r + 1) * p}),
          il::MatrixOperator::Tranpose,
          A.view(il::Range{0, n0 * p}, il::Range{rank * p, (rank + 1) * p}),
          0.0, il::io, ref_ars_arank);
      // Compute (Ar*.Arank).Brank
      il::Array2D<double> ars_arank_brank{p, n1 * p, 0.0};
      il::Array2DEdit<double> ref_ars_arank_brank = ars_arank_brank.Edit();
      il::blas(
          1.0, ars_arank.view(),
          B.view(il::Range{rank * p, (rank + 1) * p}, il::Range{0, n1 * p}),
          0.0, il::io, ref_ars_arank_brank);

      for (il::int_t j1 = 0; j1 < n1 * p; ++j1) {
        for (il::int_t b = 0; b < p; ++b) {
          scalar_product += B(r * p + b, j1) * ars_arank_brank(b, j1);
        }
      }
    }
    frobenius_low_rank += 2 * scalar_product + frobenius_norm_ab;

    i0_search = hmat::searchI0(M, A, range0, range1, i0_used, i1_search, rank);
    ++rank;

    //    il::Array2D<double> low_rank =
    //        hmat::lowRankApproximation(M, range0, range1, A, B, rank);
    //    const double forbenius_norm_low_rank = hmat::frobeniusNorm(low_rank);

    //     Just to check
    //    il::Array2D<double> difference_matrix =
    //        hmat::fullDifference(M, range0, range1, A, B, rank);
    //    frobenius_norm_difference =
    //        hmat::frobeniusNorm(difference_matrix);

    if (i0_search == -1 ||
        frobenius_norm_ab <= il::ipow<2>(epsilon) * frobenius_low_rank ||
        rank == il::min(n0, n1)) {
      break;
    }
  }

  //  il::Array2D<double> difference_matrix =
  //      hmat::fullDifference(M, range0, range1, A, B, rank);
  //  frobenius_norm_difference = hmat::frobeniusNorm(difference_matrix);
  //  const double frobenius_norm_matrix =  frobeniusNorm(hmat::fullMatrix(M,
  //  range0, range1));
  //  IL_EXPECT_MEDIUM(std::isfinite(frobenius_norm_difference));
  //  IL_EXPECT_MEDIUM(std::isfinite(frobenius_norm_matrix));
  //  std::cout << "Relative Error: " << frobenius_norm_difference /
  //  frobenius_norm_matrix << std::endl;

  return hmat::SmallRank<double>{std::move(A), std::move(B)};
}

}  // namespace hmat
