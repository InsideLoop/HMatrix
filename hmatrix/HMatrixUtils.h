#pragma once

#include <hmatrix/HMatrix.h>
#include <il/linearAlgebra.h>

namespace il {

template <typename T>
void toArray2D(const il::HMatrix<T> &H, il::spot_t s, il::io_t,
               il::Array2DEdit<T> E) {
  if (H.isFullRank(s)) {
    il::Array2DView<T> A = H.asFullRank(s);
    IL_EXPECT_MEDIUM(A.size(0) == E.size(0));
    IL_EXPECT_MEDIUM(A.size(1) == E.size(1));
    for (il::int_t i1 = 0; i1 < A.size(1); ++i1) {
      for (il::int_t i0 = 0; i0 < A.size(0); ++i0) {
        E(i0, i1) = A(i0, i1);
      }
    }
  } else if (H.isLowRank(s)) {
    il::Array2DView<T> A = H.asLowRankA(s);
    il::Array2DView<T> B = H.asLowRankB(s);
    IL_EXPECT_MEDIUM(A.size(0) == E.size(0));
    IL_EXPECT_MEDIUM(B.size(0) == E.size(1));
    IL_EXPECT_MEDIUM(A.size(1) == B.size(1));
    il::blas(static_cast<T>(1), A, B, il::MatrixOperator::Transpose,
             static_cast<T>(0), il::io, E);
  } else if (H.isHierarchical(s)) {
    const il::spot_t s00 = H.child(s, 0, 0);
    const il::spot_t s10 = H.child(s, 1, 0);
    const il::spot_t s01 = H.child(s, 0, 1);
    const il::spot_t s11 = H.child(s, 1, 1);
    il::toArray2D(
        H, s00, il::io,
        E.Edit(il::Range{0, H.size(0, s00)}, il::Range{0, H.size(1, s00)}));
    il::toArray2D(
        H, s10, il::io,
        E.Edit(il::Range{H.size(0, s00), H.size(0, s00) + H.size(0, s10)},
               il::Range{0, H.size(1, s00)}));
    il::toArray2D(
        H, s01, il::io,
        E.Edit(il::Range{0, H.size(0, s00)},
               il::Range{H.size(1, s00), H.size(1, s00) + H.size(1, s01)}));
    il::toArray2D(
        H, s11, il::io,
        E.Edit(il::Range{H.size(0, s00), H.size(0, s00) + H.size(0, s10)},
               il::Range{H.size(1, s00), H.size(1, s00) + H.size(1, s01)}));
  } else {
    IL_UNREACHABLE;
  }
}

template <typename T>
il::Array2D<double> toArray2D(const il::HMatrix<T> &H) {
  const il::int_t n0 = H.size(0);
  const il::int_t n1 = H.size(1);
  il::Array2D<T> A{n0, n1};
  il::Array2DEdit<T> E = A.Edit();
  il::spot_t s = H.root();
  toArray2D(H, s, il::io, E);
  return A;
}

template <typename T>
il::int_t nbElements(const il::HMatrix<T> &H, il::spot_t s) {
  if (H.isFullRank(s)) {
    il::Array2DView<T> A = H.asFullRank(s);
    return A.size(0) * A.size(1);
  } else if (H.isLowRank(s)) {
    il::Array2DView<T> A = H.asLowRankA(s);
    il::Array2DView<T> B = H.asLowRankB(s);
    return A.size(0) * A.size(1) + B.size(0) * B.size(1);
  } else if (H.isHierarchical(s)) {
    const il::spot_t s00 = H.child(s, 0, 0);
    const il::spot_t s10 = H.child(s, 1, 0);
    const il::spot_t s01 = H.child(s, 0, 1);
    const il::spot_t s11 = H.child(s, 1, 1);
    return il::nbElements(H, s00) + nbElements(H, s10) + nbElements(H, s01) +
           nbElements(H, s11);
  } else {
    IL_UNREACHABLE;
  }
}

template <typename T>
double compressionRatio(const il::HMatrix<T> &H) {
  const il::int_t n0 = H.size(0);
  const il::int_t n1 = H.size(1);
  return nbElements(H, H.root()) / static_cast<double>(n0 * n1);
}

}  // namespace il
