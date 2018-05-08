#pragma once

#include <hmatrix/HMatrix.h>
#include <il/linearAlgebra/dense/factorization/luDecomposition.h>

namespace il {

template <typename T>
void luDecomposition(double epsilon, il::io_t, il::HMatrix<T>& H);

template <typename T>
void luDecomposition(double epsilon, il::spot_t s, il::io_t,
                     il::HMatrix<T>& H);

template <typename T>
void luDecomposition(double epsilon, il::io_t, il::HMatrix<T>& H) {
  luDecomposition(epsilon, H.root(), il::io, H);
}

template <typename T>
void luDecomposition(double epsilon, il::spot_t s, il::io_t,
                     il::HMatrix<T>& H) {
  if (H.isFullRank(s)) {
    H.ConvertToFullLu(s);
    il::ArrayEdit<int> pivot = H.AsFullLuPivot(s);
    il::Array2DEdit<T> lu = H.AsFullLu(s);
    il::luDecomposition(il::io, pivot, lu);
  } else if (H.isHierarchical(s)) {
    const il::spot_t s00 = H.child(s, 0, 0);
    const il::spot_t s01 = H.child(s, 0, 1);
    const il::spot_t s10 = H.child(s, 1, 0);
    const il::spot_t s11 = H.child(s, 1, 1);
    luDecomposition(epsilon, s00, il::io, H);
    il::solveLower(epsilon, H, s00, s01, il::io, H);
    il::solveUpperRight(epsilon, H, s00, s10, il::io, H);
    il::blas(epsilon, T{-1.0}, H, s10, H, s01, T{1.0}, s11, il::io, H);
    il::luDecomposition(epsilon, s11, il::io, H);
  }
}

}  // namespace il
