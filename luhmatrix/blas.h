#pragma once

#include <linearAlgebra/blas/dot.h>
#include <luhmatrix/LuHMatrix.h>

namespace il {

inline void blas_rec(double alpha, const il::LuHMatrix<double, int>& A,
                     il::spot_t s, il::MatrixType type, il::ArrayView<double> x,
                     double beta, il::io_t, il::ArrayEdit<double> y) {
  IL_EXPECT_FAST(A.size(0, s) == y.size());
  IL_EXPECT_FAST(A.size(1, s) == x.size());

  if (A.isFullRank(s)) {
    il::Array2DView<double> a = A.asFullRank(s);
    if (type == il::MatrixType::Regular) {
      il::blas(alpha, a, x, beta, il::io, y);
    } else if (type == il::MatrixType::LowerUnit) {
      IL_UNREACHABLE;
    } else {
      IL_UNREACHABLE;
    }
    return;
  } else if (A.isLowRank(s)) {
    IL_EXPECT_MEDIUM(type == il::MatrixType::Regular);
    il::Array2DView<double> a = A.asLowRankA(s);
    il::Array2DView<double> b = A.asLowRankB(s);
    const il::int_t r = a.size(1);
    il::Array<double> tmp{r, 0.0};
    il::blas(1.0, b, il::Dot::Transpose, x, 0.0, il::io, tmp.Edit());
    il::blas(alpha, a, tmp.view(), beta, il::io, y);
    return;
  } else if (A.isHierarchical(s)) {
    const il::spot_t s00 = A.child(s, 0, 0);
    const il::spot_t s10 = A.child(s, 1, 0);
    const il::spot_t s01 = A.child(s, 0, 1);
    const il::spot_t s11 = A.child(s, 1, 1);
    const il::int_t n00 = A.size(0, s00);
    const il::int_t n10 = A.size(0, s10);
    const il::int_t n01 = A.size(1, s00);
    const il::int_t n11 = A.size(1, s01);
    il::ArrayView<double> x0 = x.view(il::Range{0, n01});
    il::ArrayView<double> x1 = x.view(il::Range{n01, n01 + n11});
    il::ArrayEdit<double> y0 = y.Edit(il::Range{0, n00});
    il::ArrayEdit<double> y1 = y.Edit(il::Range{n00, n00 + n10});
    if (type == il::MatrixType::LowerUnit) {
      il::blas_rec(alpha, A, s00, il::MatrixType::LowerUnit, x0, beta, il::io,
                   y0);
      il::blas_rec(alpha, A, s10, il::MatrixType::Regular, x0, beta, il::io,
                   y1);
      il::blas_rec(alpha, A, s11, il::MatrixType::LowerUnit, x1, beta, il::io,
                   y1);
    } else {
      IL_UNREACHABLE;
    }
    return;
  } else {
    IL_UNREACHABLE;
  }
}

inline void blas(double alpha, const il::LuHMatrix<double, int>& lu,
                 il::spot_t s, il::MatrixType type, il::ArrayView<double> x,
                 double beta, il::io_t, il::ArrayEdit<double> y) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == y.size());
  IL_EXPECT_MEDIUM(lu.size(1, s) == x.size());

  il::blas_rec(alpha, lu, s, type, x, beta, il::io, y);
}

inline void blas_rec(double alpha, const il::LuHMatrix<double, int>& A,
                     il::spot_t s, il::MatrixType type,
                     il::Array2DView<double> B, double beta, il::io_t,
                     il::Array2DEdit<double> C) {
  IL_EXPECT_FAST(A.size(0, s) == C.size(0));
  IL_EXPECT_FAST(A.size(1, s) == B.size(0));

  if (A.isFullRank(s)) {
    il::Array2DView<double> a = A.asFullRank(s);
    if (type == il::MatrixType::Regular) {
      il::blas(alpha, a, B, beta, il::io, C);
    } else if (type == il::MatrixType::LowerUnit) {
      IL_UNREACHABLE;
    } else {
      IL_UNREACHABLE;
    }
    return;
  } else if (A.isLowRank(s)) {
    IL_EXPECT_MEDIUM(type == il::MatrixType::Regular);
    il::Array2DView<double> a = A.asLowRankA(s);
    il::Array2DView<double> b = A.asLowRankB(s);
    il::Array2D<double> tmp{b.size(1), B.size(1), 0.0};
    il::blas(1.0, b, il::Dot::Transpose, B, 0.0, il::io, tmp.Edit());
    il::blas(alpha, a, tmp.view(), beta, il::io, C);
    return;
  } else if (A.isHierarchical(s)) {
    const il::spot_t s00 = A.child(s, 0, 0);
    const il::spot_t s10 = A.child(s, 1, 0);
    const il::spot_t s01 = A.child(s, 0, 1);
    const il::spot_t s11 = A.child(s, 1, 1);
    const il::int_t n00 = A.size(0, s00);
    const il::int_t n10 = A.size(0, s10);
    const il::int_t n01 = A.size(1, s00);
    const il::int_t n11 = A.size(1, s01);
    il::Array2DView<double> B0 =
        B.view(il::Range{0, n01}, il::Range{0, B.size(1)});
    il::Array2DView<double> B1 =
        B.view(il::Range{n01, n01 + n11}, il::Range{0, B.size(1)});
    il::Array2DEdit<double> C0 =
        C.Edit(il::Range{0, n00}, il::Range{0, C.size(1)});
    il::Array2DEdit<double> C1 =
        C.Edit(il::Range{n00, n00 + n10}, il::Range{0, C.size(1)});
    if (type == il::MatrixType::LowerUnit) {
      il::blas_rec(alpha, A, s00, il::MatrixType::LowerUnit, B0, beta, il::io,
                   C0);
      il::blas_rec(alpha, A, s10, il::MatrixType::Regular, B0, beta, il::io,
                   C1);
      il::blas_rec(alpha, A, s11, il::MatrixType::LowerUnit, B1, beta, il::io,
                   C1);
    } else {
      IL_UNREACHABLE;
    }
    return;
  } else {
    IL_UNREACHABLE;
  }
}

inline void blas(double alpha, const il::LuHMatrix<double, int>& lu,
                 il::spot_t s, il::MatrixType type, il::Array2DView<double> A,
                 double beta, il::io_t, il::Array2DEdit<double> B) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == B.size(0));
  IL_EXPECT_MEDIUM(lu.size(1, s) == A.size(0));

  il::blas_rec(alpha, lu, s, type, A, beta, il::io, B);
}


}  // namespace il
