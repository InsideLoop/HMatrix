#include <il/blas.h>
#include <linearAlgebra/blas/hblas.h>

namespace il {

void blas(double alpha, const il::HMatrix<double>& A, il::spot_t sa,
          const il::HMatrix<double>& B, il::spot_t sb, double beta,
          il::spot_t sc, il::io_t, il::HMatrix<double>& C) {
  IL_EXPECT_MEDIUM(A.size(0, sa) == C.size(0, sc));
  IL_EXPECT_MEDIUM(B.size(1, sb) == C.size(1, sc));
  IL_EXPECT_MEDIUM(A.size(1, sa) == B.size(0, sb));

  if (C.isFullRank(sc)) {
    il::Array2DEdit<double> c = C.AsFullRank(sc);
    if (A.isFullRank(sa) && B.isFullRank(sb)) {
      il::Array2DView<double> a = A.asFullRank(sa);
      il::Array2DView<double> b = B.asFullRank(sb);
      il::blas(alpha, a, b, beta, il::io, c);
    } else if (A.isFullRank(sa) && B.isLowRank(sb)) {
      il::Array2DView<double> a = A.asFullRank(sa);
      il::Array2DView<double> ba = B.asLowRankA(sb);
      il::Array2DView<double> bb = B.asLowRankB(sb);
      il::Array2D<double> tmp{a.size(0), ba.size(1)};
      il::blas(1.0, a, ba, 0.0, il::io, tmp.Edit());
      il::blas(alpha, tmp.view(), bb, il::Dot::Transpose, beta, il::io, c);
    } else if (A.isLowRank(sa) && B.isFullRank(sb)) {
      il::Array2DView<double> aa = A.asLowRankA(sa);
      il::Array2DView<double> ab = A.asLowRankB(sa);
      il::Array2DView<double> b = A.asFullRank(sb);
      il::Array2D<double> tmp{ab.size(1), b.size(1)};
      il::blas(1.0, ab, il::Dot::Transpose, b, 0.0, il::io, tmp.Edit());
      il::blas(alpha, aa, tmp.view(), beta, il::io, c);
    } else if (A.isLowRank(sa) && B.isLowRank(sb)) {
      il::Array2DView<double> aa = A.asLowRankA(sa);
      il::Array2DView<double> ab = A.asLowRankB(sa);
      il::Array2DView<double> ba = B.asLowRankA(sb);
      il::Array2DView<double> bb = B.asLowRankB(sb);
      il::Array2D<double> tmp0{ab.size(1), ba.size(1)};
      il::blas(1.0, ab, il::Dot::Transpose, ba, 0.0, il::io, tmp0.Edit());
      il::Array2D<double> tmp1{aa.size(0), ba.size(1)};
      il::blas(1.0, aa, tmp0.view(), 0.0, il::io, tmp1.Edit());
      il::blas(alpha, tmp1.view(), bb, il::Dot::Transpose, beta, il::io, c);
    } else if (A.isFullRank(sa) && B.isHierarchical(sb)) {
      il::Array2DView<double> a = A.asFullRank(sa);
      il::blas(alpha, a, B, sb, beta, il::io, c);
    } else if (A.isHierarchical(sa) && B.isFullRank(sb)) {
      il::Array2DView<double> b = B.asFullRank(sb);
      il::blas(alpha, A, sa, b, beta, il::io, c);
    } else if (A.isLowRank(sa) && B.isHierarchical(sb)) {
      il::Array2DView<double> aa = A.asLowRankA(sa);
      il::Array2DView<double> ab = A.asLowRankB(sa);
      il::Array2D<double> tmp{B.size(1, sb), ab.size(1)};
      il::blas(1.0, B, sb, il::Dot::Transpose, ab, 0.0, il::io, tmp.Edit());
      il::blas(alpha, aa, tmp.view(), il::Dot::Transpose, beta, il::io, c);
    } else if (A.isHierarchical(sa) && B.isLowRank(sb)) {
      il::Array2DView<double> ba = B.asLowRankA(sb);
      il::Array2DView<double> bb = B.asLowRankB(sb);
      il::Array2D<double> tmp{A.size(0, sa), ba.size(1)};
      il::blas(1.0, A, sa, ba, 0.0, il::io, tmp.Edit());
      il::blas(alpha, tmp.view(), bb, il::Dot::Transpose, beta, il::io, c);
    } else if (A.isHierarchical(sa) && B.isHierarchical(sb)) {
      // The trick is to convert the matrix A into a low rank matrix and follow
      // one of the previous method
      IL_UNREACHABLE;
    }
  } else if (C.isLowRank(sc)) {
    if (A.isLowRank(sa) && B.isLowRank(sb)) {
      // FIXME : The branching could be improved as we do exactly as if
      // C were a Hierarchical matrix (copy and paste of the code)
      il::Array2DView<double> aa = A.asLowRankA(sa);
      il::Array2DView<double> ab = A.asLowRankB(sa);
      il::Array2DView<double> ba = B.asLowRankA(sb);
      il::Array2DView<double> bb = B.asLowRankB(sb);
      il::Array2D<double> tmp0{ab.size(1), ba.size(1)};
      il::blas(1.0, ab, il::Dot::Transpose, ba, 0.0, il::io, tmp0.Edit());
      il::Array2D<double> tmp1{aa.size(0), ba.size(1)};
      il::blas(1.0, aa, tmp0.view(), 0.0, il::io, tmp1.Edit());
      il::blasLowRank(alpha, tmp1.view(), bb, beta, sc, il::io, C);
    } else if (A.isFullRank(sa) && B.isLowRank(sb)) {
      IL_UNREACHABLE;
    } else if (A.isLowRank(sa) && B.isFullRank(sb)) {
      IL_UNREACHABLE;
    } else if (A.isHierarchical(sa) && B.isLowRank(sb)) {
      // FIXME : The branching could be improved as we do exactly as if
      // C were a Hierarchical matrix (copy and paste of the code)
      il::Array2DView<double> ba = B.asLowRankA(sb);
      il::Array2DView<double> bb = B.asLowRankB(sb);
      il::Array2D<double> tmp{A.size(0, sa), ba.size(1)};
      il::blas(1.0, A, sa, ba, 0.0, il::io, tmp.Edit());
      il::blasLowRank(alpha, tmp.view(), bb, beta, sc, il::io, C);
    } else if (A.isLowRank(sa) && B.isHierarchical(sb)) {
      // FIXME : The branching could be improved as we do exactly as if
      // C were a Hierarchical matrix (copy and paste of the code)
      il::Array2DView<double> aa = A.asLowRankA(sa);
      il::Array2DView<double> ab = A.asLowRankB(sa);
      il::Array2D<double> tmp{B.size(1, sb), ab.size(1)};
      il::blas(1.0, B, sb, il::Dot::Transpose, ab, 0.0, il::io, tmp.Edit());
      il::blasLowRank(alpha, aa, tmp.view(), beta, sc, il::io, C);
    } else if (A.isFullRank(sa) && B.isFullRank(sb)) {
      IL_UNREACHABLE;
    } else if (A.isFullRank(sa) && B.isHierarchical(sb)) {
      IL_UNREACHABLE;
    } else if (A.isHierarchical(sa) && B.isFullRank(sb)) {
      IL_UNREACHABLE;
    } else if (A.isHierarchical(sa) && B.isHierarchical(sb)) {
      IL_UNREACHABLE;
    }
  } else if (C.isHierarchical(sc)) {
    if (A.isHierarchical(sa) && B.isHierarchical(sb)) {
      il::blas(alpha, A, A.child(sa, 0, 0), B, B.child(sb, 0, 0), beta,
               C.child(sc, 0, 0), il::io, C);
      il::blas(alpha, A, A.child(sa, 0, 1), B, B.child(sb, 1, 0), 1.0,
               C.child(sc, 0, 0), il::io, C);
      il::blas(alpha, A, A.child(sa, 0, 0), B, B.child(sb, 0, 1), beta,
               C.child(sc, 0, 1), il::io, C);
      il::blas(alpha, A, A.child(sa, 0, 1), B, B.child(sb, 1, 1), 1.0,
               C.child(sc, 0, 1), il::io, C);
      il::blas(alpha, A, A.child(sa, 1, 0), B, B.child(sb, 0, 0), beta,
               C.child(sc, 1, 0), il::io, C);
      il::blas(alpha, A, A.child(sa, 1, 1), B, B.child(sb, 1, 0), 1.0,
               C.child(sc, 1, 0), il::io, C);
      il::blas(alpha, A, A.child(sa, 1, 0), B, B.child(sb, 0, 1), beta,
               C.child(sc, 1, 1), il::io, C);
      il::blas(alpha, A, A.child(sa, 1, 1), B, B.child(sb, 1, 1), 1.0,
               C.child(sc, 1, 1), il::io, C);
    } else if (A.isFullRank(sa) && B.isLowRank(sb)) {
      IL_UNREACHABLE;
    } else if (A.isLowRank(sa) && B.isFullRank(sb)) {
      IL_UNREACHABLE;
    } else if (A.isLowRank(sa) && B.isLowRank(sb)) {
      il::Array2DView<double> aa = A.asLowRankA(sa);
      il::Array2DView<double> ab = A.asLowRankB(sa);
      il::Array2DView<double> ba = B.asLowRankA(sb);
      il::Array2DView<double> bb = B.asLowRankB(sb);
      il::Array2D<double> tmp0{ab.size(1), ba.size(1)};
      il::blas(1.0, ab, il::Dot::Transpose, ba, 0.0, il::io, tmp0.Edit());
      il::Array2D<double> tmp1{aa.size(0), ba.size(1)};
      il::blas(1.0, aa, tmp0.view(), 0.0, il::io, tmp1.Edit());
      il::blasLowRank(alpha, tmp1.view(), bb, beta, sc, il::io, C);
    } else if (A.isHierarchical(sa) && B.isLowRank(sb)) {
      il::Array2DView<double> ba = B.asLowRankA(sb);
      il::Array2DView<double> bb = B.asLowRankB(sb);
      il::Array2D<double> tmp{A.size(0, sa), ba.size(1)};
      il::blas(1.0, A, sa, ba, 0.0, il::io, tmp.Edit());
      il::blasLowRank(alpha, tmp.view(), bb, beta, sc, il::io, C);
    } else if (A.isLowRank(sa) && B.isHierarchical(sb)) {
      il::Array2DView<double> aa = A.asLowRankA(sa);
      il::Array2DView<double> ab = A.asLowRankB(sa);
      il::Array2D<double> tmp{B.size(1, sb), ab.size(1)};
      il::blas(1.0, B, sb, il::Dot::Transpose, ab, 0.0, il::io, tmp.Edit());
      il::blasLowRank(alpha, aa, tmp.view(), beta, sc, il::io, C);
    } else if (A.isHierarchical(sa) && B.isFullRank(sb)) {
      IL_UNREACHABLE;
    } else if (A.isFullRank(sa) && B.isHierarchical(sb)) {
      IL_UNREACHABLE;
    } else if (A.isFullRank(sa) && B.isFullRank(sb)) {
      IL_UNREACHABLE;
    }
  }
}

void blas(double alpha, const il::HMatrix<double>& A, il::spot_t s,
          il::Array2DView<double> B, double beta, il::io_t,
          il::Array2DEdit<double> C) {
  IL_EXPECT_FAST(A.size(0, s) == C.size(0));
  IL_EXPECT_FAST(A.size(1, s) == B.size(0));
  IL_EXPECT_FAST(B.size(1) == C.size(1));

  if (A.isFullRank(s)) {
    il::Array2DView<double> a = A.asFullRank(s);
    il::blas(alpha, a, B, beta, il::io, C);
  } else if (A.isLowRank(s)) {
    il::Array2DView<double> aa = A.asLowRankA(s);
    il::Array2DView<double> ab = A.asLowRankB(s);
    const il::int_t r = aa.size(1);
    il::Array2D<double> tmp{r, B.size(1)};
    il::blas(1.0, ab, il::Dot::Transpose, B, 0.0, il::io, tmp.Edit());
    il::blas(alpha, aa, tmp.view(), beta, il::io, C);
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
    il::blas(alpha, A, s00, B0, beta, il::io, C0);
    il::blas(alpha, A, s01, B1, 1.0, il::io, C0);
    il::blas(alpha, A, s10, B0, beta, il::io, C1);
    il::blas(alpha, A, s11, B1, 1.0, il::io, C1);
  } else {
    IL_UNREACHABLE;
  }
}

void blas(double alpha, const il::HMatrix<double>& A, il::spot_t s, il::Dot op,
          il::Array2DView<double> B, double beta, il::io_t,
          il::Array2DEdit<double> C) {
  IL_EXPECT_FAST(op == il::Dot::Transpose);
  IL_EXPECT_FAST(A.size(1, s) == C.size(0));
  IL_EXPECT_FAST(A.size(0, s) == B.size(0));
  IL_EXPECT_FAST(B.size(1) == C.size(1));

  if (A.isFullRank(s)) {
    il::Array2DView<double> a = A.asFullRank(s);
    il::blas(alpha, a, il::Dot::Transpose, B, beta, il::io, C);
  } else if (A.isLowRank(s)) {
    il::Array2DView<double> aa = A.asLowRankA(s);
    il::Array2DView<double> ab = A.asLowRankB(s);
    const il::int_t r = aa.size(1);
    il::Array2D<double> tmp{r, B.size(1)};
    il::blas(1.0, aa, il::Dot::Transpose, B, 0.0, il::io, tmp.Edit());
    il::blas(alpha, ab, tmp.view(), beta, il::io, C);
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
        B.view(il::Range{0, n00}, il::Range{0, B.size(1)});
    il::Array2DView<double> B1 =
        B.view(il::Range{n00, n00 + n10}, il::Range{0, B.size(1)});
    il::Array2DEdit<double> C0 =
        C.Edit(il::Range{0, n01}, il::Range{0, C.size(1)});
    il::Array2DEdit<double> C1 =
        C.Edit(il::Range{n01, n01 + n11}, il::Range{0, C.size(1)});
    il::blas(alpha, A, s00, il::Dot::Transpose, B0, beta, il::io, C0);
    il::blas(alpha, A, s10, il::Dot::Transpose, B1, 1.0, il::io, C0);
    il::blas(alpha, A, s01, il::Dot::Transpose, B0, beta, il::io, C1);
    il::blas(alpha, A, s11, il::Dot::Transpose, B1, 1.0, il::io, C1);
  } else {
    IL_UNREACHABLE;
  }
}

void blas(double alpha, il::Array2DView<double> A, const il::HMatrix<double>& B,
          il::spot_t s, double beta, il::io_t, il::Array2DEdit<double> C) {
  IL_EXPECT_FAST(A.size(0) == C.size(0));
  IL_EXPECT_FAST(A.size(1) == B.size(0, s));
  IL_EXPECT_FAST(B.size(1, s) == C.size(1));

  if (B.isFullRank(s)) {
    il::Array2DView<double> b = B.asFullRank(s);
    il::blas(alpha, A, b, beta, il::io, C);
  } else if (B.isLowRank(s)) {
    il::Array2DView<double> ba = B.asLowRankA(s);
    il::Array2DView<double> bb = B.asLowRankB(s);
    const il::int_t r = ba.size(1);
    il::Array2D<double> tmp{A.size(0), r};
    il::blas(1.0, A, ba, 0.0, il::io, tmp.Edit());
    il::blas(alpha, tmp.view(), bb, il::Dot::Transpose, beta, il::io, C);
  } else if (B.isHierarchical(s)) {
    const il::spot_t s00 = B.child(s, 0, 0);
    const il::spot_t s10 = B.child(s, 1, 0);
    const il::spot_t s01 = B.child(s, 0, 1);
    const il::spot_t s11 = B.child(s, 1, 1);
    const il::int_t n00 = B.size(0, s00);
    const il::int_t n10 = B.size(0, s10);
    const il::int_t n01 = B.size(1, s00);
    const il::int_t n11 = B.size(1, s01);
    il::Array2DView<double> A0 =
        A.view(il::Range{0, A.size(0)}, il::Range{0, n00});
    il::Array2DView<double> A1 =
        A.view(il::Range{0, A.size(0)}, il::Range{n00 + n10});
    il::Array2DEdit<double> C0 =
        C0.Edit(il::Range{0, C.size(0)}, il::Range{0, n01});
    il::Array2DEdit<double> C1 =
        C1.Edit(il::Range{0, C.size(0)}, il::Range{n01 + n11});
    il::blas(alpha, A0, B, s00, beta, il::io, C0);
    il::blas(alpha, A1, B, s10, beta, il::io, C0);
    il::blas(alpha, A0, B, s01, beta, il::io, C1);
    il::blas(alpha, A1, B, s11, beta, il::io, C1);
  } else {
    IL_UNREACHABLE;
  }
}

void blasLowRank(double alpha, il::Array2DView<double> A,
                 il::Array2DView<double> B, double beta, il::spot_t s, il::io_t,
                 il::HMatrix<double>& C) {
  IL_EXPECT_FAST(A.size(1) == B.size(1));
  IL_EXPECT_FAST(A.size(0) == C.size(0, s));
  IL_EXPECT_FAST(B.size(0) == C.size(1, s));

  if (C.isFullRank(s)) {
    il::Array2DEdit<double> c = C.AsFullRank(s);
    il::blas(alpha, A, B, il::Dot::Transpose, beta, il::io, c);
  } else if (C.isLowRank(s)) {
    const il::int_t r0 = C.rankOfLowRank(s);
    const il::int_t r1 = A.size(1);
    C.UpdateRank(s, r0 + r1);
    il::Array2DEdit<double> ca = C.AsLowRankA(s);
    il::Array2DEdit<double> cb = C.AsLowRankB(s);
    il::Array2DEdit<double> cb_old =
        cb.Edit(il::Range{0, cb.size(0)}, il::Range{0, r0});
    for (il::int_t i1 = 0; i1 < cb_old.size(1); ++i1) {
      for (il::int_t i0 = 0; i0 < cb_old.size(0); ++i0) {
        cb_old(i0, i1) *= beta;
      }
    }
    il::Array2DEdit<double> ca_new =
        ca.Edit(il::Range{0, ca.size(0)}, il::Range{r0, r0 + r1});
    il::Array2DEdit<double> cb_new =
        cb.Edit(il::Range{0, cb.size(0)}, il::Range{r0, r0 + r1});
    il::copy(A, il::io, ca_new);
    il::copy(B, il::io, cb_new);
    for (il::int_t i1 = 0; i1 < cb_new.size(1); ++i1) {
      for (il::int_t i0 = 0; i0 < cb_new.size(0); ++i0) {
        cb_new(i0, i1) *= alpha;
      }
    }
  } else if (C.isHierarchical(s)) {
    const il::spot_t s00 = C.child(s, 0, 0);
    const il::spot_t s10 = C.child(s, 1, 0);
    const il::spot_t s01 = C.child(s, 0, 1);
    const il::spot_t s11 = C.child(s, 1, 1);
    const il::int_t n00 = C.size(0, s00);
    const il::int_t n10 = C.size(0, s10);
    const il::int_t n01 = C.size(1, s00);
    const il::int_t n11 = C.size(1, s01);
    const il::int_t r = A.size(1);
    il::blasLowRank(alpha, A.view(il::Range{0, n00}, il::Range{0, r}),
                    B.view(il::Range{0, n01}, il::Range{0, r}), beta, s00,
                    il::io, C);
    il::blasLowRank(alpha, A.view(il::Range{0, n00}, il::Range{0, r}),
                    B.view(il::Range{n01, n01 + n11}, il::Range{0, r}), beta,
                    s01, il::io, C);
    il::blasLowRank(alpha, A.view(il::Range{n00, n00 + n10}, il::Range{0, r}),
                    B.view(il::Range{0, n01}, il::Range{0, r}), beta, s10,
                    il::io, C);
    il::blasLowRank(alpha, A.view(il::Range{n00, n00 + n10}, il::Range{0, r}),
                    B.view(il::Range{n01, n01 + n11}, il::Range{0, r}), beta,
                    s11, il::io, C);
  } else {
    IL_UNREACHABLE;
  }
}

void blas_rec(double alpha, const il::HMatrix<double>& A, il::spot_t s,
              il::MatrixType type, il::ArrayView<double> x, double beta,
              il::io_t, il::ArrayEdit<double> y) {
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

void blas(double alpha, const il::HMatrix<double>& lu, il::spot_t s,
          il::MatrixType type, il::ArrayView<double> x, double beta, il::io_t,
          il::ArrayEdit<double> y) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == y.size());
  IL_EXPECT_MEDIUM(lu.size(1, s) == x.size());

  il::blas_rec(alpha, lu, s, type, x, beta, il::io, y);
}

void blas_rec(double alpha, const il::HMatrix<double>& A, il::spot_t s,
              il::MatrixType type, il::Array2DView<double> B, double beta,
              il::io_t, il::Array2DEdit<double> C) {
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
      // FIXME: Are we really going there?????
      // What the point of this il::MatrixType::LowerUnit anyway???
      // Anyway, I believe that the coefficients are wrong
      il::abort();
      il::blas_rec(alpha, A, s00, il::MatrixType::LowerUnit, B0, beta, il::io,
                   C0);
      il::blas_rec(alpha, A, s10, il::MatrixType::Regular, B0, beta, il::io,
                   C1);
      il::blas_rec(alpha, A, s11, il::MatrixType::LowerUnit, B1, beta, il::io,
                   C1);
    } else if (type == il::MatrixType::Regular) {
      il::blas_rec(alpha, A, s00, il::MatrixType::Regular, B0, beta, il::io,
                   C0);
      il::blas_rec(alpha, A, s01, il::MatrixType::Regular, B1, 1.0, il::io, C0);
      il::blas_rec(alpha, A, s10, il::MatrixType::Regular, B0, beta, il::io,
                   C1);
      il::blas_rec(alpha, A, s11, il::MatrixType::Regular, B1, 1.0, il::io, C1);
    }
    return;
  } else {
    IL_UNREACHABLE;
  }
}

void blas(double alpha, const il::HMatrix<double>& lu, il::spot_t s,
          il::MatrixType type, il::Array2DView<double> A, double beta, il::io_t,
          il::Array2DEdit<double> B) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == B.size(0));
  IL_EXPECT_MEDIUM(lu.size(1, s) == A.size(0));

  il::blas_rec(alpha, lu, s, type, A, beta, il::io, B);
}

}  // namespace il
