#include <luhmatrix/blas.h>
#include <luhmatrix/hblas.h>
#include <luhmatrix/solve.h>

namespace il {

void solve(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
           il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(1, s) == x.size());

  solveLower(lu, s, il::io, x);
  solveUpper(lu, s, il::io, x);
}

void solveLower(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(1, s) == x.size());

  if (lu.isFullLu(s)) {
    il::ArrayView<int> full_pivot = lu.asFullLuPivot(s);
    il::Array2DView<double> full_lu = lu.asFullLu(s);
    il::solve(full_pivot, full_lu, il::MatrixType::LowerUnit, il::io, x.Edit());
  } else if (lu.isHierarchical(s)) {
    const il::spot_t s00 = lu.child(s, 0, 0);
    const il::spot_t s10 = lu.child(s, 1, 0);
    const il::spot_t s11 = lu.child(s, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    solveLower(lu, s00, il::io, x.Edit(il::Range{0, n0}));
    il::blas(-1.0, lu, s10, il::MatrixType::Regular, x.view(il::Range{0, n0}),
             1.0, il::io, x.Edit(il::Range{n0, n0 + n1}));
    solveLower(lu, s11, il::io, x.Edit(il::Range{n0, n0 + n1}));
  } else {
    IL_UNREACHABLE;
  }
}

void solveLower(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::Array2DEdit<double> A) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(0, s) == A.size(0));

  if (lu.isFullLu(s)) {
    il::ArrayView<int> pivot = lu.asFullLuPivot(s);
    il::Array2DView<double> full = lu.asFullLu(s);
    il::solve(pivot, full, il::MatrixType::LowerUnit, il::io, A);
  } else if (lu.isHierarchical(s)) {
    const il::spot_t s00 = lu.child(s, 0, 0);
    const il::spot_t s10 = lu.child(s, 1, 0);
    const il::spot_t s11 = lu.child(s, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    solveLower(lu, s00, il::io,
               A.Edit(il::Range{0, n0}, il::Range{0, A.size(1)}));
    il::blas(-1.0, lu, s10, il::MatrixType::Regular,
             A.view(il::Range{0, n0}, il::Range{0, A.size(1)}), 1.0, il::io,
             A.Edit(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)}));
    solveLower(lu, s11, il::io,
               A.Edit(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)}));
  } else {
    IL_UNREACHABLE;
  }
}

void solveLower(const il::HMatrix<double>& lu, il::spot_t slu, il::spot_t s,
                il::io_t, il::HMatrix<double>& A) {
  const il::int_t debug_n0 = lu.size(0, slu);
  const il::int_t debug_n1 = lu.size(1, slu);
  IL_EXPECT_MEDIUM(lu.size(0, slu) == lu.size(1, slu));
  IL_EXPECT_MEDIUM(lu.size(1, slu) == A.size(0, s));

  if (lu.isFullRank(slu)) {
    il::ArrayView<int> pivot = lu.asFullLuPivot(slu);
    il::Array2DView<double> lower = lu.asFullRank(slu);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::solve(pivot, lower, il::MatrixType::LowerUnit, il::io, full);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> full = A.AsLowRankA(s);
      il::solve(pivot, lower, il::MatrixType::LowerUnit, il::io, full);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else if (lu.isHierarchical(slu)) {
    const il::spot_t s00 = lu.child(slu, 0, 0);
    const il::spot_t s10 = lu.child(slu, 1, 0);
    const il::spot_t s11 = lu.child(slu, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s10);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::Array2DEdit<double> full0 =
          full.Edit(il::Range{0, n0}, il::Range{0, full.size(1)});
      il::Array2DEdit<double> full1 =
          full.Edit(il::Range{n0, n0 + n1}, il::Range{0, full.size(1)});
      il::solveLower(lu, s00, il::io, full0);
      il::blas(-1.0, lu, s10, il::MatrixType::LowerUnit, full0, 1.0, il::io,
               full1);
      il::solveLower(lu, s11, il::io, full1);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> lowA = A.AsLowRankA(s);
      il::Array2DEdit<double> lowA0 =
          lowA.Edit(il::Range{0, n0}, il::Range{0, lowA.size(1)});
      il::Array2DEdit<double> lowA1 =
          lowA.Edit(il::Range{n0, n0 + n1}, il::Range{0, lowA.size(1)});
      il::solveLower(lu, s00, il::io, lowA0);
      il::blas(-1.0, lu, s10, il::MatrixType::Regular, lowA0, 1.0, il::io,
               lowA1);
      il::solveLower(lu, s11, il::io, lowA1);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else {
    IL_UNREACHABLE;
  }
}

void solveUpper(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(1, s) == x.size());

  if (lu.isFullLu(s)) {
    il::Array2DView<double> full_lu = lu.asFullLu(s);
    il::solve(full_lu, il::MatrixType::UpperNonUnit, il::io, x.Edit());
  } else if (lu.isHierarchical(s)) {
    const il::spot_t s00 = lu.child(s, 0, 0);
    const il::spot_t s01 = lu.child(s, 0, 1);
    const il::spot_t s11 = lu.child(s, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    solveUpper(lu, s11, il::io, x.Edit(il::Range{n0, n0 + n1}));
    il::blas(-1.0, lu, s01, il::MatrixType::Regular,
             x.view(il::Range{n0, n0 + n1}), 1.0, il::io,
             x.Edit(il::Range{0, n0}));
    solveUpper(lu, s00, il::io, x.Edit(il::Range{0, n0}));
  } else {
    IL_UNREACHABLE;
  }
}

void solveUpper(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::Array2DEdit<double> A) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(1, s) == A.size(0));

  if (lu.isFullLu(s)) {
    il::Array2DView<double> full_lu = lu.asFullLu(s);
    il::solve(full_lu, il::MatrixType::UpperNonUnit, il::io, A);
  } else if (lu.isHierarchical(s)) {
    const il::spot_t s00 = lu.child(s, 0, 0);
    const il::spot_t s01 = lu.child(s, 0, 1);
    const il::spot_t s11 = lu.child(s, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    solveUpper(lu, s11, il::io,
               A.Edit(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)}));
    il::blas(-1.0, lu, s01, il::MatrixType::Regular,
             A.view(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)}), 1.0,
             il::io, A.Edit(il::Range{0, n0}, il::Range{A.size(1)}));
    solveUpper(lu, s00, il::io,
               A.Edit(il::Range{0, n0}, il::Range{0, A.size(1)}));
  } else {
    IL_UNREACHABLE;
  }
}

void solveUpper(const il::HMatrix<double>& lu, il::spot_t slu, il::spot_t s,
                il::io_t, il::HMatrix<double>& A) {
  IL_EXPECT_MEDIUM(lu.size(0, slu) == lu.size(1, slu));
  IL_EXPECT_MEDIUM(lu.size(1, slu) == A.size(0, s));

  if (lu.isFullRank(slu)) {
    il::ArrayView<int> pivot = lu.asFullLuPivot(slu);
    il::Array2DView<double> upper = lu.asFullRank(slu);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::solve(upper, il::MatrixType::UpperNonUnit, il::io, full);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> full = A.AsLowRankA(s);
      il::solve(upper, il::MatrixType::UpperNonUnit, il::io, full);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else if (lu.isHierarchical(slu)) {
    const il::spot_t s00 = lu.child(slu, 0, 0);
    const il::spot_t s01 = lu.child(slu, 0, 1);
    const il::spot_t s11 = lu.child(slu, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::Array2DEdit<double> full0 =
          full.Edit(il::Range{0, n0}, il::Range{0, full.size(1)});
      il::Array2DEdit<double> full1 =
          full.Edit(il::Range{n0, n0 + n1}, il::Range{0, full.size(1)});

      il::solveUpper(lu, s11, il::io, full1);
      il::blas(-1.0, lu, s01, il::MatrixType::Regular, full1, 1.0, il::io,
               full0);
      il::solveUpper(lu, s00, il::io, full0);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> lowA = A.AsLowRankA(s);
      il::Array2DEdit<double> lowA0 =
          lowA.Edit(il::Range{0, n0}, il::Range{0, lowA.size(1)});
      il::Array2DEdit<double> lowA1 =
          lowA.Edit(il::Range{n0, n0 + n1}, il::Range{0, lowA.size(1)});
      il::solveUpper(lu, s11, il::io, lowA1);
      il::blas(-1.0, lu, s01, il::MatrixType::Regular, lowA1, 1.0, il::io,
               lowA0);
      il::solveLower(lu, s00, il::io, lowA0);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else {
    IL_UNREACHABLE;
  }
}

void solveUpperTranspose(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                         il::Array2DEdit<double> A) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(0, s) == A.size(0));

  if (lu.isFullLu(s)) {
    il::Array2DView<double> full_lu = lu.asFullLu(s);
    il::solve(full_lu, il::MatrixType::UpperNonUnit, il::Dot::Transpose, il::io,
              A);
  } else if (lu.isHierarchical(s)) {
    const il::spot_t s00 = lu.child(s, 0, 0);
    const il::spot_t s01 = lu.child(s, 0, 1);
    const il::spot_t s11 = lu.child(s, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    il::Array2DEdit<double> A0 =
        A.Edit(il::Range{0, n0}, il::Range{0, A.size(1)});
    il::Array2DEdit<double> A1 =
        A.Edit(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)});
    solveUpperTranspose(lu, s00, il::io, A0);
    il::blas(-1.0, lu, s01, il::Dot::Transpose, A0, 1.0, il::io, A1);
    solveUpperTranspose(lu, s11, il::io, A1);
  } else {
    IL_UNREACHABLE;
  }
}

void solveUpperTranspose(const il::HMatrix<double>& lu, il::spot_t slu,
                         il::spot_t s, il::io_t, il::HMatrix<double>& A) {
  IL_EXPECT_MEDIUM(lu.size(0, slu) == lu.size(1, slu));
  IL_EXPECT_MEDIUM(lu.size(0, slu) == A.size(0, s));

  if (lu.isFullLu(slu)) {
    il::Array2DView<double> upper = lu.asFullRank(slu);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::solve(upper, il::MatrixType::UpperNonUnit, il::Dot::Transpose, il::io,
                full);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> full = A.AsLowRankA(s);
      il::solve(upper, il::MatrixType::UpperNonUnit, il::Dot::Transpose, il::io,
                full);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else if (lu.isHierarchical(slu)) {
    const il::spot_t s00 = lu.child(slu, 0, 0);
    const il::spot_t s01 = lu.child(slu, 0, 1);
    const il::spot_t s11 = lu.child(slu, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::Array2DEdit<double> full0 =
          full.Edit(il::Range{0, n0}, il::Range{0, full.size(1)});
      il::Array2DEdit<double> full1 =
          full.Edit(il::Range{n0, n0 + n1}, il::Range{0, full.size(1)});

      il::solveUpperTranspose(lu, s00, il::io, full0);
      il::blas(-1.0, lu, s01, il::Dot::Transpose, full0, 1.0, il::io, full1);
      il::solveUpperTranspose(lu, s11, il::io, full1);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> lowb = A.AsLowRankB(s);
      il::Array2DEdit<double> lowb0 =
          lowb.Edit(il::Range{0, n0}, il::Range{0, lowb.size(1)});
      il::Array2DEdit<double> lowb1 =
          lowb.Edit(il::Range{n0, n0 + n1}, il::Range{0, lowb.size(1)});
      il::solveUpperTranspose(lu, s00, il::io, lowb0);
      il::blas(-1.0, lu, s01, il::Dot::Transpose, lowb0, 1.0, il::io, lowb1);
      il::solveUpperTranspose(lu, s11, il::io, lowb1);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else {
    IL_UNREACHABLE;
  }
}

void solve(il::ArrayView<int> pivot, il::Array2DView<double> A,
           il::MatrixType type, il::io_t, il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(pivot.size() == A.size(0));
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(x.size() == A.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::LowerUnit);

  {
    const int layout = LAPACK_COL_MAJOR;
    const lapack_int n = 1;
    const lapack_int lda = x.size();
    const lapack_int k1 = 1;
    const lapack_int k2 = x.size();
    const lapack_int incx = 1;
    const lapack_int lapack_error =
        LAPACKE_dlaswp(layout, n, x.Data(), lda, k1, k2, pivot.data(), incx);
    IL_EXPECT_MEDIUM(lapack_error == 0);
  }

  {
    const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasLower;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasUnit;
    const MKL_INT m = static_cast<MKL_INT>(x.size());
    const MKL_INT n = static_cast<MKL_INT>(1);
    const double alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(x.size());
    cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
                lda, x.Data(), ldb);
  }
}

void solve(il::ArrayView<int> pivot, il::Array2DView<double> A,
           il::MatrixType type, il::io_t, il::Array2DEdit<double> B) {
  IL_EXPECT_MEDIUM(pivot.size() == A.size(0));
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(A.size(0) == B.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::LowerUnit);

  {
    const int layout = LAPACK_COL_MAJOR;
    const lapack_int n = static_cast<lapack_int>(B.size(1));
    const lapack_int lda = static_cast<lapack_int>(B.stride(1));
    const lapack_int k1 = 1;
    const lapack_int k2 = pivot.size();
    const lapack_int incx = 1;
    const lapack_int lapack_error =
        LAPACKE_dlaswp(layout, n, B.Data(), lda, k1, k2, pivot.data(), incx);
    IL_EXPECT_MEDIUM(lapack_error == 0);
  }

  {
    const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
    const CBLAS_SIDE side = CblasLeft;
    const CBLAS_UPLO uplo = CblasLower;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_DIAG diag = CblasUnit;
    const MKL_INT m = static_cast<MKL_INT>(B.size(0));
    const MKL_INT n = static_cast<MKL_INT>(B.size(1));
    const double alpha = 1.0;
    const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
    const MKL_INT ldb = static_cast<MKL_INT>(B.stride(1));
    cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
                lda, B.Data(), ldb);
  }
}

void solve(il::Array2DView<double> A, il::MatrixType type, il::io_t,
           il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(x.size() == A.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::UpperNonUnit);

  const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
  const CBLAS_SIDE side = CblasLeft;
  const CBLAS_UPLO uplo = CblasUpper;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_DIAG diag = CblasNonUnit;
  const MKL_INT m = static_cast<MKL_INT>(x.size());
  const MKL_INT n = static_cast<MKL_INT>(1);
  const double alpha = 1.0;
  const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
  const MKL_INT ldb = static_cast<MKL_INT>(x.size());
  cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
              lda, x.Data(), ldb);
}

void solve(il::Array2DView<double> A, il::MatrixType type, il::io_t,
           il::Array2DEdit<double> B) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(B.size(0) == A.size(0));
  IL_EXPECT_MEDIUM(type == il::MatrixType::UpperNonUnit);

  const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
  const CBLAS_SIDE side = CblasLeft;
  const CBLAS_UPLO uplo = CblasUpper;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_DIAG diag = CblasNonUnit;
  const MKL_INT m = static_cast<MKL_INT>(B.size(0));
  const MKL_INT n = static_cast<MKL_INT>(B.size(1));
  const double alpha = 1.0;
  const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
  const MKL_INT ldb = static_cast<MKL_INT>(B.stride(1));
  cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
              lda, B.Data(), ldb);
}

void solve(il::Array2DView<double> A, il::MatrixType type, il::Dot op, il::io_t,
           il::Array2DEdit<double> B) {
  IL_EXPECT_FAST(A.size(0) == A.size(1));
  IL_EXPECT_FAST(B.size(0) == A.size(0));
  IL_EXPECT_FAST(type == il::MatrixType::UpperNonUnit);
  IL_EXPECT_FAST(op == il::Dot::Transpose);

  const IL_CBLAS_LAYOUT cblas_layout = CblasColMajor;
  const CBLAS_SIDE side = CblasLeft;
  const CBLAS_UPLO uplo = CblasUpper;
  const CBLAS_TRANSPOSE transa = CblasTrans;
  const CBLAS_DIAG diag = CblasNonUnit;
  const MKL_INT m = static_cast<MKL_INT>(B.size(0));
  const MKL_INT n = static_cast<MKL_INT>(B.size(1));
  const double alpha = 1.0;
  const MKL_INT lda = static_cast<MKL_INT>(A.stride(1));
  const MKL_INT ldb = static_cast<MKL_INT>(B.stride(1));
  cblas_dtrsm(cblas_layout, side, uplo, transa, diag, m, n, alpha, A.data(),
              lda, B.Data(), ldb);
}

}  // namespace il