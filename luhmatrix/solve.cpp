#include <luhmatrix/blas.h>
#include <luhmatrix/solve.h>

namespace il {

void solve(const il::LuHMatrix<double, int>& lu, il::spot_t s, il::io_t,
           il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(1, s) == x.size());

  solveLower(lu, s, il::io, x);
  solveUpper(lu, s, il::io, x);
}

void solveLower(const il::LuHMatrix<double, int>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(1, s) == x.size());

  if (lu.isFullRank(s)) {
    il::ArrayView<int> full_pivot = lu.asFullRankPivot(s);
    il::Array2DView<double> full_lu = lu.asFullRank(s);
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

void solveUpper(const il::LuHMatrix<double, int>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(1, s) == x.size());

  if (lu.isFullRank(s)) {
    il::Array2DView<double> full_lu = lu.asFullRank(s);
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

}  // namespace il