#include <luhmatrix/lu.h>

#include <il/blas.h>

#ifdef IL_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#define IL_CBLAS_INT MKL_INT
#define IL_CBLAS_LAYOUT CBLAS_LAYOUT
#elif IL_OPENBLAS
#include <OpenBLAS/cblas.h>
#include <OpenBLAS/lapacke.h>
#define IL_CBLAS_INT int
#define IL_CBLAS_LAYOUT CBLAS_ORDER
#endif

namespace il {

il::LuHMatrix<double, int> lu(const il::HMatrix<double>& H) {
  il::LuHMatrix<double, int> LU{};
  il::spot_t sh = H.root();
  il::spot_t slu = LU.root();
  lu(H, sh, slu, il::io, LU);
  return LU;
};

void lu(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu, il::io_t,
        il::LuHMatrix<double, int>& LU) {
  if (H.isFullRank(sh)) {
    il::Array2DView<double> F = H.asFullRank(sh);
    LU.SetFullRank(slu, F.size(0), F.size(1));
    il::Array2DEdit<double> lu = LU.AsFullRank(slu);
    il::ArrayEdit<int> pivot = LU.AsFullRankPivot(slu);
    il::copy(F, il::io, lu);
    il::luForFull(il::io, lu, pivot);
  } else if (H.isHierarchical(sh)) {
    const il::spot_t sh00 = H.child(sh, 0, 0);
    const il::spot_t sh10 = H.child(sh, 1, 0);
    const il::spot_t sh01 = H.child(sh, 0, 1);
    const il::spot_t sh11 = H.child(sh, 1, 1);

    LU.SetHierarchical(slu);
    const il::spot_t slu00 = LU.child(slu, 0, 0);
    const il::spot_t slu10 = LU.child(slu, 1, 0);
    const il::spot_t slu01 = LU.child(slu, 0, 1);
    const il::spot_t slu11 = LU.child(slu, 1, 1);
    lu(H, sh00, slu00, il::io, LU);
    upperRight(H, sh, slu, il::io, LU);
    lowerLeft(H, sh, slu, il::io, LU);
    lowerRight(H, sh, slu, il::io, LU);
  }
}

void luForFull(il::io_t, il::Array2DEdit<double> A, il::ArrayEdit<int> pivot) {
  IL_EXPECT_MEDIUM(A.size(0) == A.size(1));
  IL_EXPECT_MEDIUM(A.size(0) == pivot.size());

  const int layout = LAPACK_COL_MAJOR;
  const lapack_int m = static_cast<lapack_int>(A.size(0));
  const lapack_int n = static_cast<lapack_int>(A.size(1));
  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
  const lapack_int lapack_error =
      LAPACKE_dgetrf(layout, m, n, A.Data(), lda, pivot.Data());

  IL_EXPECT_FAST(lapack_error == 0);
}

void upperRight(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu,
                il::io_t, il::LuHMatrix<double, int>& LU) {
  const il::spot_t sh00 = H.child(sh, 0, 0);
  if (H.isFullRank(sh00)) {
    const il::spot_t sh01 = H.child(sh, 0, 1);
    const il::spot_t slu00 = LU.child(slu, 0, 0);
    const il::spot_t slu01 = LU.child(slu, 0, 1);
    IL_EXPECT_MEDIUM(LU.isFullRank(slu00))
    il::ArrayView<int> PLU00 = LU.asFullRankPivot(slu00);
    il::Array2DView<double> LLU00 = LU.asFullRank(slu00);
    if (H.isLowRank(sh01)) {
      // We need to solve P.L.U = A.B^T. The solution is
      // U = (L^{-1}.P^{-1}.A).B^T . So B is just copied, and for the new A,
      // all we need is to swap the rows (apply P^{-1}) and then solve the
      // lower triangular system.
      il::Array2DView<double> A = H.asLowRankA(sh01);
      il::Array2DView<double> B = H.asLowRankB(sh01);
      IL_EXPECT_MEDIUM(A.size(1) == B.size(1))
      LU.SetLowRank(slu01, A.size(0), B.size(0), A.size(1));
      il::Array2DEdit<double> LUA = LU.AsLowRankA(slu01);
      il::Array2DEdit<double> LUB = LU.AsLowRankB(slu01);
      il::copy(B, il::io, LUB);
      il::copy(A, il::io, LUA);
      {
        const int layout = LAPACK_COL_MAJOR;
        const lapack_int n = static_cast<lapack_int>(LUA.size(1));
        const lapack_int lda = static_cast<lapack_int>(LUA.stride(1));
        const lapack_int k1 = 1;
        const lapack_int k2 = PLU00.size();
        const lapack_int incx = 1;
        const lapack_int lapack_error = LAPACKE_dlaswp(
            layout, n, LUA.Data(), lda, k1, k2, PLU00.data(), incx);
        IL_EXPECT_MEDIUM(lapack_error == 0);
      }

      {
        const IL_CBLAS_LAYOUT layout = CblasColMajor;
        const CBLAS_SIDE side = CblasLeft;
        const CBLAS_UPLO uplo = CblasLower;
        const CBLAS_TRANSPOSE transa = CblasNoTrans;
        const CBLAS_DIAG diag = CblasUnit;
        const MKL_INT m = static_cast<MKL_INT>(LLU00.size(0));
        const MKL_INT n = static_cast<MKL_INT>(LLU00.size(1));
        const double alpha = 1.0;
        const MKL_INT lda = static_cast<MKL_INT>(LLU00.stride(1));
        const MKL_INT ldb = static_cast<MKL_INT>(LUA.stride(1));
        cblas_dtrsm(layout, side, uplo, transa, diag, m, n, alpha, LLU00.data(),
                    lda, LUA.Data(), ldb);
      }
    } else {
      IL_UNREACHABLE;
    }
  } else {
    IL_UNREACHABLE;
  }
}

void lowerLeft(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu,
               il::io_t, il::LuHMatrix<double, int>& LU) {
  const il::spot_t sh00 = H.child(sh, 0, 0);
  if (H.isFullRank(sh00)) {
    const il::spot_t sh10 = H.child(sh, 1, 0);
    const il::spot_t slu00 = LU.child(slu, 0, 0);
    const il::spot_t slu10 = LU.child(slu, 1, 0);
    IL_EXPECT_MEDIUM(LU.isFullRank(slu00))
    il::Array2DView<double> ULU00 = LU.asFullRank(slu00);
    if (H.isLowRank(sh10)) {
      // We need to solve L.U = A.B^{T}. The solution is L = A.B^{T}.U^{-1}
      // which is L = A.((U^T)^{-1}.B)^T
      il::Array2DView<double> A = H.asLowRankA(sh10);
      il::Array2DView<double> B = H.asLowRankB(sh10);
      IL_EXPECT_MEDIUM(A.size(1) == B.size(1))
      LU.SetLowRank(slu10, A.size(0), B.size(0), A.size(1));
      il::Array2DEdit<double> LUA = LU.AsLowRankA(slu10);
      il::Array2DEdit<double> LUB = LU.AsLowRankB(slu10);
      il::copy(B, il::io, LUB);
      il::copy(A, il::io, LUA);
      const IL_CBLAS_LAYOUT layout = CblasColMajor;
      const CBLAS_SIDE side = CblasLeft;
      const CBLAS_UPLO uplo = CblasUpper;
      const CBLAS_TRANSPOSE transa = CblasTrans;
      const CBLAS_DIAG diag = CblasNonUnit;
      const MKL_INT m = static_cast<MKL_INT>(LUB.size(0));
      const MKL_INT n = static_cast<MKL_INT>(LUB.size(1));
      const double alpha = 1.0;
      const MKL_INT lda = static_cast<MKL_INT>(ULU00.stride(1));
      const MKL_INT ldb = static_cast<MKL_INT>(LUB.stride(1));
      cblas_dtrsm(layout, side, uplo, transa, diag, m, n, alpha, ULU00.data(),
                  lda, LUB.Data(), ldb);
    } else {
      IL_UNREACHABLE;
    }
  } else {
    IL_UNREACHABLE;
  }
}

void lowerRight(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu,
                il::io_t, il::LuHMatrix<double, int>& LU) {
  const il::spot_t sh00 = H.child(sh, 0, 0);
  if (H.isFullRank(sh00)) {
    const il::spot_t sh11 = H.child(sh, 1, 1);
    const il::spot_t slu01 = LU.child(slu, 0, 1);
    const il::spot_t slu10 = LU.child(slu, 1, 0);
    const il::spot_t slu11 = LU.child(slu, 1, 1);
    IL_EXPECT_MEDIUM(LU.isLowRank(slu01) && LU.isLowRank(slu10));
    if (H.isFullRank(sh11)) {
      // We need to solve P.L.U = H - L.U
      il::Array2DView<double> FH = H.asFullRank(sh11);
      IL_EXPECT_MEDIUM(FH.size(0) == FH.size(1));
      LU.SetFullRank(slu11, FH.size(0), FH.size(1));
      il::Array2DEdit<double> A = LU.AsFullRank(slu11);
      il::ArrayEdit<int> pivot = LU.AsFullRankPivot(slu11);
      il::Array2DView<double> LA = LU.asLowRankA(slu10);
      il::Array2DView<double> LB = LU.asLowRankB(slu10);
      il::Array2DView<double> UA = LU.asLowRankA(slu01);
      il::Array2DView<double> UB = LU.asLowRankB(slu01);
      il::Array2D<double> tmp{LB.size(1), UA.size(1)};
      il::blas(1.0, LB, il::Dot::Transpose, UA, 0.0, il::io, tmp.Edit());
      il::Array2D<double> tmp2{LB.size(1), UB.size(0)};
      il::blas(1.0, tmp.view(), UB, il::Dot::Transpose, 0.0, il::io,
               tmp2.Edit());
      il::blas(1.0, LA, tmp2.view(), 0.0, il::io, A);
      il::blas(1.0, FH, -1.0, il::io, A);
      il::luForFull(il::io, A, pivot);
    } else {
      IL_UNREACHABLE;
    }
  } else {
    IL_UNREACHABLE;
  }
}

void copy(il::Array2DView<double> A, il::io_t, il::Array2DEdit<double> B) {
  IL_EXPECT_MEDIUM(A.size(0) == B.size(0));
  IL_EXPECT_MEDIUM(A.size(1) == B.size(1));

  for (il::int_t i1 = 0; i1 < A.size(1); ++i1) {
    for (il::int_t i0 = 0; i0 < A.size(0); ++i0) {
      B(i0, i1) = A(i0, i1);
    }
  }
}

}  // namespace il