
#include <il/blas.h>
#include <il/linearAlgebra/dense/blas/solve.h>
#include <il/linearAlgebra/dense/factorization/luDecomposition.h>
#include <linearAlgebra/blas/hblas.h>
#include <linearAlgebra/blas/hsolve.h>
#include <linearAlgebra/factorization/luDecomposition.h>

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

void luDecomposition(il::io_t, il::HMatrix<double>& H) {
  luDecomposition(H.root(), il::io, H);
}
void luDecomposition(il::spot_t s, il::io_t, il::HMatrix<double>& H) {
  if (H.isFullRank(s)) {
    H.ConvertToFullLu(s);
    il::ArrayEdit<int> pivot = H.AsFullLuPivot(s);
    il::Array2DEdit<double> lu = H.AsFullLu(s);
    il::luDecomposition(il::io, pivot, lu);
  } else if (H.isHierarchical(s)) {
    const il::spot_t s00 = H.child(s, 0, 0);
    luDecomposition(s00, il::io, H);
    upperRight(s, il::io, H);
    lowerLeft(s, il::io, H);
    lowerRight(s, il::io, H);
  }
}

void upperRight(il::spot_t s, il::io_t, il::HMatrix<double>& H) {
  const il::spot_t s00 = H.child(s, 0, 0);
  const il::spot_t s01 = H.child(s, 0, 1);
  if (H.isFullLu(s00)) {
    il::ArrayView<int> PLU00 = H.asFullLuPivot(s00);
    il::Array2DView<double> LLU00 = H.asFullLu(s00);
    if (H.isLowRank(s01)) {
      // We need to solve P.L.U = A.B^T. The solution is
      // U = (L^{-1}.P^{-1}.A).B^T . So B is just copied, and for the new A,
      // all we need is to swap the rows (apply P^{-1}) and then solve the
      // lower triangular system.
      il::Array2DEdit<double> LUA = H.AsLowRankA(s01);
      il::Array2DEdit<double> LUB = H.AsLowRankB(s01);
      il::solve(PLU00, il::io, LUA);
      il::solve(LLU00, il::MatrixType::LowerUnit, il::io, LUA);
    } else {
      IL_UNREACHABLE;
    }
  } else if (H.isHierarchical(s00)) {
    const il::spot_t s00 = H.child(s, 0, 0);
    il::solveLower(H, s00, s01, il::io, H);
  } else {
    IL_EXPECT_MEDIUM(H.isLowRank(s00));
    IL_UNREACHABLE;
  }
}

void lowerLeft(il::spot_t s, il::io_t, il::HMatrix<double>& H) {
  const il::spot_t s00 = H.child(s, 0, 0);
  const il::spot_t s10 = H.child(s, 1, 0);
  if (H.isFullLu(s00)) {
    il::Array2DView<double> ULU00 = H.asFullLu(s00);
    if (H.isLowRank(s10)) {
      // We need to solve L.U = A.B^{T}. The solution is L = A.B^{T}.U^{-1}
      // which is L = A.((U^T)^{-1}.B)^T
      il::Array2DEdit<double> LUA = H.AsLowRankA(s10);
      il::Array2DEdit<double> LUB = H.AsLowRankB(s10);
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
  } else if (H.isHierarchical(s00)) {
    il::solveUpperTranspose(H, s00, s10, il::io, H);
  } else {
    IL_UNREACHABLE;
  }
}

void lowerRight(il::spot_t s, il::io_t, il::HMatrix<double>& H) {
  const il::spot_t s00 = H.child(s, 0, 0);
  if (H.isFullLu(s00)) {
    const il::spot_t s11 = H.child(s, 1, 1);
    const il::spot_t s01 = H.child(s, 0, 1);
    const il::spot_t s10 = H.child(s, 1, 0);
    IL_EXPECT_MEDIUM(H.isLowRank(s01) && H.isLowRank(s10));
    if (H.isFullRank(s11)) {
      // We need to solve P.L.U = H - L.U
      il::Array2DEdit<double> A = H.AsFullRank(s11);
      il::Array2DView<double> LA = H.asLowRankA(s10);
      il::Array2DView<double> LB = H.asLowRankB(s10);
      il::Array2DView<double> UA = H.asLowRankA(s01);
      il::Array2DView<double> UB = H.asLowRankB(s01);
      il::Array2D<double> tmp{LB.size(1), UA.size(1)};
      il::blas(1.0, LB, il::Dot::Transpose, UA, 0.0, il::io, tmp.Edit());
      il::Array2D<double> tmp2{LB.size(1), UB.size(0)};
      il::blas(1.0, tmp.view(), UB, il::Dot::Transpose, 0.0, il::io,
               tmp2.Edit());
      il::blas(-1.0, LA, tmp2.view(), 1.0, il::io, A);
      H.ConvertToFullLu(s11);
      il::ArrayEdit<int> pivot = H.AsFullLuPivot(s11);
      il::Array2DEdit<double> Abis = H.AsFullLu(s11);
      il::luDecomposition(il::io, pivot, Abis);
    } else {
      IL_UNREACHABLE;
    }
  } else {
    const il::spot_t s10 = H.child(s, 1, 0);
    const il::spot_t s01 = H.child(s, 0, 1);
    const il::spot_t s11 = H.child(s, 1, 1);
    il::blas(-1.0, H, s10, H, s01, 1.0, s11, il::io, H);
    il::luDecomposition(s11, il::io, H);
  }
}

//void copy(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu, il::io_t,
//          il::HMatrix<double>& LU) {
//  if (H.isFullRank(sh)) {
//    il::Array2DView<double> a = H.asFullRank(sh);
//    LU.SetFullRank(slu, a.size(0), a.size(1));
//    il::copy(a, il::io, LU.AsFullRank(slu));
//  } else if (H.isLowRank(sh)) {
//    il::Array2DView<double> a = H.asLowRankA(sh);
//    il::Array2DView<double> b = H.asLowRankB(sh);
//    LU.SetLowRank(slu, a.size(0), b.size(0), a.size(1));
//    il::copy(a, il::io, LU.AsLowRankA(slu));
//    il::copy(b, il::io, LU.AsLowRankB(slu));
//  } else {
//    IL_EXPECT_MEDIUM(H.isHierarchical(sh));
//    LU.SetHierarchical(slu);
//    const il::spot_t sh00 = H.child(sh, 0, 0);
//    const il::spot_t slu00 = LU.child(slu, 0, 0);
//    il::copy(H, sh00, slu00, il::io, LU);
//    const il::spot_t sh01 = H.child(sh, 0, 1);
//    const il::spot_t slu01 = LU.child(slu, 0, 1);
//    il::copy(H, sh01, slu01, il::io, LU);
//    const il::spot_t sh10 = H.child(sh, 1, 0);
//    const il::spot_t slu10 = LU.child(slu, 1, 0);
//    il::copy(H, sh10, slu10, il::io, LU);
//    const il::spot_t sh11 = H.child(sh, 1, 1);
//    const il::spot_t slu11 = LU.child(slu, 1, 1);
//    il::copy(H, sh11, slu11, il::io, LU);
//  }
//}

}  // namespace il
