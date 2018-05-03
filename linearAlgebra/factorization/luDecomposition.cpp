
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
    const il::spot_t s01 = H.child(s, 0, 1);
    const il::spot_t s10 = H.child(s, 1, 0);
    const il::spot_t s11 = H.child(s, 1, 1);
    luDecomposition(s00, il::io, H);
    il::solveLower(H, s00, s01, il::io, H);
    il::solveUpperRight(H, s00, s10, il::io, H);
    il::blas(-1.0, H, s10, H, s01, 1.0, s11, il::io, H);
    il::luDecomposition(s11, il::io, H);
  }
}

}  // namespace il
