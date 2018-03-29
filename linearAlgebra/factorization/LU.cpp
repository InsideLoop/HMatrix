//#include "LU.h"
//#include "OldHMatrix.h"

//#ifdef IL_MKL
//#include <mkl_lapacke.h>
//#elif IL_OPENBLAS
//#include <OpenBLAS/lapacke.h>
//#endif
//
//namespace il {
//
//LU<il::OldHMatrix<double>>::LU(il::OldHMatrix<double> A, il::io_t, il::Status& status) {
//  A_ = std::move(A);
//
//  if (A_.isFullRank()) {
//    il::Array2D<double>& full_A = A_.Full();
//    const int layout = LAPACK_COL_MAJOR;
//    const lapack_int m = static_cast<lapack_int>(full_A.size(0));
//    const lapack_int n = static_cast<lapack_int>(full_A.size(1));
//    const lapack_int lda = static_cast<lapack_int>(full_A.capacity(0));
//    il::Array<lapack_int> ipiv{il::min(full_A.size(0), full_A.size(1))};
//    const lapack_int lapack_error =
//        LAPACKE_dgetrf(layout, m, n, full_A.Data(), lda, ipiv.Data());
//
//    IL_EXPECT_FAST(lapack_error == 0);
//    // We should find out where we store the pivots for the full matrix as
//    // there might be some pivoting
//  } else {
//    // Handle the general case
//  }
//
//}
//
//}