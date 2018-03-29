#pragma once

#include <il/Status.h>

#include <HMatrix/HMatrix.h>
#include <HMatrix/LUHMatrix.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif
#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif
namespace il {

template <typename T>
void recursive_lu(const il::HMatrix<T>& a, il::int_t k, il::io_t,
                  il::LUHMatrix<T>& lu);

template <typename T>
void recursive_solve_ltl(const il::HMatrix<T>& a, il::int_t s00,
                         il::int_t s01, il::io_t, il::LUHMatrix<T>& lu);

template <typename T>
void recursive_solve_ltr(const il::HMatrix<T>& a, il::int_t s00,
                         il::int_t s01, il::io_t, il::LUHMatrix<T>& lu);

template <typename T>
void recursive_blas(const il::HMatrix<T>& a, il::int_t s01, il::int_t s10,
                    il::int_t s11, il::io_t, il::LUHMatrix<T>& lu);

template <typename T>
il::LUHMatrix<T> lu(const il::HMatrix<T>& A, il::io_t, il::Status& status) {
  il::LUHMatrix<T> ans{};
  il::recursive_lu<T>(A, 0, il::io, ans);

  status.SetOk();
  return ans;
}

template <typename T>
void recursive_lu(const il::HMatrix<T>& a, il::int_t k, il::io_t,
                  il::LUHMatrix<T>& lu) {
  if (a.isFullRank(k)) {
    il::Array2DView<T> full = a.asFullRank(k);
    const il::int_t n0 = full.size(0);
    const il::int_t n1 = full.size(1);

    lu.SetFullRank(k, n0, n1);
    il::Array2DEdit<T> full_lu = lu.AsFullRank(k);
    il::ArrayEdit<int> full_pivot = lu.AsPivotFull(k);
    for (il::int_t i1 = 0; i1 < n0; ++i1) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        full_lu(i0, i1) = full(i0, i1);
      }
    }

    const int layout = LAPACK_COL_MAJOR;
    const lapack_int m = static_cast<lapack_int>(n0);
    const lapack_int n = static_cast<lapack_int>(n1);
    const lapack_int lda = static_cast<lapack_int>(full.stride(1));

    const lapack_int lapack_error =
        LAPACKE_dgetrf(layout, m, n, full_lu.Data(), lda, full_pivot.Data());
    IL_EXPECT_FAST(lapack_error == 0);
  } else {
    const il::int_t s00 = a.child(k, 0, 0);
    il::recursive_lu(a, s00, il::io, lu);

    const il::int_t s01 = a.child(k, 0, 1);
    il::recursive_solve_ltl(a, s00, s01, il::io, lu);

    const il::int_t s10 = a.child(k, 0, 0);
    il::recursive_solve_ltr(a, s00, s01, il::io, lu);

    const il::int_t s11 = a.child(k, 1, 1);
    il::recursive_blas(a, s01, s10, s11, il::io, lu);
  }
}

template <typename T>
void recursive_solve_ltl(const il::HMatrix<T>& a, il::int_t s00,
                         il::int_t s01, il::io_t, il::LUHMatrix<T>& lu) {

}

template <typename T>
void recursive_solve_ltr(const il::HMatrix<T>& a, il::int_t s00,
                         il::int_t s01, il::io_t, il::LUHMatrix<T>& lu) {

}

template <typename T>
void recursive_blas(const il::HMatrix<T>& a, il::int_t s01, il::int_t s10,
                    il::int_t s11, il::io_t, il::LUHMatrix<T>& lu) {

}

}  // namespace il