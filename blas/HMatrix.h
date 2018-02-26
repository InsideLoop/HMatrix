#ifndef HMAT_HMATRIX_H
#define HMAT_HMATRIX_H

#include <il/Array2D.h>
#include <il/linearAlgebra/dense/blas/blas.h>

namespace hmat {

template <typename T>
struct LowRank {
  il::Array2D<T> A;
  il::Array2D<T> B;

  il::int_t size(il::int_t d) const;
};

template <typename T>
il::int_t LowRank<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(d >= 0 && d < 2);

  return d == 0 ? A.size(0) : B.size(1);
}

template <typename T>
class HMatrix {
 private:
  il::int_t n0_;
  il::int_t n1_;

 public:
  il::int_t size(il::int_t d) const;
};

template <typename T>
il::int_t HMatrix<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(d >= 0 && d < 2);

  return d == 0 ? n0_ : n1_;
}

void blas(const il::Array<double> &x, const HMatrix<double> &H, il::io_t,
          il::Array<double> &y);
void blas(const HMatrix<double> &H, const il::Array<double> &x, il::io_t,
          il::Array<double> &y);

// We want to define the following operation: H0 = H0 + H1 x H2
// for different hierarchical matrices

void blas(const il::Array2D<double> &H1, const il::Array2D<double> &H2,
          il::io_t, il::Array2D<double> &H0) {
  IL_EXPECT_FAST(H0.size(0) == H1.size(0));
  IL_EXPECT_FAST(H1.size(1) == H2.size(0));
  IL_EXPECT_FAST(H2.size(1) == H0.size(1));

  il::blas(1.0, H1, H2, 1.0, il::io, H0);
}

void blas(const il::Array2D<double> &H1, const hmat::LowRank<double> &H2,
          il::io_t, il::Array2D<double> &H0) {
  IL_EXPECT_FAST(H0.size(0) == H1.size(0));
  IL_EXPECT_FAST(H1.size(1) == H2.size(0));
  IL_EXPECT_FAST(H2.size(1) == H0.size(1));

  il::Array2D<double> tmp{H2.size(0), H2.size(1), 0.0};
  il::blas(1.0, H2.A, H2.B, 0.0, il::io, tmp);
  il::blas(1.0, H1, tmp, 1.0, il::io, H0);
}

void blas(const hmat::LowRank<double> &H1, const il::Array2D<double> &H2,
          il::io_t, il::Array2D<double> &H0) {
  IL_EXPECT_FAST(H0.size(0) == H1.size(0));
  IL_EXPECT_FAST(H1.size(1) == H2.size(0));
  IL_EXPECT_FAST(H2.size(1) == H0.size(1));

  il::Array2D<double> tmp{H1.size(0), H1.size(1), 0.0};
  il::blas(1.0, H1.A, H1.B, 0.0, il::io, tmp);
  il::blas(1.0, tmp, H2, 1.0, il::io, H0);
}

void blas(const hmat::LowRank<double> &H1, const hmat::LowRank<double> &H2,
          il::io_t, il::Array2D<double> &H0) {
  IL_EXPECT_FAST(H0.size(0) == H1.size(0));
  IL_EXPECT_FAST(H1.size(1) == H2.size(0));
  IL_EXPECT_FAST(H2.size(1) == H0.size(1));

  il::Array2D<double> tmp0{H1.B.size(0), H2.A.size(1), 0.0};
  il::blas(1.0, H1.B, H2.A, 0.0, il::io, tmp0);
  il::Array2D<double> tmp1{H1.A.size(0), H2.A.size(1), 0.0};
  il::blas(1.0, H1.A, tmp0, 0.0, il::io, tmp1);
  il::blas(1.0, tmp1, H2.B, 0.0, il::io, H0);
}

void blas(const il::Array2D<double> &H1, const hmat::HMatrix<double> &H2,
          il::io_t, il::Array2D<double> &H0) {
  IL_EXPECT_FAST(H0.size(0) == H1.size(0));
  IL_EXPECT_FAST(H1.size(1) == H2.size(0));
  IL_EXPECT_FAST(H2.size(1) == H0.size(1));

  il::Array2D<double> tmp{H1.size(0), H2.size(1), 0.0};
  for (il::int_t i0 = 0; i0 < tmp.size(0); ++i0) {
    il::Array<double> row{H1.size(1), 0.0};
    for (il::int_t i1 = 0; i1 < H1.size(1); ++i1) {
      row[i1] = H1(i0, i1);
    }
    il::Array<double> hrow{H2.size(1), 0.0};
    hmat::blas(row, H2, il::io, hrow);
    for (il::int_t i1 = 0; i1 < tmp.size(1); ++i1) {
      tmp(i0, i1) = hrow[i1];
    }
  }

  il::blas(1.0, tmp, 1.0, il::io, H0);
}

void blas(const hmat::HMatrix<double> &H1, const il::Array2D<double> &H2,
          il::io_t, il::Array2D<double> &H0) {
  IL_EXPECT_FAST(H0.size(0) == H1.size(0));
  IL_EXPECT_FAST(H1.size(1) == H2.size(0));
  IL_EXPECT_FAST(H2.size(1) == H0.size(1));

  il::Array2D<double> tmp{H1.size(0), H2.size(1), 0.0};
  for (il::int_t i1 = 0; i1 < tmp.size(1); ++i1) {
    il::Array<double> column{H2.size(0), 0.0};
    for (il::int_t i0 = 0; i0 < H2.size(0); ++i0) {
      column[i0] = H2(i0, i1);
    }
    il::Array<double> hcolumn{H1.size(0), 0.0};
    hmat::blas(H1, column, il::io, hcolumn);
    for (il::int_t i0 = 0; i0 < tmp.size(0); ++i0) {
      tmp(i0, i1) = hcolumn[i1];
    }
  }

  il::blas(1.0, tmp, 1.0, il::io, H0);
}

void blas(const hmat::LowRank<double> &H1, const hmat::HMatrix<double> &H2,
          il::io_t, il::Array2D<double> &H0) {
  IL_EXPECT_FAST(H0.size(0) == H1.size(0));
  IL_EXPECT_FAST(H1.size(1) == H2.size(0));
  IL_EXPECT_FAST(H2.size(1) == H0.size(1));


}



}  // namespace hmat

#endif  // HMAT_HMATRIX_H
