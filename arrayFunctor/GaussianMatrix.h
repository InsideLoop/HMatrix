#pragma once

#include <il/math.h>

#include <arrayFunctor/MatrixGenerator.h>

namespace il {

template <typename T>
class GaussianMatrix : public MatrixGenerator<T> {
 private:
  il::int_t n_;
  il::Range range0_;
  il::Range range1_;
  double alpha_;

 public:
  GaussianMatrix(il::int_t n, double alpha);
  GaussianMatrix(il::int_t n, il::Range range0, il::Range range1, double alpha);
  il::int_t size(il::int_t d) const override;
  il::int_t blockSize() const override;
  il::int_t sizeAsBlocks(il::int_t d) const override;
  void set(il::int_t b0, il::int_t b1, il::io_t,
           il::Array2DEdit<double> M) const override;
};

template <typename T>
GaussianMatrix<T>::GaussianMatrix(il::int_t n, double alpha) {
  IL_EXPECT_MEDIUM(n >= 0);

  n_ = n;
  range0_ = il::Range{0, n};
  range1_ = il::Range{0, n};
  alpha_ = alpha;
};

template <typename T>
GaussianMatrix<T>::GaussianMatrix(il::int_t n, il::Range range0,
                                  il::Range range1, double alpha) {
  IL_EXPECT_MEDIUM(n >= 0);
  IL_EXPECT_MEDIUM(range0.begin >= 0 && range0.end <= n);
  IL_EXPECT_MEDIUM(range1.begin >= 0 && range1.end <= n);

  n_ = n;
  range0_ = range0;
  range1_ = range1;
  alpha_ = alpha;
};

template <typename T>
il::int_t GaussianMatrix<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(d == 0 || d == 1);

  switch (d) {
    case 0:
      return range0_.end - range0_.begin;
    case 1:
      return range1_.end - range1_.begin;
    default:
      IL_UNREACHABLE;
  }
  IL_UNREACHABLE;
  return -1;
};

template <typename T>
il::int_t GaussianMatrix<T>::blockSize() const {
  return 1;
}

template <typename T>
il::int_t GaussianMatrix<T>::sizeAsBlocks(il::int_t d) const {
  IL_EXPECT_MEDIUM(d == 0 || d == 1);

  return size(d);
}

template <typename T>
void GaussianMatrix<T>::set(il::int_t b0, il::int_t b1, il::io_t,
                            il::Array2DEdit<double> M) const {
  IL_EXPECT_MEDIUM(b0 + M.size(0) <= size(0));
  IL_EXPECT_MEDIUM(b1 + M.size(1) <= size(1));

  const double beta = il::ipow<2>(1.0 / (n_ - 1));
  for (il::int_t i1 = 0; i1 < M.size(1); ++i1) {
    il::int_t j1 = range1_.begin + b1 + i1;
    for (il::int_t i0 = 0; i0 < M.size(0); ++i0) {
      il::int_t j0 = range0_.begin + b0 + i0;
      M(i0, i1) = std::exp(-alpha_ * il::ipow<2>(j0 - j1) * beta);
    }
  }
}

}  // namespace il