#pragma once

#include <cmath>

#include <il/Array2D.h>
#include <il/StaticArray2D.h>
#include <il/math.h>

#include <src/core/SegmentData.h>
#include <src/elasticity/Simplified3D.h>

namespace il {

template <typename T>
class Matrix {
 private:
  il::Array2D<double> point_;
  double alpha_;

 public:
  Matrix(const il::Array2D<T>& point, T alpha);
  il::int_t size(il::int_t d) const;
  il::int_t blockSize() const;
  il::int_t sizeAsBlocks(il::int_t d) const;
  void set(il::int_t i0, il::int_t i1, il::io_t, il::Array2DEdit<T> M) const;
};

template <typename T>
Matrix<T>::Matrix(const il::Array2D<T>& point, T alpha)
    : point_{point}, alpha_{alpha} {
  IL_EXPECT_FAST(point_.size(1) == 2);
};

template <typename T>
il::int_t Matrix<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(d == 0 || d == 1);

  return point_.size(0);
};

template <typename T>
il::int_t Matrix<T>::blockSize() const {
  return 1;
}

template <typename T>
il::int_t Matrix<T>::sizeAsBlocks(il::int_t d) const {
  IL_EXPECT_MEDIUM(d == 0 || d == 1);

  return point_.size(0);
}

template <typename T>
void Matrix<T>::set(il::int_t i0, il::int_t i1, il::io_t,
                    il::Array2DEdit<T> M) const {
  IL_EXPECT_MEDIUM(M.size(0) % blockSize() == 0);
  IL_EXPECT_MEDIUM(M.size(1) % blockSize() == 0);
  IL_EXPECT_MEDIUM(i0 + M.size(0) / blockSize() <= point_.size(0));
  IL_EXPECT_MEDIUM(i1 + M.size(1) / blockSize() <= point_.size(0));

  for (il::int_t j1 = 0; j1 < M.size(1); ++j1) {
    for (il::int_t j0 = 0; j0 < M.size(0); ++j0) {
      il::int_t k0 = i0 + j0;
      il::int_t k1 = i1 + j1;
      const double dx = point_(k0, 0) - point_(k1, 0);
      const double dy = point_(k0, 1) - point_(k1, 1);
      M(j0, j1) = std::exp(-alpha_ * (dx * dx + dy * dy));
    }
  }
}

}  // namespace il
