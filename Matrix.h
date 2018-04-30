#pragma once

#include <arrayFunctor/MatrixGenerator.h>

namespace il {

class Matrix : public MatrixGenerator<double> {
 private:
  il::Array2D<double> point_;
  double alpha_;

 public:
  Matrix(const il::Array2D<double>& point, double alpha);
  il::int_t size(il::int_t d) const override;
  il::int_t blockSize() const override;
  il::int_t sizeAsBlocks(il::int_t d) const override;
  void set(il::int_t b0, il::int_t b1, il::io_t,
           il::Array2DEdit<double> M) const override;
};

inline Matrix::Matrix(const il::Array2D<double>& point, double alpha)
    : point_{point}, alpha_{alpha} {
  IL_EXPECT_FAST(point_.size(1) == 2);
};

inline il::int_t Matrix::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(d == 0 || d == 1);

  return point_.size(0);
};

inline il::int_t Matrix::blockSize() const { return 1; }

inline il::int_t Matrix::sizeAsBlocks(il::int_t d) const {
  IL_EXPECT_MEDIUM(d == 0 || d == 1);

  return point_.size(0);
}

inline void Matrix::set(il::int_t b0, il::int_t b1, il::io_t,
                 il::Array2DEdit<double> M) const {
  IL_EXPECT_MEDIUM(M.size(0) % blockSize() == 0);
  IL_EXPECT_MEDIUM(M.size(1) % blockSize() == 0);
  IL_EXPECT_MEDIUM(b0 + M.size(0) / blockSize() <= point_.size(0));
  IL_EXPECT_MEDIUM(b1 + M.size(1) / blockSize() <= point_.size(0));

  for (il::int_t j1 = 0; j1 < M.size(1); ++j1) {
    for (il::int_t j0 = 0; j0 < M.size(0); ++j0) {
      il::int_t k0 = b0 + j0;
      il::int_t k1 = b1 + j1;
      const double dx = point_(k0, 0) - point_(k1, 0);
      const double dy = point_(k0, 1) - point_(k1, 1);
      M(j0, j1) = std::exp(-alpha_ * (dx * dx + dy * dy));
    }
  }
}

}  // namespace il
