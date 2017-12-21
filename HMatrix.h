#pragma once

#include <memory>

#include <il/linear_algebra/dense/blas/blas.h>

namespace hmat {

enum class HMatrixType { LowRank, FullRank, HMatrix };

template <typename T>
class HMatrix {
 private:
  hmat::HMatrixType type_;
  il::Array2D<T> A_;
  il::Array2D<T> B_;
  il::Array2D<T> F_;
  il::StaticArray2D<std::unique_ptr<hmat::HMatrix<T>>, 2, 2> submatrix_;

 public:
  HMatrix();
  HMatrix(il::Array2D<T> F);
  HMatrix(il::Array2D<T> A, il::Array2D<T> B);
  HMatrix(std::unique_ptr<hmat::HMatrix<T>> M00,
          std::unique_ptr<hmat::HMatrix<T>> M10,
          std::unique_ptr<hmat::HMatrix<T>> M01,
          std::unique_ptr<hmat::HMatrix<T>> M11);
  il::int_t size(il::int_t d) const;
  double compressionRatio() const;
  hmat::HMatrixType type() const;
  il::Array<T> dot(const il::Array<T>& x) const;

 private:
  il::int_t nbStoredElements() const;
  void dot(const il::ArrayView<T>& x, il::io_t, il::ArrayEdit<T> y) const;
};

template <typename T>
HMatrix<T>::HMatrix() : A_{}, B_{}, F_{}, submatrix_{} {
  type_ = hmat::HMatrixType::FullRank;
}

template <typename T>
HMatrix<T>::HMatrix(il::Array2D<T> F)
    : A_{}, B_{}, F_{std::move(F)}, submatrix_{} {
  type_ = hmat::HMatrixType::FullRank;
}

template <typename T>
HMatrix<T>::HMatrix(il::Array2D<T> A, il::Array2D<T> B)
    : A_{std::move(A)}, B_{std::move(B)}, F_{}, submatrix_{} {
  IL_EXPECT_MEDIUM(A.size(1) == B.size(0));

  type_ = hmat::HMatrixType::LowRank;
}

template <typename T>
HMatrix<T>::HMatrix(std::unique_ptr<hmat::HMatrix<T>> M00,
                    std::unique_ptr<hmat::HMatrix<T>> M10,
                    std::unique_ptr<hmat::HMatrix<T>> M01,
                    std::unique_ptr<hmat::HMatrix<T>> M11)
    : A_{}, B_{}, F_{}, submatrix_{} {
  IL_EXPECT_MEDIUM(M00->size(0) == M01->size(0));
  IL_EXPECT_MEDIUM(M10->size(0) == M11->size(0));
  IL_EXPECT_MEDIUM(M00->size(1) == M10->size(1));
  IL_EXPECT_MEDIUM(M01->size(1) == M11->size(1));

  submatrix_(0, 0) = std::move(M00);
  submatrix_(1, 0) = std::move(M10);
  submatrix_(0, 1) = std::move(M01);
  submatrix_(1, 1) = std::move(M11);
  type_ = hmat::HMatrixType::HMatrix;
}

template <typename T>
il::int_t HMatrix<T>::nbStoredElements() const {
  const il::int_t total_size = size(0) * size(1);
  il::int_t ans;

  switch (type_) {
    case hmat::HMatrixType::LowRank:
      ans = (A_.size(0) + B_.size(1)) * A_.size(1);
      break;
    case hmat::HMatrixType::FullRank:
      ans = F_.size(0) * F_.size(1);
      break;
    case hmat::HMatrixType::HMatrix:
      ans = submatrix_(0, 0)->nbStoredElements() +
             submatrix_(1, 0)->nbStoredElements() +
             submatrix_(0, 1)->nbStoredElements() +
             submatrix_(1, 1)->nbStoredElements();
      break;
    default:
      IL_UNREACHABLE;
  }

  return ans;
}

template <typename T>
double HMatrix<T>::compressionRatio() const {
  const il::int_t n0 = size(0);
  const il::int_t n1 = size(1);

  return nbStoredElements() / static_cast<double>(n0 * n1);
}

template <typename T>
il::int_t HMatrix<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));

  switch (type_) {
    case hmat::HMatrixType::LowRank:
      return (d == 0) ? A_.size(0) : B_.size(1);
      break;
    case hmat::HMatrixType::FullRank:
      return F_.size(d);
      break;
    case hmat::HMatrixType::HMatrix:
      return (d == 0) ? (submatrix_(0, 0)->size(0) + submatrix_(1, 0)->size(0))
                      : (submatrix_(0, 0)->size(1) + submatrix_(0, 1)->size(1));
      break;
    default:
      IL_UNREACHABLE;
  }
}

template <typename T>
hmat::HMatrixType HMatrix<T>::type() const {
  return type_;
}

template <typename T>
il::Array<T> HMatrix<T>::dot(const il::Array<T>& x) const {
  IL_EXPECT_FAST(size(1) == x.size());

  il::Array<T> y{size(0), 0.0};
  il::ArrayView<T> x_view = x.view();
  il::ArrayEdit<T> y_edit = y.edit();
  dot(x_view, il::io, y_edit);

  return y;
}

template <typename T>
void HMatrix<T>::dot(const il::ArrayView<T>& x, il::io_t,
                     il::ArrayEdit<T> y) const {
  IL_EXPECT_FAST(size(1) == x.size());

  switch (type_) {
    case hmat::HMatrixType::LowRank: {
      il::Array<T> tmp{B_.size(0)};
      il::blas(1.0, B_.view(), x, 1.0, il::io, tmp.edit());
      il::blas(1.0, A_.view(), tmp.view(), 1.0, il::io, y.edit());
    } break;
    case hmat::HMatrixType::FullRank: {
      il::blas(1.0, F_.view(), x, 1.0, il::io, y.edit());
    } break;
    case hmat::HMatrixType::HMatrix: {
      const il::int_t n00 = submatrix_(0, 0)->size(0);
      const il::int_t n01 = submatrix_(1, 0)->size(0);
      const il::int_t n10 = submatrix_(0, 0)->size(1);
      const il::int_t n11 = submatrix_(0, 1)->size(1);
      il::ArrayView<T> x0 = x.view(il::Range{0, n10});
      il::ArrayView<T> x1 = x.view(il::Range{n10, n10 + n11});
      il::ArrayEdit<T> y0 = y.edit(il::Range{0, n00});
      il::ArrayEdit<T> y1 = y.edit(il::Range{n00, n00 + n01});
      submatrix_(0, 0)->dot(x0, il::io, y0);
      submatrix_(0, 1)->dot(x1, il::io, y0);
      submatrix_(1, 0)->dot(x0, il::io, y1);
      submatrix_(1, 1)->dot(x1, il::io, y1);
    } break;
    default:
      IL_UNREACHABLE;
  }
}

}  // namespace hmat
