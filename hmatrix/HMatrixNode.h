#pragma once

#include <hmatrix/HMatrixType.h>
#include <il/Array2D.h>
#include <il/linearAlgebra/Matrix.h>

namespace il {

template <typename T>
class HMatrixNode {
 private:
  bool empty_;
  il::HMatrixType matrix_type_;
  il::Array2D<T> A_;
  il::Array2D<T> B_;

 public:
  HMatrixNode();
  HMatrixNode(il::Array2D<T> A);
  HMatrixNode(il::Array2D<T> A, il::Array2D<T> B);
  il::int_t size(il::int_t d) const;
  bool isEmpty() const;
  bool isFullRank() const;
  bool isLowRank() const;
  bool isHierarchical() const;
  il::HMatrixType type() const;
  void SetEmpty();
  void SetHierarchical();
  void SetFullRank(il::Array2D<T> A);
  void SetLowRank(il::Array2D<T> A, il::Array2D<T> B);
  const il::Array2D<T>& asFullRank() const;
  il::Array2D<T>& AsFullRank();
  const il::Array2D<T>& asLowRankA() const;
  il::Array2D<T>& AsLowRankA();
  const il::Array2D<T>& asLowRankB() const;
  il::Array2D<T>& AsLowRankB();
};

template <typename T>
HMatrixNode<T>::HMatrixNode() : A_{}, B_{} {
  empty_ = true;
};

template <typename T>
HMatrixNode<T>::HMatrixNode(il::Array2D<T> A) : A_{std::move(A)}, B_{} {
  empty_ = false;
  matrix_type_ = il::HMatrixType::FullRank;
};

template <typename T>
HMatrixNode<T>::HMatrixNode(il::Array2D<T> A, il::Array2D<T> B)
    : A_{std::move(A)}, B_{std::move(B)} {
  empty_ = false;
  matrix_type_ = il::HMatrixType::LowRank;
}

template <typename T>
il::int_t HMatrixNode<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(!empty_);
  IL_EXPECT_MEDIUM(matrix_type_ == il::HMatrixType::LowRank ||
                   matrix_type_ == il::HMatrixType::FullRank);
  IL_EXPECT_MEDIUM(d == 0 || d == 1);

  if (matrix_type_ == il::HMatrixType::LowRank) {
    if (d == 0) {
      return A_.size(0);
    } else {
      return B_.size(0);
    }
  } else if (matrix_type_ == il::HMatrixType::FullRank) {
    return A_.size(d);
  }
}

template <typename T>
bool HMatrixNode<T>::isEmpty() const {
  return empty_;
}

template <typename T>
bool HMatrixNode<T>::isFullRank() const {
  return !empty_ && matrix_type_ == il::HMatrixType::FullRank;
}

template <typename T>
bool HMatrixNode<T>::isLowRank() const {
  return !empty_ && matrix_type_ == il::HMatrixType::LowRank;
}

template <typename T>
bool HMatrixNode<T>::isHierarchical() const {
  return !empty_ && matrix_type_ == il::HMatrixType::Hierarchical;
}

template <typename T>
il::HMatrixType HMatrixNode<T>::type() const {
  IL_EXPECT_MEDIUM(!empty_);

  return matrix_type_;
}

template <typename T>
void HMatrixNode<T>::SetEmpty() {
  empty_ = true;
  A_ = il::Array2D<T>{};
  B_ = il::Array2D<T>{};
}

template <typename T>
void HMatrixNode<T>::SetHierarchical() {
  empty_ = false;
  matrix_type_ = il::HMatrixType::Hierarchical;
  A_ = il::Array2D<T>{};
  B_ = il::Array2D<T>{};
}

template <typename T>
void HMatrixNode<T>::SetFullRank(il::Array2D<T> A) {
  empty_ = false;
  matrix_type_ = il::HMatrixType::FullRank;
  A_ = std::move(A);
  B_ = il::Array2D<T>{};
}

template <typename T>
void HMatrixNode<T>::SetLowRank(il::Array2D<T> A, il::Array2D<T> B) {
  empty_ = false;
  matrix_type_ = il::HMatrixType::LowRank;
  A_ = std::move(A);
  B_ = std::move(B);
}

template <typename T>
const il::Array2D<T>& HMatrixNode<T>::asFullRank() const {
  return A_;
}

template <typename T>
il::Array2D<T>& HMatrixNode<T>::AsFullRank() {
  return A_;
}

template <typename T>
const il::Array2D<T>& HMatrixNode<T>::asLowRankA() const {
  return A_;
}

template <typename T>
il::Array2D<T>& HMatrixNode<T>::AsLowRankA() {
  return A_;
}

template <typename T>
const il::Array2D<T>& HMatrixNode<T>::asLowRankB() const {
  return B_;
}

template <typename T>
il::Array2D<T>& HMatrixNode<T>::AsLowRankB() {
  return B_;
}

}  // namespace il
