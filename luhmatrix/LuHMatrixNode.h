#pragma once

#include <hmatrix/HMatrixType.h>

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/linearAlgebra/Matrix.h>

namespace il {

template <typename T, typename I>
class LuHMatrixNode {
 private:
  bool empty_;
  il::HMatrixType matrix_type_;
  il::Array2D<T> A_;
  il::Array2D<T> B_;
  il::Array<I> pivot_;

 public:
  LuHMatrixNode();
  LuHMatrixNode(il::Array2D<T> A, il::Array<I> pivot);
  LuHMatrixNode(il::Array2D<T> A, il::Array2D<T> B);
  il::int_t size(il::int_t d) const;
  bool isEmpty() const;
  bool isFullRank() const;
  bool isLowRank() const;
  bool isHierarchical() const;
  il::HMatrixType type() const;
  void SetEmpty();
  void SetHierarchical();
  void SetFullRank(il::Array2D<T> A, il::Array<I> pivot);
  void SetLowRank(il::Array2D<T> A, il::Array2D<T> B);
  const il::Array2D<T>& asFullRank() const;
  il::Array2D<T>& AsFullRank();
  const il::Array<I>& asFullRankPivot() const;
  il::Array<I>& AsFullRankPivot();
  const il::Array2D<T>& asLowRankA() const;
  il::Array2D<T>& AsLowRankA();
  const il::Array2D<T>& asLowRankB() const;
  il::Array2D<T>& AsLowRankB();
};

template <typename T, typename I>
LuHMatrixNode<T, I>::LuHMatrixNode() : A_{}, B_{}, pivot_{} {
  empty_ = true;
};

template <typename T, typename I>
LuHMatrixNode<T, I>::LuHMatrixNode(il::Array2D<T> A, il::Array<I> pivot)
    : A_{std::move(A)}, B_{}, pivot_{std::move(pivot)} {
  empty_ = false;
  matrix_type_ = il::HMatrixType::FullRank;
};

template <typename T, typename I>
LuHMatrixNode<T, I>::LuHMatrixNode(il::Array2D<T> A, il::Array2D<T> B)
    : A_{std::move(A)}, B_{std::move(B)} {
  empty_ = false;
  matrix_type_ = il::HMatrixType::LowRank;
}

template <typename T, typename I>
il::int_t LuHMatrixNode<T, I>::size(il::int_t d) const {
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
  IL_UNREACHABLE;
  return -1;
}

template <typename T, typename I>
bool LuHMatrixNode<T, I>::isEmpty() const {
  return empty_;
}

template <typename T, typename I>
bool LuHMatrixNode<T, I>::isFullRank() const {
  return !empty_ && matrix_type_ == il::HMatrixType::FullRank;
}

template <typename T, typename I>
bool LuHMatrixNode<T, I>::isLowRank() const {
  return !empty_ && matrix_type_ == il::HMatrixType::LowRank;
}

template <typename T, typename I>
bool LuHMatrixNode<T, I>::isHierarchical() const {
  return !empty_ && matrix_type_ == il::HMatrixType::Hierarchical;
}

template <typename T, typename I>
il::HMatrixType LuHMatrixNode<T, I>::type() const {
  IL_EXPECT_MEDIUM(!empty_);

  return matrix_type_;
}

template <typename T, typename I>
void LuHMatrixNode<T, I>::SetEmpty() {
  empty_ = true;
  A_ = il::Array2D<T>{};
  B_ = il::Array2D<T>{};
}

template <typename T, typename I>
void LuHMatrixNode<T, I>::SetHierarchical() {
  empty_ = false;
  matrix_type_ = il::HMatrixType::Hierarchical;
  A_ = il::Array2D<T>{};
  B_ = il::Array2D<T>{};
}

template <typename T, typename I>
void LuHMatrixNode<T, I>::SetFullRank(il::Array2D<T> A, il::Array<I> pivot) {
  empty_ = false;
  matrix_type_ = il::HMatrixType::FullRank;
  A_ = std::move(A);
  B_ = il::Array2D<T>{};
  pivot_ = std::move(pivot);
}

template <typename T, typename I>
void LuHMatrixNode<T, I>::SetLowRank(il::Array2D<T> A, il::Array2D<T> B) {
  empty_ = false;
  matrix_type_ = il::HMatrixType::LowRank;
  A_ = std::move(A);
  B_ = std::move(B);
  pivot_ = il::Array<I>{};
}

template <typename T, typename I>
const il::Array2D<T>& LuHMatrixNode<T, I>::asFullRank() const {
  return A_;
}

template <typename T, typename I>
il::Array2D<T>& LuHMatrixNode<T, I>::AsFullRank() {
  return A_;
}

template <typename T, typename I>
const il::Array<I>& LuHMatrixNode<T, I>::asFullRankPivot() const {
  return pivot_;
}

template <typename T, typename I>
il::Array<I>& LuHMatrixNode<T, I>::AsFullRankPivot() {
  return pivot_;
}

template <typename T, typename I>
const il::Array2D<T>& LuHMatrixNode<T, I>::asLowRankA() const {
  return A_;
}

template <typename T, typename I>
il::Array2D<T>& LuHMatrixNode<T, I>::AsLowRankA() {
  return A_;
}

template <typename T, typename I>
const il::Array2D<T>& LuHMatrixNode<T, I>::asLowRankB() const {
  return B_;
}

template <typename T, typename I>
il::Array2D<T>& LuHMatrixNode<T, I>::AsLowRankB() {
  return B_;
}

}  // namespace il
