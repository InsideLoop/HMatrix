#pragma once

#include <il/Array.h>
#include <il/Array2D.h>

#include <linearAlgebra/factorization/old/LUHMatrixNode.h>
#include <HMatrix/LowRank.h>
#include <HMatrix/HMatrixType.h>

namespace il {

template <typename T>
class LUHMatrix {
 private:
  il::int_t size0_;
  il::int_t size1_;
  il::Array<il::LUHMatrixNode<T>> tree_;

 public:
  LUHMatrix();
  LUHMatrix(il::int_t n0, il::int_t n1, il::HMatrixType type);

  il::int_t childSpot(il::int_t k, il::int_t i0, il::int_t i1);
  void Set(il::int_t k, il::Array2D<T> lu, il::Array<int> ipiv);
  void SetFullRank(il::int_t k, il::int_t n0, il::int_t n1);
  il::Array2DEdit<T> AsFullRank(il::int_t k);
  il::ArrayEdit<int> AsPivotFull(il::int_t k);
};

template <typename T>
LUHMatrix<T>::LUHMatrix(il::int_t n0, il::int_t n1, il::HMatrixType type) {
  size0_ = n0;
  size1_ = n1;
  tree_.Append(il::emplace, il::Array2D<T>{n0, n1},
               il::Array<int>{il::min(n0, n1)});
}

template <typename T>
LUHMatrix<T>::LUHMatrix() : tree_{} {
  size0_ = 0;
  size1_ = 0;
}

template <typename T>
void LUHMatrix<T>::Set(il::int_t k, il::Array2D<T> lu, il::Array<int> ipiv) {
  IL_EXPECT_MEDIUM(k == 0);
  IL_EXPECT_MEDIUM(tree_.size() == 0);

  tree_.Append(il::emplace, std::move(lu), std::move(ipiv));
}

template <typename T>
void LUHMatrix<T>::SetFullRank(il::int_t k, il::int_t n0, il::int_t n1) {
  IL_EXPECT_MEDIUM(k == 0);
  IL_EXPECT_MEDIUM(tree_.size() == 0);

  tree_.Append(il::emplace, il::Array2D<T>{n0, n1},
               il::Array<int>(il::min(n0, n1)));
}

template <typename T>
il::Array2DEdit<T> LUHMatrix<T>::AsFullRank(il::int_t k) {
  return tree_[k].AsFullRank();
}

template <typename T>
il::ArrayEdit<int> LUHMatrix<T>::AsPivotFull(il::int_t k) {
  return tree_[k].AsPivotFull();
}

}  // namespace il
