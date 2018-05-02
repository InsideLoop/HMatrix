#pragma once

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/Map.h>

#include <hmatrix/HMatrixNode.h>
#include <hmatrix/HMatrixType.h>

namespace il {

template <typename T>
class HMatrix {
 private:
  il::Array<il::HMatrixNode<T>> tree_;

 public:
  HMatrix();

  il::int_t size(il::int_t d) const;
  il::int_t size(il::int_t d, il::spot_t s) const;

  il::spot_t root() const;
  il::spot_t parent(il::spot_t s) const;
  il::spot_t child(il::spot_t s, il::int_t i0, il::int_t i1) const;

  bool isFullRank(il::spot_t s) const;
  bool isLowRank(il::spot_t s) const;
  bool isHierarchical(il::spot_t s) const;
  bool isFullLu(il::spot_t s) const;
  il::HMatrixType type(il::spot_t s) const;

  il::int_t rankOfLowRank(il::spot_t s) const;
  void UpdateRank(il::spot_t s, il::int_t r);

  void SetHierarchical(il::spot_t s);
  void SetFullRank(il::spot_t s, il::int_t n0, il::int_t n1);
  void SetLowRank(il::spot_t s, il::int_t n0, il::int_t n1, il::int_t r);
  void ConvertToFullLu(il::spot_t s);

  il::Array2DView<T> asFullRank(il::spot_t s) const;
  il::Array2DEdit<T> AsFullRank(il::spot_t s);

  il::Array2DView<T> asLowRankA(il::spot_t s) const;
  il::Array2DEdit<T> AsLowRankA(il::spot_t s);
  il::Array2DView<T> asLowRankB(il::spot_t s) const;
  il::Array2DEdit<T> AsLowRankB(il::spot_t s);

  il::ArrayView<int> asFullLuPivot(il::spot_t s) const;
  il::ArrayEdit<int> AsFullLuPivot(il::spot_t s);
  il::Array2DView<T> asFullLu(il::spot_t s) const;
  il::Array2DEdit<T> AsFullLu(il::spot_t s);

  bool isBuilt() const;

 private:
  bool isBuilt(il::spot_t s) const;
};

template <typename T>
HMatrix<T>::HMatrix() : tree_{1} {}

template <typename T>
il::int_t HMatrix<T>::size(il::int_t d) const {
  return size(d, il::spot_t{0});
}

template <typename T>
il::spot_t HMatrix<T>::root() const {
  return il::spot_t{0};
}

template <typename T>
il::spot_t HMatrix<T>::parent(il::spot_t s) const {
  return il::spot_t{s.index / 4};
}

template <typename T>
il::spot_t HMatrix<T>::child(il::spot_t s, il::int_t i0, il::int_t i1) const {
  return il::spot_t{4 * s.index + 1 + i1 * 2 + i0};
}

template <typename T>
bool HMatrix<T>::isFullRank(il::spot_t s) const {
  return tree_[s.index].isFullRank();
}

template <typename T>
bool HMatrix<T>::isLowRank(il::spot_t s) const {
  return tree_[s.index].isLowRank();
}

template <typename T>
bool HMatrix<T>::isHierarchical(il::spot_t s) const {
  return tree_[s.index].isHierarchical();
}

template <typename T>
bool HMatrix<T>::isFullLu(il::spot_t s) const {
  return tree_[s.index].isFullLu();
}

template <typename T>
il::HMatrixType HMatrix<T>::type(il::spot_t s) const {
  return tree_[s.index].type();
}

template <typename T>
il::int_t HMatrix<T>::rankOfLowRank(il::spot_t s) const {
  return tree_[s.index].rankOfLowRank();
}

template <typename T>
void HMatrix<T>::UpdateRank(il::spot_t s, il::int_t r) {
  return tree_[s.index].UpdateRank(r);
}

template <typename T>
void HMatrix<T>::SetHierarchical(il::spot_t s) {
  tree_[s.index].SetHierarchical();
  if (tree_.size() < 4 * (s.index + 1) + 1) {
    tree_.Resize(4 * (s.index + 1) + 1);
  }
}

template <typename T>
void HMatrix<T>::SetFullRank(il::spot_t s, il::int_t n0, il::int_t n1) {
//  tree_[s.index].SetFullRank(il::Array2D<T>{n0, n1});
  tree_[s.index].SetFullRank(n0, n1);
}

template <typename T>
void HMatrix<T>::SetLowRank(il::spot_t s, il::int_t n0, il::int_t n1,
                            il::int_t r) {
//  tree_[s.index].SetLowRank(il::Array2D<T>{n0, r}, il::Array2D<T>{n1, r});
  tree_[s.index].SetLowRank(n0, n1, r);
}

template <typename T>
void HMatrix<T>::ConvertToFullLu(il::spot_t s) {
  return tree_[s.index].ConvertToFullLu();
}

template <typename T>
il::Array2DView<T> HMatrix<T>::asFullRank(il::spot_t s) const {
  return tree_[s.index].asFullRank().view();
}

template <typename T>
il::Array2DEdit<T> HMatrix<T>::AsFullRank(il::spot_t s) {
  return tree_[s.index].AsFullRank().Edit();
}

template <typename T>
il::Array2DView<T> HMatrix<T>::asLowRankA(il::spot_t s) const {
  return tree_[s.index].asLowRankA().view();
}

template <typename T>
il::Array2DEdit<T> HMatrix<T>::AsLowRankA(il::spot_t s) {
  return tree_[s.index].AsLowRankA().Edit();
}

template <typename T>
il::Array2DView<T> HMatrix<T>::asLowRankB(il::spot_t s) const {
  return tree_[s.index].asLowRankB().view();
}

template <typename T>
il::Array2DEdit<T> HMatrix<T>::AsLowRankB(il::spot_t s) {
  return tree_[s.index].AsLowRankB().Edit();
}

template <typename T>
il::ArrayView<int> HMatrix<T>::asFullLuPivot(il::spot_t s) const {
  return tree_[s.index].asFullLuPivot().view();
}

template <typename T>
il::ArrayEdit<int> HMatrix<T>::AsFullLuPivot(il::spot_t s) {
  return tree_[s.index].AsFullLuPivot().Edit();
}

template <typename T>
il::Array2DView<T> HMatrix<T>::asFullLu(il::spot_t s) const {
  return tree_[s.index].asFullLu().view();
}

template <typename T>
il::Array2DEdit<T> HMatrix<T>::AsFullLu(il::spot_t s) {
  return tree_[s.index].AsFullLu().Edit();
}

template <typename T>
bool HMatrix<T>::isBuilt() const {
  return isBuilt(root());
}

template <typename T>
il::int_t HMatrix<T>::size(il::int_t d, il::spot_t s) const {
  if (tree_[s.index].isFullRank() || tree_[s.index].isLowRank() || tree_[s.index].isFullLu()) {
    return tree_[s.index].size(d);
  } else if (tree_[s.index].isEmpty()) {
    IL_UNREACHABLE;
  } else {
    const il::int_t n00 = size(d, child(s, 0, 0));
    const il::int_t n10 = size(d, child(s, 1, 0));
    const il::int_t n01 = size(d, child(s, 0, 1));
    const il::int_t n11 = size(d, child(s, 1, 1));
    if (d == 0) {
      IL_EXPECT_MEDIUM(n00 == n01);
      IL_EXPECT_MEDIUM(n10 == n11);
      return n00 + n10;
    } else {
      IL_EXPECT_MEDIUM(n00 == n10);
      IL_EXPECT_MEDIUM(n01 == n11);
      return n00 + n01;
    }
  }
  IL_UNREACHABLE;
  return -1;
}

template <typename T>
bool HMatrix<T>::isBuilt(il::spot_t s) const {
  if (tree_[s.index].isFullRank() || tree_[s.index].isLowRank()) {
    return true;
  } else if (tree_[s.index].isEmpty()) {
    return false;
  } else {
    const il::spot_t s00 = child(s, 0, 0);
    const il::spot_t s10 = child(s, 1, 0);
    const il::spot_t s01 = child(s, 0, 1);
    const il::spot_t s11 = child(s, 1, 1);
    return isBuilt(s00) && isBuilt(s10) && isBuilt(s01) && isBuilt(s11);
  }
}

}  // namespace il