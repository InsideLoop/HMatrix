#pragma once

#include <HMatrix/LowRank.h>
#include <il/Array.h>
#include <il/Array2D.h>

namespace il {

template <typename T>
class LUArray2D {
 public:
  il::Array2D<T> lu_;
  il::Array<int> ipiv_;

 public:
  LUArray2D(il::Array2D<T> lu, il::Array<int> ipiv)
      : lu_{std::move(lu)}, ipiv_(std::move(ipiv)){};
};

template <typename T>
class LUHMatrixNode {
 private:
  // - void is the pointer is nullprt
  // - A full matrix if the pointer ends with a 01
  // - A LowRank matrix if the pointer end with a 10
  // - A Hierarchical matrix if the pointer ends with an 11
  union {
    void *p_;
    unsigned char raw_[8];
  };

 public:
  LUHMatrixNode();
  LUHMatrixNode(il::Array2D<T> lu, il::Array<int> ipiv);
  LUHMatrixNode(const il::LUHMatrixNode<T> &n);
  LUHMatrixNode(il::LUHMatrixNode<T> &&n);
  LUHMatrixNode &operator=(const il::LUHMatrixNode<T> &n) = delete;
  LUHMatrixNode &operator=(il::LUHMatrixNode<T> &&n);
  ~LUHMatrixNode();
  il::Array2DEdit<T> AsFullRank();
  il::ArrayEdit<int> AsPivotFull();

 private:
  il::Array2D<T>* FullPointer();
  il::Array<int>* PivotFullPointer();
};

template <typename T>
LUHMatrixNode<T>::LUHMatrixNode() {
  p_ = nullptr;
};

template <typename T>
LUHMatrixNode<T>::LUHMatrixNode(il::Array2D<T> lu, il::Array<int> ipiv) {
  p_ = static_cast<void *>(
      new il::LUArray2D<T>{std::move(lu), std::move(ipiv)});
  raw_[7] = 128;
};

template <typename T>
LUHMatrixNode<T>::LUHMatrixNode(const il::LUHMatrixNode<T> &n) {
  if (n.raw_[7] = 128) {
    p_ = static_cast<void *>(
        new il::LUArray2D<T>{*static_cast<il::LUArray2D<T> *>(n.p_)});
    raw_[7] = 128;
  } else {
    IL_UNREACHABLE;
  }
}

template <typename T>
LUHMatrixNode<T>::LUHMatrixNode(il::LUHMatrixNode<T> &&n) {
  p_ = n.p_;
  n.p_ = nullptr;
}

template <typename T>
LUHMatrixNode<T>::~LUHMatrixNode() {
  if (raw_[7] == 128) {
    delete reinterpret_cast<il::LUArray2D<T> *>(p_);
  }
}

template <typename T>
il::Array2DEdit<T> LUHMatrixNode<T>::AsFullRank() {
  return FullPointer()->Edit();
}

template <typename T>
il::ArrayEdit<int> LUHMatrixNode<T>::AsPivotFull() {
  return PivotFullPointer()->Edit();
}

template <typename T>
il::Array2D<T>* LUHMatrixNode<T>::FullPointer() {
  union {
    il::Array2D<T>* p;
    unsigned char raw[8];
  };
  p = &((*static_cast<il::LUArray2D<T>*>(p_)).lu_);
  raw[7] = 0;
  return p;
}

template <typename T>
il::Array<int>* LUHMatrixNode<T>::PivotFullPointer() {
  union {
    il::Array<int>* p;
    unsigned char raw[8];
  };
  p = &((*static_cast<il::LUArray2D<T>*>(p_)).ipiv_);
  raw[7] = 0;
  return p;
}

}// namespace il
