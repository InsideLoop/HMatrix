#pragma once

#include <il/Array2D.h>

namespace il {

template <typename T>
class LowRank {
 private:
  // LowRank<T> stores the matrices as M = A.B^T
  il::Array2D<T> A_;
  il::Array2D<T> B_;

 public:
  LowRank();
  LowRank(il::Array2D<T> A, il::Array2D<T> B);
  il::int_t size(il::int_t d) const;
};

template <typename T>
LowRank<T>::LowRank() : A_{}, B_{} {}

template <typename T>
LowRank<T>::LowRank(il::Array2D<T> A, il::Array2D<T> B)
    : A_{std::move(A)}, B_{std::move(B)} {
  IL_EXPECT_FAST(A_.size(1) == B.size(1));
}

template <typename T>
il::int_t LowRank<T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(d == 0 || d == 1);

  return d == 0 ? A_.size(0) : B_.size(0);
}

}  // namespace il
