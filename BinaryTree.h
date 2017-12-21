#ifndef HMATRIX_BINARYTREE_H
#define HMATRIX_BINARYTREE_H

#include <il/Array.h>

namespace hmat {

class BinaryTree {
 private:
  il::Array<il::Range> data_;

 public:
  BinaryTree();
  void insert(il::int_t i, il::Range range);
  bool hasChild(il::int_t i, il::int_t j) const;
  il::int_t child(il::int_t i, il::int_t j) const;
  il::Range range(il::int_t i) const;
};

inline BinaryTree::BinaryTree() : data_{} {}

inline void BinaryTree::insert(il::int_t i, il::Range range) {
  const il::int_t n = data_.size();
  if (i >= n) {
    const il::int_t new_n = 2 * n + 1;
    data_.resize(new_n);
    for (il::int_t j = n; j < new_n; ++j) {
      data_[j].begin = -1;
    }
    if (i >= new_n) {
      IL_UNREACHABLE;
    }
  }
  data_[i] = range;
}

inline bool BinaryTree::hasChild(il::int_t i, il::int_t j) const {
  IL_EXPECT_MEDIUM(j >= 0 && j < 4);

  const il::int_t i_child = 4 * i + 1 + j;
  return (i_child < data_.size()) && (data_[i_child].begin >= 0);
}

inline il::int_t BinaryTree::child(il::int_t i, il::int_t j) const {
  return 4 * i + 1 + j;
}

inline il::Range BinaryTree::range(il::int_t i) const { return data_[i]; }

}  // namespace hmat

#endif  // HMATRIX_BINARYTREE_H
