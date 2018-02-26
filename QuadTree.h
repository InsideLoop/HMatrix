#ifndef HMATRIX_HMATRIXTREE_H
#define HMATRIX_HMATRIXTREE_H

#include <il/Array2D.h>
#include <il/algorithmArray2D.h>

#include "MatrixType.h"



namespace hmat {

class QuadTree {
 private:
  struct Data {
    il::Range range_row;
    il::Range range_column;
    hmat::MatrixType matrix_type;
  };
  il::Array<Data> data_;

 public:
  QuadTree();
  void insert(il::int_t i, il::Range range_0, il::Range range_1,
              hmat::MatrixType type);
  bool hasChild(il::int_t i, il::int_t j) const;
  il::int_t child(il::int_t i, il::int_t j) const;
  il::Range rangeRow(il::int_t i) const;
  il::Range rangeColumn(il::int_t i) const;
  hmat::MatrixType matrixType(il::int_t i) const;
};

inline QuadTree::QuadTree() : data_{} {}

inline void QuadTree::insert(il::int_t i, il::Range range_0, il::Range range_1,
                             hmat::MatrixType type) {
  const il::int_t n = data_.size();
  if (i >= n) {
    const il::int_t new_n = 4 * n + 1;
    data_.Resize(new_n);
    for (il::int_t j = n; j < new_n; ++j) {
      data_[j].range_row.begin = -1;
    }
    if (i >= new_n) {
      IL_UNREACHABLE;
    }
  }
  data_[i].range_row = range_0;
  data_[i].range_column = range_1;
  data_[i].matrix_type = type;
}

inline bool QuadTree::hasChild(il::int_t i, il::int_t j) const {
  IL_EXPECT_MEDIUM(j >= 0 && j < 4);

  const il::int_t i_child = 4 * i + 1 + j;
  return (i_child < data_.size()) && (data_[i_child].range_row.begin >= 0);
}

inline il::int_t QuadTree::child(il::int_t i, il::int_t j) const {
  return 4 * i + 1 + j;
}

inline il::Range QuadTree::rangeRow(il::int_t i) const {
  return data_[i].range_row;
}

inline il::Range QuadTree::rangeColumn(il::int_t i) const {
  return data_[i].range_column;
}

inline hmat::MatrixType QuadTree::matrixType(il::int_t i) const {
  return data_[i].matrix_type;
}

}  // namespace hmat

#endif  // HMATRIX_HMATRIXTREE_H
