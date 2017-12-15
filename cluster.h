#ifndef HMATRIX_CLUSTER_H
#define HMATRIX_CLUSTER_H

#include <limits>

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>
#include <il/StaticArray.h>

// k: is the node number. The tree is numbered as
//
//                 0            -- Level 0
//               /   \
//             1       2        -- Level 1
//           /  |    /   |
//         3     4  5     6     -- Level 2
//                   \
//                    12        -- Level 3 -- Depth 3
//
// At construction we reserve memory as if the tree would be balanced. Then, at
// runtime, we grow the different nodes of the tree.
//
class BinaryTree {
 private:
  il::int_t initial_depth_;
  il::int_t depth_;
  il::Array<il::int_t> data_;

 public:
  BinaryTree(il::int_t nb_nodes, il::int_t leaf_size);
  void addLeft(il::int_t k, il::int_t i_begin, il::int_t i_end);
  void addRight(il::int_t k, il::int_t i_begin, il::int_t i_end);
  bool hasLeftNode(il::int_t k) const;
  bool hasRightNode(il::int_t k) const;
  il::int_t leftNode(il::int_t k) const;
  il::int_t rightNode(il::int_t k) const;
  il::int_t begin(il::int_t k) const;
  il::int_t end(il::int_t k) const;
  il::int_t root() const;
  il::int_t initialDepth() const;
  il::int_t depth() const;
};

inline BinaryTree::BinaryTree(il::int_t nb_nodes, il::int_t leaf_size) {
  IL_EXPECT_FAST(nb_nodes >= 0);
  IL_EXPECT_FAST(leaf_size > 0);

  il::int_t depth = 0;
  il::int_t power = 1;
  while (power * leaf_size < nb_nodes) {
    depth += 1;
    power *= 2;
  }
  const il::int_t n = 2 * (2 * power - 1);

  initial_depth_ = depth;
  depth_ = depth;
  data_.resize(n);
  for (il::int_t i = 0; i < n; ++i) {
    data_[i] = -1;
  }

  data_[0] = 0;
  data_[1] = nb_nodes;
}

inline void BinaryTree::addLeft(il::int_t k, il::int_t i_begin,
                                il::int_t i_end) {
  IL_EXPECT_FAST(2 * k + 1 < data_.size());

  const il::int_t k_left = 2 * k + 1;
  if (2 * (2 * k_left + 1) >= data_.size()) {
    data_.resize(2 * data_.size() + 1, -1);
    ++depth_;
  }
  data_[2 * k_left] = i_begin;
  data_[2 * k_left + 1] = i_end;
}

inline void BinaryTree::addRight(il::int_t k, il::int_t i_begin,
                                 il::int_t i_end) {
  IL_EXPECT_FAST(2 * k + 1 < data_.size());

  const il::int_t k_right = 2 * k + 2;
  if (2 * (2 * k_right + 1) >= data_.size()) {
    data_.resize(2 * data_.size() + 1, -1);
    ++depth_;
  }
  data_[2 * k_right] = i_begin;
  data_[2 * k_right + 1] = i_end;
}

inline bool BinaryTree::hasLeftNode(il::int_t k) const {
  return (2 * (2 * k + 1) < data_.size()) && data_[2 * (2 * k + 1)] >= 0;
}

inline bool BinaryTree::hasRightNode(il::int_t k) const {
  return (2 * (2 * k + 2) < data_.size()) && data_[2 * (2 * k + 2)] >= 0;
}

inline il::int_t BinaryTree::leftNode(il::int_t k) const { return 2 * k + 1; }

inline il::int_t BinaryTree::rightNode(il::int_t k) const { return 2 * k + 2; }

inline il::int_t BinaryTree::begin(il::int_t k) const { return data_[2 * k]; }

inline il::int_t BinaryTree::end(il::int_t k) const { return data_[2 * k + 1]; }

inline il::int_t BinaryTree::root() const { return 0; }

inline il::int_t BinaryTree::initialDepth() const { return initial_depth_; }

inline il::int_t BinaryTree::depth() const { return depth_; }

struct Reordering {
 public:
  BinaryTree partition;
  il::Array<il::int_t> permutation;

 public:
  Reordering(il::int_t nb_nodes, il::int_t leaf_size)
      : partition{nb_nodes, leaf_size}, permutation{nb_nodes} {
    for (il::int_t i = 0; i < nb_nodes; ++i) {
      permutation[i] = i;
    }
  }
};

Reordering clustering(il::int_t leaf_size, il::io_t, il::Array2D<double>& node);

template <il::int_t dim>
void aux_clustering_bis(il::int_t k, il::int_t leaf_size, il::io_t,
                        Reordering& reordering,
                        il::Array<il::StaticArray<double, dim>>& node) {
  const il::int_t i_begin = reordering.partition.begin(k);
  const il::int_t i_end = reordering.partition.end(k);

  if (i_end - i_begin <= leaf_size) {
    return;
  } else {
    ////////////////////////////////////////////
    // Find the dimension in which we will split
    ////////////////////////////////////////////
    const il::int_t nb_nodes = node.size();

    il::StaticArray<double, dim> coordinate_minimum{
        std::numeric_limits<double>::max()};
    il::StaticArray<double, dim> coordinate_maximum{
        -std::numeric_limits<double>::max()};
    for (il::int_t i = i_begin; i < i_end; ++i) {
      for (il::int_t d = 0; d < dim; ++d) {
        if (node[i][d] < coordinate_minimum[d]) {
          coordinate_minimum[d] = node[i][d];
        }
        if (node[i][d] > coordinate_maximum[d]) {
          coordinate_maximum[d] = node[i][d];
        }
      }
    }
    il::StaticArray<double, dim> width_box{};
    il::StaticArray<double, dim> middle_box{};
    for (il::int_t d = 0; d < dim; ++d) {
      width_box[d] = coordinate_maximum[d] - coordinate_minimum[d];
      middle_box[d] = coordinate_minimum[d] + width_box[d] / 2;
    }

    double width_maximum = 0.0;
    il::int_t d_max = -1;
    for (il::int_t d = 0; d < dim; ++d) {
      if (width_box[d] > width_maximum) {
        width_maximum = width_box[d];
        d_max = d;
      }
    }

    ////////////////////
    // Reorder the nodes
    ////////////////////
    const double middle = middle_box[d_max];

    il::int_t j = i_begin;
    for (il::int_t i = i_begin; i < i_end; ++i) {
      if (node[i][d_max] < middle) {
        il::StaticArray<double, dim> point{};
        // Swap node(i) and node (j)
        point = node[i];
        const il::int_t index = reordering.permutation[i];
        node[i] = node[j];
        reordering.permutation[i] = reordering.permutation[j];
        node[j] = point;
        reordering.permutation[j] = index;
        ++j;
      }
    }
    if (j == i_begin) {
      ++j;
    } else if (j == i_end) {
      --j;
    }

    reordering.partition.addLeft(k, i_begin, j);
    reordering.partition.addRight(k, j, i_end);

    aux_clustering_bis(reordering.partition.leftNode(k), leaf_size, il::io,
                       reordering, node);
    aux_clustering_bis(reordering.partition.rightNode(k), leaf_size, il::io,
                       reordering, node);
  }
}

template <il::int_t dim>
Reordering clustering_bis(il::int_t leaf_size, il::io_t,
                          il::Array<il::StaticArray<double, dim>>& node) {
  const il::int_t nb_nodes = node.size();

  Reordering reordering{nb_nodes, leaf_size};
  aux_clustering_bis(reordering.partition.root(), leaf_size, il::io, reordering,
                     node);

  return reordering;
}

#endif  // HMATRIX_CLUSTER_H
