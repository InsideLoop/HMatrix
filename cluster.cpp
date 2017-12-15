#include "cluster.h"

#include <limits>

void aux_clustering(il::int_t k, il::int_t leaf_size, il::io_t,
                    Reordering& reordering, il::Array2D<double>& node) {
  const il::int_t i_begin = reordering.partition.begin(k);
  const il::int_t i_end = reordering.partition.end(k);

  if (i_end - i_begin <= leaf_size) {
    return;
  } else {
    ////////////////////////////////////////////
    // Find the dimension in which we will split
    ////////////////////////////////////////////
    const il::int_t nb_nodes = node.size(0);
    const il::int_t dim = node.size(1);
    il::Array<double> width_box{dim};
    il::Array<double> middle_box{dim};

    for (il::int_t d = 0; d < dim; ++d) {
      double coordinate_minimum = std::numeric_limits<double>::max();
      double coordinate_maximum = -std::numeric_limits<double>::max();
      for (il::int_t i = i_begin; i < i_end; ++i) {
        if (node(i, d) < coordinate_minimum) {
          coordinate_minimum = node(i, d);
        }
        if (node(i, d) > coordinate_maximum) {
          coordinate_maximum = node(i, d);
        }
      }
      width_box[d] = coordinate_maximum - coordinate_minimum;
      middle_box[d] = coordinate_minimum + width_box[d] / 2;
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
    il::Array<double> point{dim};

    il::int_t j = i_begin;
    for (il::int_t i = i_begin; i < i_end; ++i) {
      if (node(i, d_max) < middle) {
        // Swap node(i) and node (j)
        for (il::int_t d = 0; d < dim; ++d) {
          point[d] = node(i, d);
        }
        const il::int_t index = reordering.permutation[i];
        for (il::int_t d = 0; d < dim; ++d) {
          node(i, d) = node(j, d);
        }
        reordering.permutation[i] = reordering.permutation[j];
        for (il::int_t d = 0; d < dim; ++d) {
          node(j, d) = point[d];
        }
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

    aux_clustering(reordering.partition.leftNode(k), leaf_size, il::io,
                   reordering, node);
    aux_clustering(reordering.partition.rightNode(k), leaf_size, il::io,
                   reordering, node);
  }
}

Reordering clustering(il::int_t leaf_size, il::io_t,
                      il::Array2D<double>& node) {
  const il::int_t nb_nodes = node.size(0);

  Reordering reordering{nb_nodes, leaf_size};
  aux_clustering(reordering.partition.root(), leaf_size, il::io, reordering,
                 node);

  return reordering;
}
