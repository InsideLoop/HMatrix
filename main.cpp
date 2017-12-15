#include <iostream>
#include <random>

#include <il/Timer.h>

#include "QuadTree.h"
#include "cluster.h"

int main() {
  const il::int_t n = 8;
  const il::int_t leaf_size = 2;
  il::Array2D<double> node{n, 1};
  for (il::int_t i = 0; i < n; ++i) {
    node(i, 0) = i * (1.0 / (n - 1));
  }

  const Reordering reordering = clustering(leaf_size, il::io, node);

  const double eta = 0.5;
  hmat::QuadTree qtree = matrixClustering(node, reordering.partition, eta);

  return 0;
}