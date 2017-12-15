#include <iostream>
#include <random>

#include <il/Timer.h>

#include "cluster.h"

int main() {
  const il::int_t dim = 2;
  const il::int_t nb_nodes = 1000000;
  const il::int_t leaf_size = 10;
  const il::int_t nb_times = 100;

  il::Array2D<double> node_master{nb_nodes, dim};
  std::default_random_engine engine{};
  std::uniform_real_distribution<double> r_dist{0.0, 1.0};
  for (il::int_t i = 0; i < nb_nodes; ++i) {
    node_master(i, 0) = r_dist(engine);
    node_master(i, 1) = r_dist(engine);
  }

  {
    il::Timer timer{};
    for (il::int_t k = 0; k < nb_times; ++k) {
      il::Array2D<double> node = node_master;
      timer.start();
      const Reordering reordering = clustering(leaf_size, il::io, node);
      timer.stop();
    }
    std::cout << "Average time: " << timer.time() / nb_times << std::endl;
  }

  {
    il::Timer timer{};
    for (il::int_t k = 0; k < nb_times; ++k) {
      il::Array<il::StaticArray<double, dim>> node{nb_nodes};
      for (il::int_t i = 0; i < nb_nodes; ++i) {
        for (il::int_t d = 0; d < dim; ++d) {
          node[i][d] = node_master(i, d);
        }
      }
      timer.start();
      const Reordering reordering = clustering_bis(leaf_size, il::io, node);
      timer.stop();
    }
    std::cout << "Average time: " << timer.time() / nb_times << std::endl;
  }

  return 0;
}