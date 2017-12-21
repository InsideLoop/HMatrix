#pragma once

#include "HMatrix.h"
#include "Matrix.h"
#include "QuadTree.h"

namespace hmat {

template <il::int_t p>
hmat::HMatrix<double> build(const hmat::Matrix<p>& M,
                            const hmat::QuadTree& quad_tree, double epsilon) {

}

template <il::int_t p>
hmat::HMatrix<double> build_aux(const hmat::Matrix<p>& M,
                            const hmat::QuadTree& quad_tree, il::int_t i, double epsilon) {
  const hmat::MatrixType type = quad_tree.matrixType(i);
  switch (type) {
    case hmat::MatrixType::FullRank: {
      const il::Range range0 = quad_tree.rangeRow(i);
    }

      break;
    default:
      IL_UNREACHABLE;
  }

}

}  // namespace hmat