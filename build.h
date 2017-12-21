#pragma once

#include "HMatrix.h"
#include "Matrix.h"
#include "QuadTree.h"
#include "adaptiveCrossApproximation.h"

namespace hmat {

template <il::int_t p>
hmat::HMatrix<double> build(const hmat::Matrix<p>& M,
                            const hmat::QuadTree& quad_tree, double epsilon) {
   return build_aux(M, quad_tree, 0, epsilon);
}

template <il::int_t p>
hmat::HMatrix<double> build_aux(const hmat::Matrix<p>& M,
                                const hmat::QuadTree& quad_tree, il::int_t i,
                                double epsilon) {
  const hmat::MatrixType type = quad_tree.matrixType(i);
  switch (type) {
    case hmat::MatrixType::FullRank: {
      il::Array2D<double> F =
          hmat::fullMatrix(M, quad_tree.rangeRow(i), quad_tree.rangeColumn(i));
      return hmat::HMatrix<double>{std::move(F)};
    } break;
    case hmat::MatrixType::LowRank: {
      hmat::SmallRank<double> sr = hmat::adaptiveCrossApproximation(
          M, quad_tree.rangeRow(i), quad_tree.rangeColumn(i), epsilon);
      return hmat::HMatrix<double>{std::move(sr.A), std::move(sr.B)};
    } break;
    case hmat::MatrixType::HMatrix: {
      std::unique_ptr<HMatrix<double>> M00 = std::make_unique<HMatrix<double>>(
          hmat::build_aux(M, quad_tree, quad_tree.child(i, 0), epsilon));
      std::unique_ptr<HMatrix<double>> M10 = std::make_unique<HMatrix<double>>(
          hmat::build_aux(M, quad_tree, quad_tree.child(i, 1), epsilon));
      std::unique_ptr<HMatrix<double>> M01 = std::make_unique<HMatrix<double>>(
          hmat::build_aux(M, quad_tree, quad_tree.child(i, 2), epsilon));
      std::unique_ptr<HMatrix<double>> M11 = std::make_unique<HMatrix<double>>(
          hmat::build_aux(M, quad_tree, quad_tree.child(i, 3), epsilon));
      return hmat::HMatrix<double>{std::move(M00), std::move(M10),
                                   std::move(M01), std::move(M11)};
    } break;
    default:
      IL_UNREACHABLE;
  }
}

}  // namespace hmat