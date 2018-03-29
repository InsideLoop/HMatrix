#pragma once

#include <il/Tree.h>

#include <HMatrix/adaptiveCrossApproximation.h>
#include <HMatrix/HMatrix.h>
#include <HMatrix/SmallRank.h>
#include <Matrix.h>

namespace il {

template <il::int_t p>
il::HMatrix<double> toHMatrix(const il::Matrix<double, p>& matrix,
                              const il::Tree<il::SubMatrix, 4>& tree,
                              double epsilon) {
  il::HMatrix<double> ans{};
  hmatrix_rec(matrix, tree, tree.root(), epsilon, ans.root(), il::io, ans);
  return ans;
}

template <il::int_t p>
void hmatrix_rec(const il::Matrix<double, p>& matrix,
                 const il::Tree<il::SubMatrix, 4>& tree, il::spot_t st,
                 double epsilon, il::spot_t shm, il::io_t,
                 il::HMatrix<double>& hm) {
  const il::SubMatrix info = tree.value(st);
  switch (info.type) {
    case il::HMatrixType::FullRank: {
      const il::int_t n0 = info.range0.end - info.range0.begin;
      const il::int_t n1 = info.range1.end - info.range1.begin;
      hm.SetFullRank(shm, n0, n1);
      il::Array2DEdit<double> sub = hm.AsFullRank(shm);
      matrix.Set(info.range0.begin, info.range1.begin, il::io, sub);
      return;
    } break;
    case il::HMatrixType::LowRank: {
      const il::int_t n0 = info.range0.end - info.range0.begin;
      const il::int_t n1 = info.range1.end - info.range1.begin;
      il::SmallRank<double> sr = il::adaptiveCrossApproximation(
          matrix, info.range0, info.range1, epsilon);
      const il::int_t r = sr.A.size(1);
      hm.SetLowRank(shm, n0, n1, r);
      il::Array2DEdit<double> A = hm.AsLowRankA(shm);
      for (il::int_t i1 = 0; i1 < A.size(1); ++i1) {
        for (il::int_t i0 = 0; i0 < A.size(0); ++i0) {
          A(i0, i1) = sr.A(i0, i1);
        }
      }
      il::Array2DEdit<double> B = hm.AsLowRankB(shm);
      for (il::int_t i1 = 0; i1 < B.size(1); ++i1) {
        for (il::int_t i0 = 0; i0 < B.size(0); ++i0) {
          B(i0, i1) = sr.B(i1, i0);
        }
      }
      return;
    } break;
    case il::HMatrixType::Hierarchical: {
      hm.SetHierarchical(shm);
      const il::spot_t st00 = tree.child(st, 0);
      const il::spot_t shm00 = hm.child(shm, 0, 0);
      hmatrix_rec(matrix, tree, st00, epsilon, shm00, il::io, hm);
      const il::spot_t st10 = tree.child(st, 1);
      const il::spot_t shm10 = hm.child(shm, 1, 0);
      hmatrix_rec(matrix, tree, st10, epsilon, shm10, il::io, hm);
      const il::spot_t st01 = tree.child(st, 2);
      const il::spot_t shm01 = hm.child(shm, 0, 1);
      hmatrix_rec(matrix, tree, st01, epsilon, shm01, il::io, hm);
      const il::spot_t st11 = tree.child(st, 3);
      const il::spot_t shm11 = hm.child(shm, 1, 1);
      hmatrix_rec(matrix, tree, st11, epsilon, shm11, il::io, hm);
      return;
    } break;
    default:
      IL_UNREACHABLE;
  }
}

// template <il::int_t p>
// il::OldHMatrix<double> build(const il::Matrix<p>& M,
//                             const il::QuadTree& quad_tree, double epsilon) {
//  return build_aux(M, quad_tree, 0, epsilon);
//}
//
// template <il::int_t p>
// il::OldHMatrix<double> build_aux(const il::Matrix<p>& M,
//                                 const il::QuadTree& quad_tree, il::int_t i,
//                                 double epsilon) {
//  const il::HMatrixType type = quad_tree.matrixType(i);
//  switch (type) {
//    case il::HMatrixType::FullRank: {
//      il::Array2D<double> F =
//          il::fullMatrix(M, quad_tree.rangeRow(i), quad_tree.rangeColumn(i));
//      return il::OldHMatrix<double>{std::move(F)};
//    } break;
//    case il::HMatrixType::LowRank: {
//      il::SmallRank<double> sr = il::adaptiveCrossApproximation(
//          M, quad_tree.rangeRow(i), quad_tree.rangeColumn(i), epsilon);
//      return il::OldHMatrix<double>{std::move(sr.A), std::move(sr.B)};
//    } break;
//    case il::HMatrixType::Hierarchical: {
//      std::unique_ptr<OldHMatrix<double>> M00 =
//          std::unique_ptr<OldHMatrix<double>>(new OldHMatrix<double>(
//              il::build_aux(M, quad_tree, quad_tree.child(i, 0), epsilon)));
//      std::unique_ptr<OldHMatrix<double>> M10 =
//          std::unique_ptr<OldHMatrix<double>>(new OldHMatrix<double>(
//              il::build_aux(M, quad_tree, quad_tree.child(i, 1), epsilon)));
//      std::unique_ptr<OldHMatrix<double>> M01 =
//          std::unique_ptr<OldHMatrix<double>>(new OldHMatrix<double>(
//              il::build_aux(M, quad_tree, quad_tree.child(i, 2), epsilon)));
//      std::unique_ptr<OldHMatrix<double>> M11 =
//          std::unique_ptr<OldHMatrix<double>>(new OldHMatrix<double>(
//              il::build_aux(M, quad_tree, quad_tree.child(i, 3), epsilon)));
//      return il::OldHMatrix<double>{std::move(M00), std::move(M10),
//                                    std::move(M01), std::move(M11)};
//    } break;
//    default:
//      IL_UNREACHABLE;
//  }
//}

}  // namespace il