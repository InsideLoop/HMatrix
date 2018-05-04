#include <gtest/gtest.h>

#include <il/Tree.h>
#include <il/math.h>

#include <hmatrix/HMatrix.h>
#include <hmatrix/HMatrixType.h>
#include <hmatrix/HMatrixUtils.h>
#include <linearAlgebra/blas/hsolve.h>
#include <linearAlgebra/blas/hdot.h>
#include <linearAlgebra/factorization/luDecomposition.h>
#include <arrayFunctor/ArrayFunctor.h>
#include <cluster/cluster.h>
#include <compression/toHMatrix.h>

TEST(solve, test0) {
  il::HMatrix<double> H{};
  const il::spot_t s = H.root();
  H.SetHierarchical(s);

  const il::spot_t s00 = H.child(s, 0, 0);
  H.SetFullRank(s00, 2, 2);
  il::Array2DEdit<double> H00 = H.AsFullRank(s00);
  H00(0, 0) = 3.0;
  H00(1, 1) = 4.0;
  H00(0, 1) = 0.0;
  H00(1, 0) = 1.0;

  const il::spot_t s11 = H.child(s, 1, 1);
  H.SetFullRank(s11, 2, 2);
  il::Array2DEdit<double> H11 = H.AsFullRank(s11);
  H11(0, 0) = 5.0;
  H11(1, 1) = 6.0;
  H11(0, 1) = 0.0;
  H11(1, 0) = 0.0;

  const il::spot_t s01 = H.child(s, 0, 1);
  H.SetLowRank(s01, 2, 2, 1);
  il::Array2DEdit<double> H01A = H.AsLowRankA(s01);
  il::Array2DEdit<double> H01B = H.AsLowRankB(s01);
  H01A(0, 0) = 1.0;
  H01A(1, 0) = 1.0;
  H01B(0, 0) = 1.0;
  H01B(1, 0) = 1.0;

  const il::spot_t s10 = H.child(s, 1, 0);
  H.SetLowRank(s10, 2, 2, 1);
  il::Array2DEdit<double> H10A = H.AsLowRankA(s10);
  il::Array2DEdit<double> H10B = H.AsLowRankB(s10);
  H10A(0, 0) = 1.0;
  H10A(1, 0) = 1.0;
  H10B(0, 0) = 1.0;
  H10B(1, 0) = 1.0;

  il::luDecomposition(il::io, H);

  il::Array<double> y = {il::value, {10.0, 16.0, 18.0, 27.0}};
  il::solve(H, il::MatrixType::LowerUnitUpperNonUnit, il::io, y.Edit());

  const double eps = 1.0e-10;

  ASSERT_TRUE(il::abs(y[0] - 1.0) <= eps && il::abs(y[1] - 2.0) <= eps &&
              il::abs(y[2] - 3.0) <= eps && il::abs(y[3] - 4.0) <= eps);
}

TEST(solve, test1) {
  il::HMatrix<double> H{};
  const il::spot_t s = H.root();
  H.SetHierarchical(s);

  const il::spot_t s00 = H.child(s, 0, 0);
  H.SetFullRank(s00, 2, 2);
  il::Array2DEdit<double> H00 = H.AsFullRank(s00);
  H00(0, 0) = 1.0;
  H00(0, 1) = 4.0;
  H00(1, 0) = 3.0;
  H00(1, 1) = 0.0;

  const il::spot_t s11 = H.child(s, 1, 1);
  H.SetFullRank(s11, 2, 2);
  il::Array2DEdit<double> H11 = H.AsFullRank(s11);
  H11(0, 0) = 0.0;
  H11(0, 1) = 6.0;
  H11(1, 0) = 5.0;
  H11(1, 1) = 1.0;

  const il::spot_t s01 = H.child(s, 0, 1);
  H.SetLowRank(s01, 2, 2, 1);
  il::Array2DEdit<double> H01A = H.AsLowRankA(s01);
  il::Array2DEdit<double> H01B = H.AsLowRankB(s01);
  H01A(0, 0) = 1.0;
  H01A(1, 0) = 1.0;
  H01B(0, 0) = 1.0;
  H01B(1, 0) = 0.5;

  const il::spot_t s10 = H.child(s, 1, 0);
  H.SetLowRank(s10, 2, 2, 1);
  il::Array2DEdit<double> H10A = H.AsLowRankA(s10);
  il::Array2DEdit<double> H10B = H.AsLowRankB(s10);
  H10A(0, 0) = 0.5;
  H10A(1, 0) = 1.0;
  H10B(0, 0) = 1.0;
  H10B(1, 0) = 1.0;

  il::luDecomposition(il::io, H);

  il::Array<double> y = {il::value, {14.0, 8.0, 25.5, 22.0}};
  il::solve(H, il::MatrixType::LowerUnitUpperNonUnit, il::io, y.Edit());

  const double eps = 1.0e-14;

  ASSERT_TRUE(il::abs(y[0] - 1.0) <= eps && il::abs(y[1] - 2.0) <= eps &&
              il::abs(y[2] - 3.0) <= eps && il::abs(y[3] - 4.0) <= eps);
}

TEST(solve, test2) {
  il::HMatrix<double> H{};
  const il::spot_t s = H.root();
  H.SetHierarchical(s);

  {
    const il::spot_t s01 = H.child(s, 0, 1);
    H.SetLowRank(s01, 4, 4, 1);
    il::Array2DEdit<double> H01A = H.AsLowRankA(s01);
    il::Array2DEdit<double> H01B = H.AsLowRankB(s01);
    for (il::int_t k = 0; k < 4; ++k) {
      H01A(k, 0) = 1.0;
      H01B(k, 0) = 1.0;
    }
  }

  {
    const il::spot_t s10 = H.child(s, 1, 0);
    H.SetLowRank(s10, 4, 4, 1);
    il::Array2DEdit<double> H10A = H.AsLowRankA(s10);
    il::Array2DEdit<double> H10B = H.AsLowRankB(s10);
    for (il::int_t k = 0; k < 4; ++k) {
      H10A(k, 0) = 1.0;
      H10B(k, 0) = 1.0;
    }
  }

  {
    const il::spot_t s00outer = H.child(s, 0, 0);
    H.SetHierarchical(s00outer);

    const il::spot_t s00 = H.child(s00outer, 0, 0);
    H.SetFullRank(s00, 2, 2);
    il::Array2DEdit<double> H00 = H.AsFullRank(s00);
    H00(0, 0) = 10.0;
    H00(1, 1) = 11.0;
    H00(0, 1) = 1.0;
    H00(1, 0) = 1.0;

    const il::spot_t s11 = H.child(s00outer, 1, 1);
    H.SetFullRank(s11, 2, 2);
    il::Array2DEdit<double> H11 = H.AsFullRank(s11);
    H11(0, 0) = 12.0;
    H11(1, 1) = 13.0;
    H11(0, 1) = 1.0;
    H11(1, 0) = 1.0;

    const il::spot_t s01 = H.child(s00outer, 0, 1);
    H.SetLowRank(s01, 2, 2, 1);
    il::Array2DEdit<double> H01A = H.AsLowRankA(s01);
    il::Array2DEdit<double> H01B = H.AsLowRankB(s01);
    H01A(0, 0) = 1.0;
    H01A(1, 0) = 1.0;
    H01B(0, 0) = 1.0;
    H01B(1, 0) = 1.0;

    const il::spot_t s10 = H.child(s00outer, 1, 0);
    H.SetLowRank(s10, 2, 2, 1);
    il::Array2DEdit<double> H10A = H.AsLowRankA(s10);
    il::Array2DEdit<double> H10B = H.AsLowRankB(s10);
    H10A(0, 0) = 1.0;
    H10A(1, 0) = 1.0;
    H10B(0, 0) = 1.0;
    H10B(1, 0) = 1.0;
  }

  {
    const il::spot_t s11outer = H.child(s, 1, 1);
    H.SetHierarchical(s11outer);

    const il::spot_t s00 = H.child(s11outer, 0, 0);
    H.SetFullRank(s00, 2, 2);
    il::Array2DEdit<double> H00 = H.AsFullRank(s00);
    H00(0, 0) = 14.0;
    H00(1, 1) = 15.0;
    H00(0, 1) = 1.0;
    H00(1, 0) = 1.0;

    const il::spot_t s11 = H.child(s11outer, 1, 1);
    H.SetFullRank(s11, 2, 2);
    il::Array2DEdit<double> H11 = H.AsFullRank(s11);
    H11(0, 0) = 16.0;
    H11(1, 1) = 17.0;
    H11(0, 1) = 1.0;
    H11(1, 0) = 1.0;

    const il::spot_t s01 = H.child(s11outer, 0, 1);
    H.SetLowRank(s01, 2, 2, 1);
    il::Array2DEdit<double> H01A = H.AsLowRankA(s01);
    il::Array2DEdit<double> H01B = H.AsLowRankB(s01);
    H01A(0, 0) = 1.0;
    H01A(1, 0) = 1.0;
    H01B(0, 0) = 1.0;
    H01B(1, 0) = 1.0;

    const il::spot_t s10 = H.child(s11outer, 1, 0);
    H.SetLowRank(s10, 2, 2, 1);
    il::Array2DEdit<double> H10A = H.AsLowRankA(s10);
    il::Array2DEdit<double> H10B = H.AsLowRankB(s10);
    H10A(0, 0) = 1.0;
    H10A(1, 0) = 1.0;
    H10B(0, 0) = 1.0;
    H10B(1, 0) = 1.0;
  }

  il::Array<double> x_initial{H.size(1), 1.0};
  il::Array<double> y = il::dot(H, x_initial);

  il::luDecomposition(il::io, H);

  il::solve(H, il::MatrixType::LowerUnitUpperNonUnit, il::io, y.Edit());

  bool result = true;
  const double eps = 1.0e-14;
  for (il::int_t i = 0; i < y.size(); ++i) {
    if (il::abs(y[i] - 1.0) >= eps) {
      result = false;
    }
  }

  ASSERT_TRUE(result);
}

TEST(solve, test3) {
  il::HMatrix<double> H{};
  const il::spot_t s = H.root();
  H.SetHierarchical(s);

  {
    const il::spot_t s01 = H.child(s, 0, 1);
    H.SetLowRank(s01, 4, 4, 1);
    il::Array2DEdit<double> H01A = H.AsLowRankA(s01);
    il::Array2DEdit<double> H01B = H.AsLowRankB(s01);
    for (il::int_t k = 0; k < 4; ++k) {
      H01A(k, 0) = 1.0 / (1 + 2 * k);
      H01B(k, 0) = 1.0 / (1 + k);
    }
  }

  {
    const il::spot_t s10 = H.child(s, 1, 0);
    H.SetLowRank(s10, 4, 4, 1);
    il::Array2DEdit<double> H10A = H.AsLowRankA(s10);
    il::Array2DEdit<double> H10B = H.AsLowRankB(s10);
    for (il::int_t k = 0; k < 4; ++k) {
      H10A(k, 0) = 1.0 / (2 + k * k);
      H10B(k, 0) = 1.0 / (1 + k * k);
    }
  }

  {
    const il::spot_t s00outer = H.child(s, 0, 0);
    H.SetHierarchical(s00outer);

    const il::spot_t s00 = H.child(s00outer, 0, 0);
    H.SetFullRank(s00, 2, 2);
    il::Array2DEdit<double> H00 = H.AsFullRank(s00);
    H00(0, 0) = 10.0;
    H00(1, 1) = 11.0;
    H00(0, 1) = 1.0 / 3.14159;
    H00(1, 0) = 1.0 / 1.414;

    const il::spot_t s11 = H.child(s00outer, 1, 1);
    H.SetFullRank(s11, 2, 2);
    il::Array2DEdit<double> H11 = H.AsFullRank(s11);
    H11(0, 0) = 12.0;
    H11(1, 1) = 13.0;
    H11(0, 1) = 1.0 / 4.14159;
    H11(1, 0) = 1.0 / 2.414;

    const il::spot_t s01 = H.child(s00outer, 0, 1);
    H.SetLowRank(s01, 2, 2, 1);
    il::Array2DEdit<double> H01A = H.AsLowRankA(s01);
    il::Array2DEdit<double> H01B = H.AsLowRankB(s01);
    H01A(0, 0) = 1.0 / 2;
    H01A(1, 0) = 1.0 / 7;
    H01B(0, 0) = 1.0 / 15;
    H01B(1, 0) = 1.0 / 12;

    const il::spot_t s10 = H.child(s00outer, 1, 0);
    H.SetLowRank(s10, 2, 2, 1);
    il::Array2DEdit<double> H10A = H.AsLowRankA(s10);
    il::Array2DEdit<double> H10B = H.AsLowRankB(s10);
    H10A(0, 0) = 1.0 / 13;
    H10A(1, 0) = 1.0 / 14;
    H10B(0, 0) = 1.0 / 25;
    H10B(1, 0) = 1.0 / 31;
  }

  {
    const il::spot_t s11outer = H.child(s, 1, 1);
    H.SetHierarchical(s11outer);

    const il::spot_t s00 = H.child(s11outer, 0, 0);
    H.SetFullRank(s00, 2, 2);
    il::Array2DEdit<double> H00 = H.AsFullRank(s00);
    H00(0, 0) = 14.0;
    H00(1, 1) = 15.0;
    H00(0, 1) = 1.0 / 6.14159;
    H00(1, 0) = 1.0 / 4.414;

    const il::spot_t s11 = H.child(s11outer, 1, 1);
    H.SetFullRank(s11, 2, 2);
    il::Array2DEdit<double> H11 = H.AsFullRank(s11);
    H11(0, 0) = 16.0;
    H11(1, 1) = 17.0;
    H11(0, 1) = 1.0 / 7.14159;
    H11(1, 0) = 1.0 / 5.414;

    const il::spot_t s01 = H.child(s11outer, 0, 1);
    H.SetLowRank(s01, 2, 2, 1);
    il::Array2DEdit<double> H01A = H.AsLowRankA(s01);
    il::Array2DEdit<double> H01B = H.AsLowRankB(s01);
    H01A(0, 0) = 1.0 / 27;
    H01A(1, 0) = 1.0 / 23;
    H01B(0, 0) = 1.0 / 21;
    H01B(1, 0) = 1.0 / 42;

    const il::spot_t s10 = H.child(s11outer, 1, 0);
    H.SetLowRank(s10, 2, 2, 1);
    il::Array2DEdit<double> H10A = H.AsLowRankA(s10);
    il::Array2DEdit<double> H10B = H.AsLowRankB(s10);
    H10A(0, 0) = 1.0 / 35;
    H10A(1, 0) = 1.0 / 39;
    H10B(0, 0) = 1.0 / 41;
    H10B(1, 0) = 1.0 / 44;
  }

  il::Array<double> x_initial{H.size(1)};
  for (il::int_t i = 0; i < x_initial.size(); ++i) {
    x_initial[i] = static_cast<double>(1 + i);
  }
  il::Array<double> y = il::dot(H, x_initial);

  il::luDecomposition(il::io, H);

  il::solve(H, il::MatrixType::LowerUnitUpperNonUnit, il::io, y.Edit());

  bool result = true;
  const double eps = 1.0e-14;
  for (il::int_t i = 0; i < y.size(); ++i) {
    if (il::abs(y[i] - (1 + i)) >= eps) {
      result = false;
    }
  }

  ASSERT_TRUE(result);
}

TEST(solve, test4) {
  const il::int_t n = 256;
  const il::int_t dim = 2;
  const il::int_t leaf_max_size = 2;

  const double radius = 1.0;
  il::Array2D<double> point{n, dim};
  for (il::int_t i = 0; i < n; ++i) {
    point(i, 0) = radius * std::cos((il::pi * (i + 0.5)) / n);
    point(i, 1) = radius * std::sin((il::pi * (i + 0.5)) / n);
  }
  const il::Cluster cluster = il::cluster(leaf_max_size, il::io, point);

  //////////////////////////////////////////////////////////////////////////////
  // Here we prepare the compression scheme of the matrix depending upon the
  // clustering and the relative diameter and distance in between the clusters
  //////////////////////////////////////////////////////////////////////////////
  //
  // And then, the clustering of the matrix, only from the geometry of the
  // points.
  // - Use eta = 0 for no compression
  // - Use eta = 1 for moderate compression
  // - Use eta = 10 for large compression
  // We have compression when Max(diam0, diam1) <= eta * distance
  // Use a large value for eta when you want everything to be Low-Rank when
  // you are outside the diagonal
  const double eta = 10.0;
  const il::Tree<il::SubHMatrix, 4> hmatrix_tree =
      il::hmatrixTree(point, cluster.partition, eta);

  //////////////////////////////////////////////////////////////////////////////
  // We build the H-Matrix
  //////////////////////////////////////////////////////////////////////////////
  const double alpha = 100.0;
  const il::Matrix M{point, alpha};
  const double epsilon = 1.0;
  il::HMatrix<double> h = il::toHMatrix(M, hmatrix_tree, epsilon);

  // First, we compute the compression ratio
  std::cout << "Compression ratio: " << il::compressionRatio(h) << std::endl;

  il::Array<double> x{h.size(0), 1.0};
  il::Array<double> y = il::dot(h, x);

  //////////////////////////////////////////////////////////////////////////////
  // We convert it to a regular matrix to compute its condition number
  //////////////////////////////////////////////////////////////////////////////
  const il::Array2D<double> full_h = il::toArray2D(h);
  il::Status status{};
  const il::LU<il::Array2D<double>> full_lu_h{full_h, il::io, status};
  status.AbortOnError();

  const double norm_full_h = il::norm(full_h, il::Norm::L1);
  const double cn = full_lu_h.conditionNumber(il::Norm::L1, norm_full_h);


  il::Array<double> y_full =   full_lu_h.solve(y);

  double relative_error_full = 0.0;
  for (il::int_t i = 0; i < y_full.size(); ++i) {
    const double re = il::abs(y_full[i] - 1.0);
    if (re > relative_error_full) {
      relative_error_full = re;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Let's play with it
  //////////////////////////////////////////////////////////////////////////////

  il::luDecomposition(il::io, h);
  il::solve(h, il::MatrixType::LowerUnitUpperNonUnit, il::io, y.Edit());

  double relative_error = 0.0;
  for (il::int_t i = 0; i < y.size(); ++i) {
    const double re = il::abs(y[i] - 1.0);
    if (re > relative_error) {
      relative_error = re;
    }
  }

  ASSERT_TRUE(relative_error <= 1.0e-10);
}
