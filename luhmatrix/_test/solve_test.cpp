#include <gtest/gtest.h>

#include <hmatrix/HMatrix.h>
#include <luhmatrix/LuHMatrix.h>
#include <luhmatrix/lu.h>
#include <luhmatrix/solve.h>

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

  il::LuHMatrix<double, int> LUH = il::lu(H);

  il::Array<double> y = {il::value, {10.0, 16.0, 18.0, 27.0}};
  il::solve(LUH, LUH.root(), il::io, y.Edit());

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

  il::LuHMatrix<double, int> LUH = il::lu(H);

  il::Array<double> y = {il::value, {14.0, 8.0, 25.5, 22.0}};
  il::solve(LUH, LUH.root(), il::io, y.Edit());

  const double eps = 1.0e-10;

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

  il::LuHMatrix<double, int> LUH = il::lu(H);

  ASSERT_TRUE(true);
}
