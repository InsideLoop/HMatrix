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
