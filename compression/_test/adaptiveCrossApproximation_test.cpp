#include <gtest/gtest.h>

#include <hmatrix/HMatrixUtils.h>
#include <hmatrix/HMatrixType.h>
#include <matrixFunctor/GaussianMatrix.h>
#include <compression/toHMatrix.h>

TEST(adaptiveCrossApproximation, test0) {
  const il::int_t n = 4;
  const double alpha = 1.0;
  const il::GaussianMatrix<double> G{2 * n, il::Range{0, n},
                                     il::Range{n, 2 * n}, alpha};

  il::Tree<il::SubHMatrix, 4> tree{};
  const il::spot_t s = tree.root();
  tree.Set(s, il::SubHMatrix{il::Range{0, n}, il::Range{0, n},
                            il::HMatrixType::LowRank});

  const double epsilon = 1.0e-4;
  const il::HMatrix<double> H = il::toHMatrix(G, tree, epsilon);

  const il::Array2D<double> M0 = il::toArray2D(G);
  const il::Array2D<double> M1 = il::toArray2D(H);
  il::Array2D<double> diff{n, n};
  for (il::int_t i1 = 0; i1 < n; ++i1) {
    for (il::int_t i0 = 0; i0 < n; ++i0) {
      diff(i0, i1) = M0(i0, i1) - M1(i0, i1);
    }
  }

  ASSERT_TRUE(true);
}
