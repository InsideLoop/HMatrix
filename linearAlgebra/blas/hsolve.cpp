#include <linearAlgebra/blas/hblas.h>
#include <linearAlgebra/blas/hsolve.h>
#include <il/linearAlgebra/dense/blas/solve.h>
#include <il/linearAlgebra/Matrix.h>

namespace il {

void solve(const il::HMatrix<double>& lu, il::MatrixType type, il::io_t,
           il::ArrayEdit<double> xy) {
  IL_EXPECT_FAST(type == il::MatrixType::LowerUnitUpperNonUnit);

  il::solve(lu, lu.root(), il::io, xy);
}

void solve(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
           il::ArrayEdit<double> x) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(1, s) == x.size());

  solveLower(lu, s, il::io, x);
  solveUpper(lu, s, il::io, x);
}

void solveLower(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x) {
  il::Array2DEdit<double> x_as_matrix{x.Data(), x.size(), 1, x.size(), 0, 0};
  solveLower(lu, s, il::io, x_as_matrix);
}

void solveLower(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::Array2DEdit<double> A) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(0, s) == A.size(0));

  if (lu.isFullLu(s)) {
    il::ArrayView<int> pivot = lu.asFullLuPivot(s);
    il::Array2DView<double> full = lu.asFullLu(s);
    il::solve(pivot, full, il::MatrixType::LowerUnit, il::io, A);
  } else if (lu.isHierarchical(s)) {
    const il::spot_t s00 = lu.child(s, 0, 0);
    const il::spot_t s10 = lu.child(s, 1, 0);
    const il::spot_t s11 = lu.child(s, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    solveLower(lu, s00, il::io,
               A.Edit(il::Range{0, n0}, il::Range{0, A.size(1)}));
    il::blas(-1.0, lu, s10, il::MatrixType::Regular,
             A.view(il::Range{0, n0}, il::Range{0, A.size(1)}), 1.0, il::io,
             A.Edit(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)}));
    solveLower(lu, s11, il::io,
               A.Edit(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)}));
  } else {
    IL_UNREACHABLE;
  }
}

void solveLower(const il::HMatrix<double>& lu, il::spot_t slu, il::spot_t s,
                il::io_t, il::HMatrix<double>& A) {
  const il::int_t debug_n0 = lu.size(0, slu);
  const il::int_t debug_n1 = lu.size(1, slu);
  IL_EXPECT_MEDIUM(lu.size(0, slu) == lu.size(1, slu));
  IL_EXPECT_MEDIUM(lu.size(1, slu) == A.size(0, s));

  if (lu.isFullLu(slu)) {
    il::ArrayView<int> pivot = lu.asFullLuPivot(slu);
    il::Array2DView<double> lower = lu.asFullLu(slu);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::solve(pivot, lower, il::MatrixType::LowerUnit, il::io, full);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> full = A.AsLowRankA(s);
      il::solve(pivot, lower, il::MatrixType::LowerUnit, il::io, full);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else if (lu.isHierarchical(slu)) {
    const il::spot_t s00 = lu.child(slu, 0, 0);
    const il::spot_t s10 = lu.child(slu, 1, 0);
    const il::spot_t s11 = lu.child(slu, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s10);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::Array2DEdit<double> full0 =
          full.Edit(il::Range{0, n0}, il::Range{0, full.size(1)});
      il::Array2DEdit<double> full1 =
          full.Edit(il::Range{n0, n0 + n1}, il::Range{0, full.size(1)});
      il::solveLower(lu, s00, il::io, full0);
      il::blas(-1.0, lu, s10, il::MatrixType::LowerUnit, full0, 1.0, il::io,
               full1);
      il::solveLower(lu, s11, il::io, full1);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> lowA = A.AsLowRankA(s);
      il::Array2DEdit<double> lowA0 =
          lowA.Edit(il::Range{0, n0}, il::Range{0, lowA.size(1)});
      il::Array2DEdit<double> lowA1 =
          lowA.Edit(il::Range{n0, n0 + n1}, il::Range{0, lowA.size(1)});
      il::solveLower(lu, s00, il::io, lowA0);
      il::blas(-1.0, lu, s10, il::MatrixType::Regular, lowA0, 1.0, il::io,
               lowA1);
      il::solveLower(lu, s11, il::io, lowA1);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else {
    IL_UNREACHABLE;
  }
}

void solveUpper(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::ArrayEdit<double> x) {
  il::Array2DEdit<double> x_as_matrix{x.Data(), x.size(), 1, x.size(), 0, 0};
  solveUpper(lu, s, il::io, x_as_matrix);
}

void solveUpper(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                il::Array2DEdit<double> A) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(1, s) == A.size(0));

  if (lu.isFullLu(s)) {
    il::Array2DView<double> full_lu = lu.asFullLu(s);
    il::solve(full_lu, il::MatrixType::UpperNonUnit, il::io, A);
  } else if (lu.isHierarchical(s)) {
    const il::spot_t s00 = lu.child(s, 0, 0);
    const il::spot_t s01 = lu.child(s, 0, 1);
    const il::spot_t s11 = lu.child(s, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    solveUpper(lu, s11, il::io,
               A.Edit(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)}));
    il::blas(-1.0, lu, s01, il::MatrixType::Regular,
             A.view(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)}), 1.0,
             il::io, A.Edit(il::Range{0, n0}, il::Range{0, A.size(1)}));
    solveUpper(lu, s00, il::io,
               A.Edit(il::Range{0, n0}, il::Range{0, A.size(1)}));
  } else {
    IL_UNREACHABLE;
  }
}

void solveUpper(const il::HMatrix<double>& lu, il::spot_t slu, il::spot_t s,
                il::io_t, il::HMatrix<double>& A) {
  IL_EXPECT_MEDIUM(lu.size(0, slu) == lu.size(1, slu));
  IL_EXPECT_MEDIUM(lu.size(1, slu) == A.size(0, s));

  if (lu.isFullRank(slu)) {
    il::ArrayView<int> pivot = lu.asFullLuPivot(slu);
    il::Array2DView<double> upper = lu.asFullRank(slu);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::solve(upper, il::MatrixType::UpperNonUnit, il::io, full);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> full = A.AsLowRankA(s);
      il::solve(upper, il::MatrixType::UpperNonUnit, il::io, full);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else if (lu.isHierarchical(slu)) {
    const il::spot_t s00 = lu.child(slu, 0, 0);
    const il::spot_t s01 = lu.child(slu, 0, 1);
    const il::spot_t s11 = lu.child(slu, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::Array2DEdit<double> full0 =
          full.Edit(il::Range{0, n0}, il::Range{0, full.size(1)});
      il::Array2DEdit<double> full1 =
          full.Edit(il::Range{n0, n0 + n1}, il::Range{0, full.size(1)});

      il::solveUpper(lu, s11, il::io, full1);
      il::blas(-1.0, lu, s01, il::MatrixType::Regular, full1, 1.0, il::io,
               full0);
      il::solveUpper(lu, s00, il::io, full0);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> lowA = A.AsLowRankA(s);
      il::Array2DEdit<double> lowA0 =
          lowA.Edit(il::Range{0, n0}, il::Range{0, lowA.size(1)});
      il::Array2DEdit<double> lowA1 =
          lowA.Edit(il::Range{n0, n0 + n1}, il::Range{0, lowA.size(1)});
      il::solveUpper(lu, s11, il::io, lowA1);
      il::blas(-1.0, lu, s01, il::MatrixType::Regular, lowA1, 1.0, il::io,
               lowA0);
      il::solveLower(lu, s00, il::io, lowA0);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else {
    IL_UNREACHABLE;
  }
}

void solveUpperRight(const il::HMatrix<double>& lu, il::spot_t s, il::io_t,
                         il::Array2DEdit<double> A) {
  IL_EXPECT_MEDIUM(lu.size(0, s) == lu.size(1, s));
  IL_EXPECT_MEDIUM(lu.size(0, s) == A.size(0));

  if (lu.isFullLu(s)) {
    il::Array2DView<double> full_lu = lu.asFullLu(s);
    il::solve(full_lu, il::MatrixType::UpperNonUnit, il::Dot::Transpose, il::io,
              A);
  } else if (lu.isHierarchical(s)) {
    const il::spot_t s00 = lu.child(s, 0, 0);
    const il::spot_t s01 = lu.child(s, 0, 1);
    const il::spot_t s11 = lu.child(s, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    il::Array2DEdit<double> A0 =
        A.Edit(il::Range{0, n0}, il::Range{0, A.size(1)});
    il::Array2DEdit<double> A1 =
        A.Edit(il::Range{n0, n0 + n1}, il::Range{0, A.size(1)});
    solveUpperRight(lu, s00, il::io, A0);
    il::blas(-1.0, lu, s01, il::Dot::Transpose, A0, 1.0, il::io, A1);
    solveUpperRight(lu, s11, il::io, A1);
  } else {
    IL_UNREACHABLE;
  }
}

void solveUpperRight(const il::HMatrix<double>& lu, il::spot_t slu,
                         il::spot_t s, il::io_t, il::HMatrix<double>& A) {
  IL_EXPECT_MEDIUM(lu.size(0, slu) == lu.size(1, slu));
  IL_EXPECT_MEDIUM(lu.size(0, slu) == A.size(0, s));

  if (lu.isFullLu(slu)) {
    il::Array2DView<double> upper = lu.asFullLu(slu);
    if (A.isFullRank(s)) {
      il::abort();
      // There should be a bug here
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::solve(upper, il::MatrixType::UpperNonUnit, il::Dot::Transpose, il::io,
                full);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> ab = A.AsLowRankB(s);
      il::solve(upper, il::MatrixType::UpperNonUnit, il::Dot::Transpose, il::io,
                ab);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else if (lu.isHierarchical(slu)) {
    const il::spot_t s00 = lu.child(slu, 0, 0);
    const il::spot_t s01 = lu.child(slu, 0, 1);
    const il::spot_t s11 = lu.child(slu, 1, 1);
    const il::int_t n0 = lu.size(0, s00);
    const il::int_t n1 = lu.size(0, s11);
    if (A.isFullRank(s)) {
      il::Array2DEdit<double> full = A.AsFullRank(s);
      il::Array2DEdit<double> full0 =
          full.Edit(il::Range{0, n0}, il::Range{0, full.size(1)});
      il::Array2DEdit<double> full1 =
          full.Edit(il::Range{n0, n0 + n1}, il::Range{0, full.size(1)});

      il::solveUpperRight(lu, s00, il::io, full0);
      il::blas(-1.0, lu, s01, il::Dot::Transpose, full0, 1.0, il::io, full1);
      il::solveUpperRight(lu, s11, il::io, full1);
    } else if (A.isLowRank(s)) {
      il::Array2DEdit<double> lowb = A.AsLowRankB(s);
      il::Array2DEdit<double> lowb0 =
          lowb.Edit(il::Range{0, n0}, il::Range{0, lowb.size(1)});
      il::Array2DEdit<double> lowb1 =
          lowb.Edit(il::Range{n0, n0 + n1}, il::Range{0, lowb.size(1)});
      il::solveUpperRight(lu, s00, il::io, lowb0);
      il::blas(-1.0, lu, s01, il::Dot::Transpose, lowb0, 1.0, il::io, lowb1);
      il::solveUpperRight(lu, s11, il::io, lowb1);
    } else {
      IL_EXPECT_MEDIUM(A.isHierarchical(s));
      IL_UNREACHABLE;
      // Not sure if this needs to be implemented
    }
  } else {
    IL_UNREACHABLE;
  }
}



}  // namespace il