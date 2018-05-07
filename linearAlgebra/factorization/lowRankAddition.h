#pragma once

#include <hmatrix/LowRank.h>

namespace il {

il::LowRank<double> lowRankAddition(double epsilon, double alpha,
                                    il::Array2DView<double> aa,
                                    il::Array2DView<double> ab, double beta,
                                    il::Array2DView<double> ba,
                                    il::Array2DView<double> bb);
}