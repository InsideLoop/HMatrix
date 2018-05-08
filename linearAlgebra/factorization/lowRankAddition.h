#pragma once

#include <complex>
#include <hmatrix/LowRank.h>

namespace il {

il::LowRank<double> lowRankAddition(double epsilon, double alpha,
                                    il::Array2DView<double> aa,
                                    il::Array2DView<double> ab, double beta,
                                    il::Array2DView<double> ba,
                                    il::Array2DView<double> bb);

il::LowRank<std::complex<double>> lowRankAddition(
    double epsilon, std::complex<double> alpha,
    il::Array2DView<std::complex<double>> aa,
    il::Array2DView<std::complex<double>> ab, std::complex<double> beta,
    il::Array2DView<std::complex<double>> ba,
    il::Array2DView<std::complex<double>> bb);

}  // namespace il