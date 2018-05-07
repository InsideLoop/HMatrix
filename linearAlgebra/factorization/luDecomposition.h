#pragma once

#include <hmatrix/HMatrix.h>

namespace il {

void luDecomposition(double epsilon, il::io_t, il::HMatrix<double>& H);
void luDecomposition(double epsilon, il::spot_t s, il::io_t,
                     il::HMatrix<double>& H);

}  // namespace il
