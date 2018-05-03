#pragma once

#include <hmatrix/HMatrix.h>
#include <hmatrix/HMatrix.h>

namespace il {

void luDecomposition(il::io_t, il::HMatrix<double>& H);
void luDecomposition(il::spot_t s, il::io_t, il::HMatrix<double>& H);

}  // namespace il
