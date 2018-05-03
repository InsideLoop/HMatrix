#pragma once

#include <hmatrix/HMatrix.h>
#include <hmatrix/HMatrix.h>

namespace il {

void luDecomposition(il::io_t, il::HMatrix<double>& H);
void luDecomposition(il::spot_t s, il::io_t, il::HMatrix<double>& H);



il::HMatrix<double> lu(const il::HMatrix<double>& H);

void lu(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu, il::io_t,
        il::HMatrix<double>& LU);

void upperRight(il::spot_t s, il::io_t, il::HMatrix<double>& H);
void lowerLeft(il::spot_t s, il::io_t, il::HMatrix<double>& H);
void lowerRight(il::spot_t s, il::io_t, il::HMatrix<double>& H);

//void copy(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu, il::io_t,
//          il::HMatrix<double>& LU);

}  // namespace il
