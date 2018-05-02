#pragma once

#include <hmatrix/HMatrix.h>
#include <hmatrix/HMatrix.h>

namespace il {

il::HMatrix<double> lu(const il::HMatrix<double>& H);

void lu(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu, il::io_t,
        il::HMatrix<double>& LU);

void luForFull(il::io_t, il::Array2DEdit<double> A, il::ArrayEdit<int> pivot);
void upperRight(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu,
                il::io_t, il::HMatrix<double>& LU);
void lowerLeft(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu,
               il::io_t, il::HMatrix<double>& LU);
void lowerRight(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu,
                il::io_t, il::HMatrix<double>& LU);

void copy(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu, il::io_t,
          il::HMatrix<double>& LU);

}  // namespace il
