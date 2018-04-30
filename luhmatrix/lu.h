#pragma once

#include <hmatrix/HMatrix.h>
#include <luhmatrix/LuHMatrix.h>

namespace il {

il::LuHMatrix<double, int> lu(const il::HMatrix<double>& H);

void lu(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu, il::io_t,
        il::LuHMatrix<double, int>& LU);

void luForFull(il::io_t, il::Array2DEdit<double> A, il::ArrayEdit<int> pivot);
void upperRight(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu,
                il::io_t, il::LuHMatrix<double, int>& LU);
void lowerLeft(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu,
               il::io_t, il::LuHMatrix<double, int>& LU);
void lowerRight(const il::HMatrix<double>& H, il::spot_t sh, il::spot_t slu,
               il::io_t, il::LuHMatrix<double, int>& LU);

void copy(il::Array2DView<double> A, il::io_t, il::Array2DEdit<double> B);

}  // namespace il
