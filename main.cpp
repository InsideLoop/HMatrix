#include <iostream>
#include <random>

#include <il/Timer.h>

#include "adaptiveCrossApproximation.h"

int main() {
  const il::int_t p = 2;
  const il::int_t n0 = 10;
  const il::int_t n1 = 10;
  const double x0 = 0.0;
  const double x1 = 5.0;
  hmat::Matrix<p> M{n0, n1, x0, x1};

  const double epsilon = 0.01;
  hmat::SmallRank<double> approx = hmat::adaptiveCrossApproximation(M, epsilon);

  return 0;
}