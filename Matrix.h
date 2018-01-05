#pragma once

#include <cmath>

#include <il/StaticArray2D.h>
#include <il/math.h>

#include <src/core/SegmentData.h>
#include <src/elasticity/Simplified3D.h>

namespace hmat {

template <il::int_t p>
class Matrix {
 private:
  il::int_t nb_elements_;
  il::Array2D<double> collocation_;
  hfp2d::ElasticProperties elastic_properties_;

 public:
  Matrix(const il::Array2D<double>& collocation,
         const hfp2d::ElasticProperties& elastic_properties) {
    IL_EXPECT_FAST(collocation.size(1) == 2);

    nb_elements_ = collocation.size(0);
    collocation_ = collocation;
    elastic_properties_ = elastic_properties;
  };
  il::int_t size(il::int_t k) const { return nb_elements_; }
  hfp2d::SegmentData segmentData(il::int_t i) const {
    il::StaticArray2D<double, 2, 2> Xs{};
    const double t = (il::pi / nb_elements_) / 2;
    const double cost = std::cos(t);
    const double sint = std::sin(t);

    Xs(0, 0) = collocation_(i, 0) * cost + collocation_(i, 1) * sint;
    Xs(0, 1) = - collocation_(i, 0) * sint + collocation_(i, 1) * cost;
    Xs(1, 0) = collocation_(i, 0) * cost - collocation_(i, 1) * sint;
    Xs(1, 1) = collocation_(i, 0) * sint + collocation_(i, 1) * cost;

    const il::int_t interpolation_order = 0;
    return hfp2d::SegmentData(Xs, interpolation_order);
  };
  il::StaticArray2D<double, 2, 2> operator()(il::int_t i0, il::int_t i1) const {
    IL_EXPECT_FAST(static_cast<std::size_t>(i0) <
                   static_cast<std::size_t>(nb_elements_));
    IL_EXPECT_FAST(static_cast<std::size_t>(i1) <
                   static_cast<std::size_t>(nb_elements_));

    const double ker_options = 1.0;
    return hfp2d::normal_shear_stress_kernel_s3d_dp0_dd_nodal(
        segmentData(i1), segmentData(i0), 0, 0, elastic_properties_,
        ker_options);
  }
};

// template <il::int_t p>
// class Matrix {
// private:
//  il::int_t n0_;
//  il::int_t n1_;
//  double x0_;
//  double x1_;
//
// public:
//  Matrix(il::int_t n0, il::int_t n1, double x0, double x1) {
//    n0_ = n0;
//    n1_ = n1;
//  }
//  il::int_t size(il::int_t k) const {
//    IL_EXPECT_FAST(k >= 0 && k <= 2);
//    return (k == 0) ? n0_ : n1_;
//  }
//  il::StaticArray2D<double, p, p> operator()(il::int_t i0, il::int_t i1) const
//  {
//    IL_EXPECT_FAST(static_cast<std::size_t>(i0) <
//        static_cast<std::size_t>(n0_));
//    IL_EXPECT_FAST(static_cast<std::size_t>(i1) <
//        static_cast<std::size_t>(n1_));
//
//    const double xi = x0_ + i0 * (1.0 / n0_);
//    const double xj = x1_ + i1 * (1.0 / n1_);
//    const double u = std::exp(-il::ipow<2>(xi - xj));
//    return il::StaticArray2D<double, p, p>{
//        il::value, {{u, 0.0}, {0.0, 2 * u}}};
//  }
//};

// template <il::int_t p>
// class Matrix {
// private:
//  il::int_t n_;
//  double lambda_;
//  il::Array2D<double> node_; //{n, dim};
//
// public:
//  Matrix(il::Array2D<double> node, double lambda) : node_{std::move(node)} {
//    n_ = node_.size(0);
//    lambda_ = lambda;
//  }
//  il::int_t size(il::int_t k) const {
//    IL_EXPECT_FAST(k >= 0 && k <= 2);
//    return n_;
//  }
//  il::StaticArray2D<double, p, p> operator()(il::int_t i0, il::int_t i1) const
//  {
//    IL_EXPECT_FAST(static_cast<std::size_t>(i0) <
//    static_cast<std::size_t>(n_)); IL_EXPECT_FAST(static_cast<std::size_t>(i1)
//    < static_cast<std::size_t>(n_));
//
//    const double xi = i0 * (1.0 / n_);
//    const double xj = i1 * (1.0 / n_);
//    const double u = std::exp(-lambda_ * il::ipow<2>(xi - xj));
//    return il::StaticArray2D<double, p, p>{il::value, {{u}}};
//  }
//};

}  // namespace hmat
