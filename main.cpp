#include <il/math.h>
#include <il/Tree.h>

#include <cluster/cluster.h>
#include <HMatrix/HMatrix.h>
#include <HMatrix/HMatrixUtils.h>
#include <HMatrix/SubMatrix.h>
#include <compression/toHMatrix.h>
#include <Matrix.h>
#include <linearAlgebra/blas/dot.h>

void hmatrix();
void clustering();


int main() {
  clustering();

  return 0;
}

void clustering() {
  const il::int_t n = 8;
  const il::int_t dim = 2;
  const il::int_t leaf_max_size = 2;

  const double radius = 1.0;

  il::Array2D<double> node{n + 1, dim};
  for (il::int_t i = 0; i < n + 1; ++i) {
    node(i, 0) = radius * std::cos((il::pi * i) / n);
    node(i, 1) = radius * std::sin((il::pi * i) / n);
  }
  il::Array2D<double> collocation{n, dim};
  for (il::int_t i = 0; i < n; ++i) {
    collocation(i, 0) = radius * std::cos((il::pi * (i + 0.5)) / n);
    collocation(i, 1) = radius * std::sin((il::pi * (i + 0.5)) / n);
  }

  const il::Cluster cluster = il::cluster(leaf_max_size, il::io, collocation);

  // And then, the clustering of the matrix, only from the geometry of the
  // points.
  // - Use eta = 0 for no compression
  // - Use eta = 1 for moderate compression
  // - Use eta = 10 for large compression
  // We have compression when Max(diam0, diam1) <= eta * distance
  // Use a large value for eta when you want everything to be Low-Rank when
  // you are outside the diagonal
  const double eta = 10.0;
  const il::Tree<il::SubMatrix, 4> hmatrix_tree =
      il::hmatrixTree(collocation, cluster.partition, eta);

  //  We build the H-Matrix
  const double alpha = 1.0;
  const il::Matrix<double> M{collocation, alpha};
  const double epsilon = 0.0;
  const il::HMatrix<double> h = il::compress(M, hmatrix_tree, epsilon);
  const double cr = il::compressionRatio(h);

  std::cout << "Compression ratio: " << cr << std::endl;

  const il::Array2D<double> h0 = il::toArray2D(h);
  il::Array2D<double> h1{n, n};
  M.set(0, 0, il::io, h1.Edit());

  il::Array2D<double> diff{n, n};
  for (il::int_t i1 = 0; i1 < n; ++i1) {
    for (il::int_t i0 = 0; i0 < n; ++i0) {
      diff(i0, i1) = h0(i0, i1) - h1(i0, i1);
    }
  }

  il::Array<double> x{n, 1.0};
  il::Array<double> y = il::dot(h, x);
  il::Array<double> y1 = il::dot(h1, x);

  std::cout << "Finished" << std::endl;
}

void hmatrix() {
  const il::int_t n = 3;

  il::HMatrix<double> H{};
  il::spot_t s = H.root();
  H.SetHierarchical(s);
  H.SetFullRank(H.child(s, 0, 0), 3, 3);
  H.SetFullRank(H.child(s, 1, 1), 3, 3);
  H.SetLowRank(H.child(s, 1, 0), 3, 3, 1);
  H.SetLowRank(H.child(s, 0, 1), 3, 3, 1);

  const double cr = il::compressionRatio(H);
}