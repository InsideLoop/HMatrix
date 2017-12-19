#ifndef HMATRIX_HMATRIX_H
#define HMATRIX_HMATRIX_H

namespace hmat {

void hmatrix() {
  // Loop over all the leafs


  // We have three kind of matrices:
  // F: for full matrices
  // A & B: Sparse matrices are of the form A.B^t
  //
  // If the current leaf correspond to a group of cluster that corresponds to
  // a low rank matrix.
  // - Compute the number of rows of the current block
  // - Compute the number of columns of the current block
  // - Compute the memory requirement for matrices A and B. The maximum rank
  //   is given by max_low_rank which has been computed before. Is it an
  //   estimation or a firm number?
  //
  // - dim F: Size of the full matrix


}

}

#endif //HMATRIX_HMATRIX_H
