.. role:: cpp(code)

    :language: cpp

Building the matrix
===================

Overview
--------

Once we have the structure of our H-Matrix given by the
:cpp:`il::Tree<il::SubHMatrix, 4>` object, we need a functor that generates the
coefficients of the matrix. For instance, imagine that we have n points in
the plane, and the matrix we want to generate has the coefficients

.. math::

    A_{i_0, i_1} = e^{- \alpha \text{d}^2(P_{i_0}, P_{i_1})}


For that, we need to create a functor. It should inherit from the class
:cpp:`il::MatrixGenerator<double>` and therefore provide the following
interface:

- :cpp:`il::int_t size(il::int_t d) const`: This function should return the number
  of rows of the matrix for :cpp:`d = 0` and the number of columns of the
  matrix for :cpp:`d = 1`.

- :cpp:`il::int_t blockSize() const`: This function should return the size of
  the blocks of the matrix for vector problems. Here the interaction in between
  the points :math:`P_{i_0}` and :math:`P_{i_1}` is represented by a scalar.
  Therefore, it should return 1.

- :cpp:`il::int_t sizeAsBlocks(il::int_t d) const`: This function should return
  the number of block rows of the matrix and the number of block columns of the
  matrix. In our case, where the block size is 1, this function returns the
  same quantity as :cpp:`size`.

- :cpp:`set(il::int_t b0, il::int_t b1, il::io_t, il::Array2DEdit<double> M) const`
  This function, should fill a submatrix of our matrix. The top left corner
  of this submatrix should be at (block)-row :math:`b_0` and (block)-column
  :math:`b_1` and our object should fill the given matrix.

In our, case, here is the full code for the program that represents our matrix:
rrayFunctor/MatrixGenerator.h>

.. code-block:: cpp

    #include <arrayFunctor/MatrixGenerator.h>

    class Matrix : public MatrixGenerator<double> {
     private:
      il::Array2D<double> point_;
      double alpha_;

     public:
      Matrix(il::Array2D<double> point, double alpha);
      il::int_t size(il::int_t d) const override;
      il::int_t blockSize() const override;
      il::int_t sizeAsBlocks(il::int_t d) const override;
      void set(il::int_t b0, il::int_t b1, il::io_t,
               il::Array2DEdit<double> M) const override;
    };

    Matrix::Matrix(il::Array2D<double> point, double alpha)
        : point_{std::move(point)}, alpha_{alpha} {
      IL_EXPECT_FAST(point_.size(1) == 2);
    };

    il::int_t Matrix::size(il::int_t d) const {
      IL_EXPECT_MEDIUM(d == 0 || d == 1);

      return point_.size(0);
    };

    il::int_t Matrix::blockSize() const { return 1; }

    il::int_t Matrix::sizeAsBlocks(il::int_t d) const {
      IL_EXPECT_MEDIUM(d == 0 || d == 1);

      return point_.size(0);
    }

    void Matrix::set(il::int_t b0, il::int_t b1, il::io_t,
                     il::Array2DEdit<double> M) const {
      IL_EXPECT_MEDIUM(M.size(0) % blockSize() == 0);
      IL_EXPECT_MEDIUM(M.size(1) % blockSize() == 0);
      IL_EXPECT_MEDIUM(b0 + M.size(0) / blockSize() <= point_.size(0));
      IL_EXPECT_MEDIUM(b1 + M.size(1) / blockSize() <= point_.size(0));

      for (il::int_t j1 = 0; j1 < M.size(1); ++j1) {
        for (il::int_t j0 = 0; j0 < M.size(0); ++j0) {
          il::int_t k0 = b0 + j0;
          il::int_t k1 = b1 + j1;
          const double dx = point_(k0, 0) - point_(k1, 0);
          const double dy = point_(k0, 1) - point_(k1, 1);
          M(j0, j1) = std::exp(-alpha_ * (dx * dx + dy * dy));
        }
      }
    }

Once we are given the matrix and we have the hierarchical tree structure of
our H-Matrix, we can easily construct our H-Matrix with the following function:

.. code-block:: cpp

    const double epsilon = 0.1;
    const il::HMatrix<double> h = il::toHMatrix(M, tree, epsilon);

where :cpp:`epsilon` is the threshold used in our adaptive cross-approximation
algorithn. In a nutshell, it is used to stop the low-rank approximation
algorithm, when using an approximation with a larger rank would not change
the approximation by a relative difference of more that :cpp:`epsilon`.
