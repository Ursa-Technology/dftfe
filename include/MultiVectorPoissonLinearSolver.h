// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//

#ifndef DFTFE_MULTIVECTORPOISSONLINEARSOLVER_H
#define DFTFE_MULTIVECTORPOISSONLINEARSOLVER_H

#include "MultiVectorLinearSolverProblem.h"

namespace dftfe
{

  template <dftfe::utils::MemorySpace memorySpace>
  class MultiVectorPoissonLinearSolverProblem : public MultiVectorLinearSolverProblem<memorySpace>
  {
  public:

    // Constructor
    MultiVectorPoissonLinearSolverProblem( const MPI_Comm &mpi_comm_parent,
                                          const MPI_Comm &mpi_comm_domain);

    // Destructor
    ~MultiVectorPoissonLinearSolverProblem();
    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    template <typename T>
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> &
    computeRhs(
               dftfe::linearAlgebra::MultiVector<T,
                                                 memorySpace> &       NDBCVec,
               dftfe::linearAlgebra::MultiVector<T,
                                                 memorySpace> &       outputVec,
               unsigned int                      blockSizeInput) override;

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    template <typename T>
    void vmult(dftfe::linearAlgebra::MultiVector<T,
                                            memorySpace> &Ax,
          dftfe::linearAlgebra::MultiVector<T,
                                            memorySpace> &x,
          unsigned int               blockSize) override;

    /**
     * @brief Apply the constraints to the solution vector.
     *
     */
    void
    distributeX() override;

    /**
     * @brief Jacobi preconditioning function.
     *
     */
    template <typename T>
    void
    precondition_Jacobi(dftfe::linearAlgebra::MultiVector<T,
                                                          memorySpace> &      dst,
                        const dftfe::linearAlgebra::MultiVector<T,
                                                                memorySpace> &src,
                        const double                     omega) const  override;


    /**
     * @brief Apply square-root of the Jacobi preconditioner function.
     *
     */
    template <typename T>
    void
    precondition_JacobiSqrt(ddftfe::linearAlgebra::MultiVector<T,
                                                               memorySpace> &      dst,
                            const dftfe::linearAlgebra::MultiVector<T,
                                                                    memorySpace> &src,
                            const double omega) const override ;

    template <typename T>
    void
      setDataForRhsVec(dftfe::utils::MemoryStorage<double, memorySpace>& inputQuadData);



  private:

    void computeMeanValueConstraint();

    bool d_isComputeDiagonalA;

    bool d_isMeanValueConstraintComputed;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    /// the vector that stores the output obtained by solving the poisson
    /// problem
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> *d_blockedXPtr, *d_blockedNDBCPtr, d_rhsVec;

    unsigned int d_matrixFreeQuadratureComponentRhs;
    unsigned int d_matrixFreeVectorComponent;
    unsigned int d_blockSize;

    dftfe::linearAlgebra::MultiVector<double,
                                      memorySpace> d_diagonalA, d_diagonalSqrtA;

    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;

    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      d_basisOperationsPtr;

    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    dftfe::utils::MemoryStorage<double, memorySpace> & d_cellStiffnessMatrix;

    dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
      d_constraintsInfo;

    /// data members for the mpi implementation
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
    size_type d_locallyOwnedSize;

    size_type d_numberDofsPerElement;
    size_type d_numCells;

    size_type d_inc;
    double d_negScalarCoeffAlpha;
    double d_scalarCoeffAlpha;
    double d_beta;
    double d_alpha;
    char d_transA;
    char d_transB;


    dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
      d_mapNodeIdToProcId;

    dftfe::utils::MemoryStorage<double, memorySpace>
      d_xCellLLevelNodalData, d_AxCellLLevelNodalData;

    size_type d_cellsBlockSizeVmult;

    dftfe::utils::MemoryStorage<double, memorySpace> *d_rhsQuadDataPtr;


  };

}


#endif // DFTFE_MULTIVECTORPOISSONLINEARSOLVER_H
