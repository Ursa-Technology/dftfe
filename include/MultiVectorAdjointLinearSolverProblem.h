// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
// @author Vishal Subramanian, Bikask Kanungo
//

#ifndef DFTFE_MULTIVECTORADJOINTLINEARSOLVERPROBLEM_H
#define DFTFE_MULTIVECTORADJOINTLINEARSOLVERPROBLEM_H


#include "MultiVectorLinearSolverProblem.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class MultiVectorAdjointLinearSolverProblem : public  MultiVectorLinearSolverProblem<memorySpace>
  {
  public:

    MultiVectorAdjointLinearSolverProblem(const MPI_Comm &mpi_comm_parent,
                                    const MPI_Comm &mpi_comm_domain);

    // Destructor
    ~MultiVectorAdjointLinearSolverProblem();

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
    precondition_JacobiSqrt(dftfe::linearAlgebra::MultiVector<T,
                                                               memorySpace> &      dst,
                            const dftfe::linearAlgebra::MultiVector<T,
                                                                    memorySpace> &src,
                            const double omega) const override ;

    void reinit(
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
        basisOperationsPtr,
      KohnShamHamiltonianOperator<
        memorySpace> & ksHamiltonianObj,
      const dealii::AffineConstraints<double> &constraintMatrix,
      const unsigned int                       matrixFreeVectorComponent,
      const unsigned int matrixFreeQuadratureComponentRhs,
      const bool              isComputeDiagonalA);

    void multiVectorDotProdQuadWise(const dftfe::linearAlgebra::MultiVector<T,
                                                                 memorySpace> &      vec1,
                               const dftfe::linearAlgebra::MultiVector<T,
                                                                       memorySpace> &vec2,
                               dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST>&
                                 dotProductOutputHost);

  private :

    /// data members for the mpi implementation
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;

    bool d_isComputeDiagonalA;


    /// the vector that stores the output obtained by solving the poisson
    /// problem
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> *d_blockedXPtr, *d_psiMemSpace;


    dftfe::linearAlgebra::MultiVector<double,
                                      memorySpace> d_diagonalSqrtA;

    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;

    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      d_basisOperationsPtr;

    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    unsigned int d_matrixFreeQuadratureComponentRhs;
    unsigned int d_matrixFreeVectorComponent;
    unsigned int d_blockSize;
    unsigned int d_locallyOwnedSize,d_numberDofsPerElement,d_numCells,d_numQuadsPerCell;

    KohnShamHamiltonianOperator<memorySpace> d_ksOperatorPtr;
    dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
      d_mapNodeIdToProcId;
    dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
      d_mapQuadIdToProcId;

    dftfe::utils::MemoryStorage<double, memorySpace> tempOutputDotProdMemSpace, oneBlockSizeMemSpace;

    dftfe::linearAlgebra::MultiVector<double,
                                      memorySpace> d_onesDevice,d_onesQuadDevice;
 dftfe::linearAlgebra::MultiVector<double,
                                       memorySpace> d_rhsMemSpace;
    dftfe::utils::MemoryStorage<T, memorySpace> vec1QuadValues, vec2QuadValues,vecOutputQuadValues;

    dftfe::utils::MemoryStorage<double, memorySpace>
      d_RMatrixMemSpace, d_MuMatrixMemSpace;

    dftfe::utils::MemoryStorage<double, memorySpace>
      d_effectiveOrbitalOccupancyMemSpace;

    std::vector<std::vector<unsigned int>> d_degenerateState;
    std::vector<double>       d_eigenValues;

    std::vector<unsigned int> d_vectorList;

    dftfe::utils::MemoryStorage<unsigned int, memorySpace>
      d_vectorListMemSpace;

    dftfe::utils::MemoryStorage<unsigned int, memorySpace>
    d_4xeffectiveOrbitalOccupancyMemSpace;

    dftfe::utils::MemoryStorage<double , memorySpace> d_inputJxWMemSpace;

    dftfe::utils::MemoryStorage<double , memorySpace>
    d_cellWaveFunctionQuadMatrixMemSpace;
    dftfe::utils::MemoryStorage<double , memorySpace>
    d_MuMatrixMemSpaceCellWise;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> d_MuMatrixHost;
  };
}

#endif // DFTFE_MULTIVECTORADJOINTLINEARSOLVERPROBLEM_H
