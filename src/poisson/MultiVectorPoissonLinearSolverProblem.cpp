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
// @author Vishal Subramanian
//

#include "MultiVectorPoissonLinearSolverProblem.h"

namespace dftfe
{
  template<dftfe::utils::MemorySpace memorySpace>
  MultiVectorPoissonLinearSolverProblem<memorySpace>::MultiVectorPoissonLinearSolverProblem( const MPI_Comm &mpi_comm_parent,
                                                                                            const MPI_Comm &mpi_comm_domain)
    : mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_isComputeDiagonalA               = true;
    d_constraintMatrixPtr              = NULL;
    d_blockedXPtr              = NULL;
    d_blockedNDBCPtr               = NULL;
    d_matrixFreeQuadratureComponentRhs = -1;
    d_matrixFreeVectorComponent        = -1;
    d_blockSize                        = 0;
    d_diagonalA.reinit(0);
    d_diagonalSqrtA.reinit(0);
    d_isMeanValueConstraintComputed = false;
  }

  template<dftfe::utils::MemorySpace memorySpace>
  MultiVectorPoissonLinearSolverProblem<memorySpace>::~MultiVectorPoissonLinearSolverProblem()
  {

  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::clear()
  {
    d_isComputeDiagonalA               = true;
    d_constraintMatrixPtr              = NULL;
    d_blockedXPtr              = NULL;
    d_blockedNDBCPtr               = NULL;
    d_matrixFreeQuadratureComponentRhs = -1;
    d_matrixFreeVectorComponent        = -1;
    d_blockSize                        = 0;
    d_diagonalA.reinit(0);
    d_diagonalSqrtA.reinit(0);
    d_isMeanValueConstraintComputed = false;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::reinit(
  std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
    BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
      basisOperationsPtr,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const unsigned int                       matrixFreeVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhs,
    const unsigned int matrixFreeQuadratureComponentAX,
    bool isComputeMeanValueConstraint)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);
    double time = MPI_Wtime();

    d_BLASWrapperPtr = BLASWrapperPtr;
    d_basisOperationsPtr        = basisOperationsPtr;
    d_matrixFreeDataPtr         = &(basisOperationsPtr->matrixFreeData());
    d_constraintMatrixPtr       = &constraintMatrix;
    d_matrixFreeVectorComponent = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponentRhs =
      matrixFreeQuadratureComponentRhs;

    d_locallyOwnedSize = d_basisOperationsPtr->nOwnedDofs();
    d_numberDofsPerElement = d_basisOperationsPtr->nDofsPerCell();
    d_numCells       = d_basisOperationsPtr->nCells();

    if (isComputeMeanValueConstraint)
      {
        computeMeanValueConstraint();
        d_isMeanValueConstraintComputed = true;
      }

    if (isComputeDiagonalA)
      {
        computeDiagonalA();
        d_isComputeDiagonalA = true;
      }

    d_basisOperationsPtr->computeStiffnessVector();
    d_cellStiffnessMatrix =
    d_basisOperationsPtr->cellStiffnessMatrixBasisData();


    d_constraintsInfo.initialize(
      d_matrixFreeDataPtr->get_vector_partitioner(
        matrixFreeVectorComponent),
      constraintMatrix);

    d_inc              = 1;
    d_negScalarCoeffAlpha = -1.0 / (4.0 * M_PI);
    d_scalarCoeffAlpha = 1.0 / (4.0 * M_PI);
    d_beta             = 0.0;
    d_alpha            = 1.0;
    d_transA           = 'N';
    d_transB           = 'N';
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::distributeX()
  {
    d_BLASWrapperPtr->axpby(d_locallyOwnedSize * d_blockSize,
                            d_alpha,
                            d_NDBCVec->data(),
                            d_alpha,
                            d_blockedXPtr->data());

    d_constraintsInfo.distribute(*d_xPtr);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::computeMeanValueConstraint()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::computeDiagonalA()
  {
    d_diagonalA = d_basisOperationsPtr->inverseStiffnessVectorBasisData();
    d_diagonalSqrtA = d_basisOperationsPtr->inverseSqrtStiffnessVectorBasisData();

    std::string errMsg = "Error in size of diagonal matrix.";
    throwException(d_diagonalA.size() != d_locallyOwnedSize, errMsg);
    throwException(d_diagonalSqrtA.size() != d_locallyOwnedSize, errMsg);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::vmult
    (dftfe::linearAlgebra::MultiVector<T,
                                          memorySpace> &Ax,
        dftfe::linearAlgebra::MultiVector<T,
                                          memorySpace> &x,
        unsigned int               blockSize)
  {
    Ax.setValue(0.0);
    d_AxCellLLevelNodalData.setValue(0.0);
    d_constraintsInfo.distribute(x, d_blockSize);

    d_basisOperationsPtr->extractToCellNodalData(x,
                                                 d_xCellLLevelNodalData.data());

    for(size_type iCell = 0; iCell < d_numCells ; iCell += d_cellsBlockSizeVmult)
      {
        std::pair<unsigned int, unsigned int> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeVmult, d_numCells));

        d_BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          d_blockSize,
          d_numberDofsPerElement,
          d_numberDofsPerElement,
          &d_scalarCoeffAlpha,
          d_xCellLLevelNodalData.data() +
            cellRange.first * d_numberDofsPerElement * d_blockSize,
          d_blockSize,
          d_numberDofsPerElement * d_blockSize,
          d_cellStiffnessMatrix.data() +
            cellRange.first * d_numberDofsPerElement * d_numberDofsPerElement,
          d_numberDofsPerElement,
          d_numberDofsPerElement * d_numberDofsPerElement,
          &d_beta,
          d_AxCellLLevelNodalData.data(),
          d_blockSize,
          d_numberDofsPerElement * d_blockSize,
          cellRange.second - cellRange.first);
      }

    d_basisOperationsPtr->accumulateFromCellNodalData(
      d_AxCellLLevelNodalData.data(),
      Ax);
    d_constraintsInfo.distribute_slave_to_master(Ax, d_blockSize);
    Ax.accumulateAddLocallyOwned();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::precondition_JacobiSqrt(dftfe::linearAlgebra::MultiVector<T,
                                                                                                                 memorySpace> &      dst,
                                                                              const dftfe::linearAlgebra::MultiVector<T,
                                                                                                                      memorySpace> &src,
                                                                              const double omega) const
  {
    double scaleValue = (4.0 * M_PI);
    scaleValue = std::sqrt(scaleValue);
    d_basisOperationsPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      scaleValue,
      d_diagonalSqrtA.data(),
      src.data(),
      dst.data(),
      d_mapNodeIdToProcId.data());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::precondition_Jacobi(ddftfe::linearAlgebra::MultiVector<T,
                                                                                                                 memorySpace> &      dst,
                                                                              const dftfe::linearAlgebra::MultiVector<T,
                                                                                                                      memorySpace> &src,
                                                                              const double omega) const
  {
    double scaleValue = (4.0 * M_PI);
    d_basisOperationsPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      scaleValue,
      d_onesVec.data(),
      src.data(),
      dst.data(),
      d_mapNodeIdToProcId.data());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  MultiVectorPoissonLinearSolverProblem<memorySpace>::setDataForRhsVec(dftfe::utils::MemoryStorage<double, memorySpace>& inputQuadData)
  {
    d_rhsQuadDataPtr = &inputQuadData;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  dftfe::linearAlgebra::MultiVector<T,
                                    memorySpace> &
  MultiVectorPoissonLinearSolverProblem<memorySpace>::computeRhs(
             dftfe::linearAlgebra::MultiVector<T,
                                               memorySpace> &       NDBCVec,
             dftfe::linearAlgebra::MultiVector<T,
                                               memorySpace> &       outputVec,
             unsigned int                      blockSizeInput)
  {
    d_cellsBlockSizeVmult = 100;
    d_basisOperationsPtr->reinit(blockSizeInput,
                                 d_cellsBlockSizeVmult,
                                 d_matrixFreeQuadratureComponentRhs,
                                 true, // TODO should this be set to true
                                 true); // TODO should this be set to true
    if(d_blockSize != blockSizeInput)
      {
        d_blockSize = blockSizeInput;
        dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::HOST>
          nodeIds;
        nodeIds.resize(d_locallyOwnedSize);
        for(size_type i = 0 ; i < d_locallyOwnedSize;i++)
          {
            nodeIds.data()[i] = i*d_blockSize;
          }
        d_mapNodeIdToProcId.resize(d_locallyOwnedSize);
        d_mapNodeIdToProcId.copy_from(nodeIds);

        d_xCellLLevelNodalData.resize(d_numCells*d_numberDofsPerElement*d_blockSize);
        d_AxCellLLevelNodalData.resize(d_numCells*d_numberDofsPerElement*d_blockSize);

        d_cellsBlockSizeVmult = 100;  // set this correctly
        d_basisOperationsPtr->reinit(d_blockSize,
                                     d_cellsBlockSizeVmult,
                                     d_matrixFreeQuadratureComponentRhs,
                                     true, // TODO should this be set to true
                                     true); // TODO should this be set to true

        d_basisOperationsPtr->createMultiVector(d_blockSize,d_rhsVec);

      }

    d_blockedXPtr = &outputVec;
    d_blockedNDBCPtr = &NDBCVec;



    dftfe::utils::MemoryStorage<double, memorySpace>
      xCellLLevelNodalData, rhsCellLLevelNodalData;

    xCellLLevelNodalData.resize(d_numCells*d_numberDofsPerElement*d_blockSize);
    rhsCellLLevelNodalData.resize(d_numCells*d_numberDofsPerElement*d_blockSize);
    //     Adding the Non homogeneous Dirichlet boundary conditions
    d_rhsVec.setValue(0.0);

    // Calculating the rhs from the quad points
    // multiVectorInput is stored on the quad points

    //Assumes that NDBC is constraints distribute is called
    // rhs  = - ( 1.0 / 4 \pi ) \int \nabla N_j \nabla N_i  d_NDBC

    d_basisOperationsPtr->extractToCellNodalData(*d_blockedNDBCPtr,
                                                 xCellLLevelNodalData.data());

    for(size_type iCell = 0; iCell < d_numCells ; iCell += d_cellsBlockSizeVmult)
      {
        std::pair<unsigned int, unsigned int> cellRange(
          iCell, std::min(iCell + d_cellsBlockSizeVmult, d_numCells));

        d_BLASWrapperPtr->xgemmStridedBatched(
          'N',
          'N',
          d_blockSize,
          d_numberDofsPerElement,
          d_numberDofsPerElement,
          &d_negScalarCoeffAlpha,
          xCellLLevelNodalData.data() +
            cellRange.first * d_numberDofsPerElement * d_blockSize,
          d_blockSize,
          d_numberDofsPerElement * d_blockSize,
          d_cellStiffnessMatrix.data() +
            cellRange.first * d_numberDofsPerElement * d_numberDofsPerElement,
          d_numberDofsPerElement,
          d_numberDofsPerElement * d_numberDofsPerElement,
          &d_beta,
          rhsCellLLevelNodalData.data(),
          d_blockSize,
          d_numberDofsPerElement * d_blockSize,
          cellRange.second - cellRange.first);
      }

    d_basisOperationsPtr->accumulateFromCellNodalData(
      rhsCellLLevelNodalData.data(),
      d_rhsVec);

    std::pair<unsigned int, unsigned int> cellRange = std::make_pair(0,d_numCells);
    d_basisOperationsPtr->integrateWithBasisKernel(d_rhsQuadDataPtr,NULL,d_rhsVec,cellRange);
    d_constraintsInfo.distribute_slave_to_master(*d_rhsQuadDataPtr, d_blockSize);
    d_rhsVec.accumulateAddLocallyOwned();

    return d_rhsVec;
  }

}
