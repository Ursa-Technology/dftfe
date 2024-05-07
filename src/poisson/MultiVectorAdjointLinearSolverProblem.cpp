//
// Created by VISHAL SUBRAMANIAN on 4/30/24.
//

#include "include/MultiVectorAdjointLinearSolverProblem.h"

namespace dftfe
{


  // constructor
  template <dftfe::utils::MemorySpace memorySpace>
  MultiVectorAdjointLinearSolverProblem<memorySpace>::MultiVectorAdjointLinearSolverProblem(
    const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain)
    : mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_isComputeShapeFunction           = true;
    d_isComputeDiagonalA               = true;
    d_constraintMatrixPtr              = NULL;
    d_blockedXPtr                      = NULL;
    d_matrixFreeQuadratureComponentRhs = -1;
    d_matrixFreeVectorComponent        = -1;
    d_blockSize                        = 0;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::reinit(
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
    const bool              isComputeDiagonalA,
    const bool              isComputeShapeFunction)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);

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

    d_ksOperatorPtr = &ksHamiltonianObj;

    if (isComputeDiagonalA)
      {
        computeDiagonalA();
        d_isComputeDiagonalA = true;
      }

    d_constraintsInfo.initialize(
      d_matrixFreeDataPtr->get_vector_partitioner(
        matrixFreeVectorComponent),
      constraintMatrix);

    d_onesDevice.resize(d_locallyOwnedSize);
    d_onesDevice.setValue(1.0);

  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
    MultiVectorAdjointLinearSolverProblem<memorySpace>::computeDiagonalA()
  {
    d_basisOperationsPtr->computeStiffnessVector();
    d_basisOperationsPtr->computeInverseSqrtMassVector();

    d_diagonalA.setValue(1.0);

    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::HOST>
      nodeIds;
    nodeIds.resize(d_locallyOwnedSize);
    for(size_type i = 0 ; i < d_locallyOwnedSize;i++)
      {
        nodeIds.data()[i] = i;
      }

    dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> mapNodeIdToProcId;
    mapNodeIdToProcId.resize(d_locallyOwnedSize);
    mapNodeIdToProcId.copy_from(nodeIds);

    auto sqrtMassMat = BLASWrapperPtr->sqrtMassVectorBasisData();
    auto inverseStiffVec = d_basisOperationsPtr->inverseStiffnessVectorBasisData();
    auto inverseSqrtStiffVec = d_basisOperationsPtr->inverseSqrtStiffnessVectorBasisData();

    d_basisOperationsPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      1.0/0.5,
      inverseStiffVec.data(),
      d_diagonalA.data(),
      d_diagonalA.data(),
      mapNodeIdToProcId.data());

    d_basisOperationsPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      1.0,
      sqrtMassMat.data(),
      d_diagonalA.data(),
      d_diagonalA.data(),
      mapNodeIdToProcId.data());

    d_basisOperationsPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      1.0,
      sqrtMassMat.data(),
      d_diagonalA.data(),
      d_diagonalA.data(),
      mapNodeIdToProcId.data());

    d_diagonalSqrtA.setValue(1.0);
    d_basisOperationsPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      std::sqrt(1.0/0.5),
      inverseStiffVec.data(),
      d_diagonalSqrtA.data(),
      d_diagonalSqrtA.data(),
      mapNodeIdToProcId.data());

    d_basisOperationsPtr->stridedBlockScaleCopy(
      1,
      d_locallyOwnedSize,
      1.0,
      sqrtMassMat.data(),
      d_diagonalSqrtA.data(),
      d_diagonalSqrtA.data(),
      mapNodeIdToProcId.data());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
    MultiVectorAdjointLinearSolverProblem<memorySpace>::precondition_Jacobi(
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> &      dst,
    const dftfe::linearAlgebra::MultiVector<T,
                                            memorySpace> &src,
    const double                     omega)
  {
    d_basisOperationsPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      1.0,
      d_diagonalA.data(),
      src.data(),
      dst.data(),
      d_mapNodeIdToProcId.data());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::
    precondition_JacobiSqrt(ddftfe::linearAlgebra::MultiVector<T,
                                                             memorySpace> &      dst,
                          const dftfe::linearAlgebra::MultiVector<T,
                                                                  memorySpace> &src,
                          const double omega) const
  {
    d_basisOperationsPtr->stridedBlockScaleCopy(
      d_blockSize,
      d_locallyOwnedSize,
      1.0,
      d_diagonalSqrtA.data(),
      src.data(),
      dst.data(),
      d_mapNodeIdToProcId.data());
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  dftfe::linearAlgebra::MultiVector<T,
                                    memorySpace> &
    MultiVectorAdjointLinearSolverProblem<memorySpace>::computeRhs(
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> &       NDBCVec,
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> &       outputVec,
    unsigned int                      blockSizeInput)
  {
    if(d_blockSize != blockSizeInput)
      {
        d_blockSize = blockSizeInput;
        dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                    dftfe::utils::MemorySpace::HOST>
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

    d_blockedXDevicePtr = &outputVecDevice;

    // psiTemp = M^{1/2} psi
    // TODO check if this works
    //    computing_timer.leave_subsection("Rhs init  MPI");
    //
    computing_timer.enter_subsection("Rhs init Device MPI");
    distributedDeviceVec<double> psiTempDevice;
    psiTempDevice.reinit(*d_psiDevice);
    dftfe::utils::deviceMemcpyD2D(
      dftfe::utils::makeDataTypeDeviceCompatible(psiTempDevice.data()),
      d_psiDevice->data(),
      d_locallyOwnedDofs * blockSizeInput * sizeof(double));
    psiTempDevice.updateGhostValues();

    distributedDeviceVec<double> psiTempDevice2;
    psiTempDevice2.reinit(*d_psiDevice);

    psiTempDevice2.setValue(0.0);

    const unsigned int totalLocallyOwnedCells =
      d_matrixFreeDataPtr->n_physical_cells();

    const dealii::Quadrature<3> &quadratureRhs =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentRhs);
    const unsigned int numberDofsPerElement =
      d_dofHandler->get_fe().dofs_per_cell;
    const unsigned numberQuadraturePointsRhs = quadratureRhs.size();


    std::vector<double> inputJxW(numberQuadraturePointsRhs *
                                   totalLocallyOwnedCells,
                                 0.0);
    d_inputJxWDevice.resize(numberQuadraturePointsRhs * totalLocallyOwnedCells,
                            0.0);
    for (unsigned int elemId = 0; elemId < totalLocallyOwnedCells; elemId++)
      {
        unsigned int quadStartId = elemId * numberQuadraturePointsRhs;
        for (unsigned int iQuad = 0; iQuad < numberQuadraturePointsRhs; iQuad++)
          {
            inputJxW[quadStartId + iQuad] =
              d_cellJxW[quadStartId + iQuad] * multiVectorInput[elemId][iQuad];
          }
      }

    dftfe::utils::deviceMemcpyH2D(d_inputJxWDevice.begin(),
                                  &inputJxW[0],
                                  numberQuadraturePointsRhs *
                                    totalLocallyOwnedCells * sizeof(double));

    computing_timer.leave_subsection("Rhs init Device MPI");

    computing_timer.enter_subsection("M^(-1/2) Device MPI");
    psiTempDevice.updateGhostValues();
    d_constraintsMatrixInfoDevice.distribute(psiTempDevice, d_blockSize);

    dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
      d_blockSize,
      d_locallyOwnedDofs,
      1.0,
      d_sqrtMassDeviceVec,
      psiTempDevice.begin());
    psiTempDevice.updateGhostValues();
    computing_timer.leave_subsection("M^(-1/2) Device MPI");

    computing_timer.enter_subsection("computeR Device MPI");
    computeRMatrixDevice(d_inputJxWDevice);
    computing_timer.leave_subsection("computeR Device MPI");
    computing_timer.enter_subsection("computeMu Device MPI");
    computeMuMatrixDevice(d_inputJxWDevice, *d_psiDevice);



    computing_timer.leave_subsection("computeMu Device MPI");


    computing_timer.enter_subsection("Mu*Psi Device MPI");
    //    rhs = 0.0;
    rhsDevice.setValue(0.0);
    // Calculating the rhs from the quad points
    // multiVectorInput is stored on the quad points
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell             = d_dofHandler->begin_active(),
      endc             = d_dofHandler->end();
    unsigned int iElem = 0;

    const unsigned int inc  = 1;
    const double       beta = 0.0, alpha = 1.0, alpha_minus_two = -2.0,
                 alpha_minus_one = -1.0;
    char transposeMat            = 'T';
    char doNotTransposeMat       = 'N';

    // rhs = Psi*Mu. Since blas/lapack assume a column-major format whereas the
    // Psi is stored in a row major format, we do Mu^T*\Psi^T = Mu*\Psi^T
    // (because Mu is symmetric)

    dftfe::utils::deviceBlasWrapper::gemm(
      d_kohnShamDeviceClassPtr->getDeviceBlasHandle(),
      dftfe::utils::DEVICEBLAS_OP_N,
      dftfe::utils::DEVICEBLAS_OP_N,
      d_blockSize,
      d_locallyOwnedDofs,
      d_blockSize,
      &alpha_minus_two,
      d_MuMatrixDevice.begin(),
      d_blockSize,
      psiTempDevice.begin(),
      d_blockSize,
      &beta,
      rhsDevice.begin(),
      d_blockSize);

    computing_timer.leave_subsection("Mu*Psi Device MPI");


    //
    // y = M^{-1/2} * R * M^{-1/2} * PsiTemp
    // 1. Do PsiTemp = M^{-1/2}*PsiTemp
    // 2. Do PsiTemp2 = R*PsiTemp
    // 3. PsiTemp2 = M^{-1/2}*PsiTemp2
    //

    computing_timer.enter_subsection("psi*M(-1/2) Device MPI");
    dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
      d_blockSize,
      d_locallyOwnedDofs,
      1.0,
      d_invSqrtMassDeviceVec,
      psiTempDevice.begin());
    psiTempDevice.updateGhostValues();
    d_constraintsMatrixInfoDevice.distribute(psiTempDevice, d_blockSize);
    computing_timer.leave_subsection("psi*M(-1/2) Device MPI");

    computing_timer.enter_subsection("R times psi Device MPI");

    // 2. Do PsiTemp2 = R*PsiTemp
    d_cellWaveFunctionMatrixDevice.setValue(0.0);
    dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
      d_blockSize,
      totalLocallyOwnedCells * d_numberDofsPerElement,
      psiTempDevice.data(),
      d_cellWaveFunctionMatrixDevice.begin(),
      d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());

    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);
    const unsigned int strideA = d_numberDofsPerElement * d_blockSize;
    const unsigned int strideB =
      d_numberDofsPerElement * d_numberDofsPerElement;
    const unsigned int strideC = d_numberDofsPerElement * d_blockSize;


    dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
      d_kohnShamDeviceClassPtr->getDeviceBlasHandle(),
      dftfe::utils::DEVICEBLAS_OP_N,
      std::is_same<dataTypes::number, std::complex<double>>::value ?
        dftfe::utils::DEVICEBLAS_OP_T :
        dftfe::utils::DEVICEBLAS_OP_N,
      d_blockSize,
      d_numberDofsPerElement,
      d_numberDofsPerElement,
      &scalarCoeffAlpha,
      d_cellWaveFunctionMatrixDevice.begin(),
      d_blockSize,
      strideA,
      d_RMatrixDevice.begin(),
      d_numberDofsPerElement,
      strideB,
      &scalarCoeffBeta,
      d_cellRMatrixTimesWaveMatrixDevice.begin(),
      d_blockSize,
      strideC,
      totalLocallyOwnedCells);

    dftfe::utils::deviceKernelsGeneric::axpyStridedBlockAtomicAdd(
      d_blockSize,
      totalLocallyOwnedCells * d_numberDofsPerElement,
      d_cellRMatrixTimesWaveMatrixDevice.begin(),
      psiTempDevice2.data(),
      d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());
    d_constraintsMatrixInfoDevice.distribute_slave_to_master(psiTempDevice2,
                                                             d_blockSize);
    psiTempDevice2.accumulateAddLocallyOwned();

    // 3. PsiTemp2 = M^{-1/2}*PsiTemp2
    computing_timer.leave_subsection("R times psi Device MPI");

    computing_timer.enter_subsection("psiTemp M^(-1/2) Device MPI");


    psiTempDevice2.updateGhostValues();
    d_constraintsMatrixInfoDevice.distribute(psiTempDevice2, d_blockSize);

    dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
      d_blockSize,
      d_locallyOwnedDofs,
      1.0,
      d_invSqrtMassDeviceVec,
      psiTempDevice2.begin());
    psiTempDevice2.updateGhostValues();
    d_constraintsMatrixInfoDevice.distribute(psiTempDevice2, d_blockSize);

    stridedBlockScaleForEigen(d_blockSize,
                              d_locallyOwnedDofs,
                              4.0,
                              d_effectiveOrbitalOccupancyDevice.data(),
                              psiTempDevice2.data(),
                              rhsDevice.data());

    d_constraintsMatrixInfoDevice.set_zero(rhsDevice, d_blockSize);

    computing_timer.leave_subsection("psiTemp M^(-1/2) Device MPI");

  }


  // TODO PLease call d_kohnShamDeviceClassPtr->reinitkPointSpinIndex() before
  // calling this functions.
  // template<typename T>

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::updateInputPsi(
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace>
      &psiInputVec, // assumes the vector as been distributed with correct
                                                     // constraints
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> &psiInputVecDevice, // need to call distribute
    std::vector<double>
                                           &effectiveOrbitalOccupancy, // incorporates spin information
    std::vector<std::vector<unsigned int>> &degeneracy,
    std::vector<double> &                   eigenValues,
    unsigned int                            blockSize)
  {
    pcout << " updating psi inside adjoint\n";
    d_psi       = &psiInputVec;
    d_psiDevice = &psiInputVecDevice;
    d_RMatrix.resize(d_totalLocallyOwnedCells * d_numberDofsPerElement *
                       d_numberDofsPerElement,
                     0.0);
    d_RMatrixDevice.resize(d_totalLocallyOwnedCells * d_numberDofsPerElement *
                             d_numberDofsPerElement,
                           0.0);
    d_MuMatrix.resize(blockSize * blockSize, 0.0);
    d_MuMatrixDevice.resize(blockSize * blockSize, 0.0);

    std::fill(d_MuMatrix.begin(), d_MuMatrix.end(), 0.0);

    // TODO deep copy not necessary;
    d_effectiveOrbitalOccupancy = effectiveOrbitalOccupancy;
    d_effectiveOrbitalOccupancyDevice.resize(blockSize, 0.0);

    dftfe::utils::deviceMemcpyH2D(d_effectiveOrbitalOccupancyDevice.begin(),
                                  &d_effectiveOrbitalOccupancy[0],
                                  blockSize * sizeof(double));


    d_degenerateState = degeneracy;
    d_eigenValues     = eigenValues;

    d_vectorList.resize(0);
    for (unsigned int iVec = 0; iVec < blockSize; iVec++)
      {
        unsigned int totalNumDegenerateStates = d_degenerateState[iVec].size();
        for (unsigned int jVec = 0; jVec < totalNumDegenerateStates; jVec++)
          {
            d_vectorList.push_back(iVec);
            d_vectorList.push_back(d_degenerateState[iVec][jVec]);
          }
      }

    d_MuMatrixDeviceCellWise.resize((d_vectorList.size() / 2) *
                                      d_totalLocallyOwnedCells,
                                    0.0);
    d_MuMatrixCellWise.resize((d_vectorList.size() / 2) *
                                d_totalLocallyOwnedCells,
                              0.0);
    d_vectorListDevice.resize(d_vectorList.size());
    d_vectorListDevice.copyFrom(d_vectorList);
    if (blockSize != d_blockSize)
      {
        dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
          d_matrixFreeDataPtr->get_vector_partitioner(
            d_matrixFreeVectorComponent),
          blockSize,
          d_psiWithFarFieldBC);

        // If the number of vectors in the size is different, then the Map has
        // to be re-initialised. The d_blockSize is set to -1 in the
        // constructor, so that this if condition is satisfied the first time
        // the code is called.
        d_blockSize = blockSize;

        vectorTools::computeCellLocalIndexSetMap(
          psiInputVec.getMPIPatternP2P(),
          *d_matrixFreeDataPtr,
          d_matrixFreeVectorComponent,
          d_blockSize,
          d_flattenedArrayCellLocalProcIndexIdMap);

        d_flattenedArrayCellLocalProcIndexIdMapDevice.resize(
          d_flattenedArrayCellLocalProcIndexIdMap.size());
        d_flattenedArrayCellLocalProcIndexIdMapDevice.copyFrom(
          d_flattenedArrayCellLocalProcIndexIdMap);


        // Setting up the constraint matrix for distributing the solution vector
        // efficiently.
        // d_constraintMatrixPtr stores the homogeneous Dirichlet boundary
        // conditions, hanging nodes and periodic constraints
        d_constraintsMatrixInfoHost.initialize(
          d_matrixFreeDataPtr->get_vector_partitioner(
            d_matrixFreeVectorComponent),
          *d_constraintMatrixPtr);

        //        d_constraintsMatrixPsiDataInfo.precomputeMaps(
        //          d_matrixFreeDataPtr->get_vector_partitioner(
        //            d_matrixFreePsiVectorComponent),
        //          psiInputVec.getMPIPatternP2P(),
        //          d_blockSize);

        d_constraintsMatrixInfoHost.precomputeMaps(
          psiInputVec.getMPIPatternP2P(), d_blockSize);


        //        d_blockedDiagonalA.reinit(rhs);
        //        calculateBlockDiagonalVec();
        d_cellLevelX.resize(d_numberDofsPerElement * d_blockSize);
        d_cellLevelAX.resize(d_numberDofsPerElement * d_blockSize);

        //        dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        //           d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
        //           blockSize,
        //	   d_AxDevice);
        //
        //        dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        //           d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
        //           blockSize,
        //	   d_xDevice);

        d_constraintsMatrixInfoDevice.initialize(
          d_matrixFreeDataPtr->get_vector_partitioner(
            d_matrixFreeVectorComponent),
          *d_constraintMatrixPtr);

        d_constraintsMatrixInfoDevice.precomputeMaps(
          d_psiWithFarFieldBC.getMPIPatternP2P(), d_blockSize);

        d_cellWaveFunctionMatrixDevice.resize(
          d_totalLocallyOwnedCells * d_numberDofsPerElement * d_blockSize, 0.0);

        d_cellRMatrixTimesWaveMatrixDevice.resize(
          d_totalLocallyOwnedCells * d_numberDofsPerElement * d_blockSize, 0.0);

        d_cellWaveFunctionQuadMatrixDevice.resize(
          d_totalLocallyOwnedCells * d_numberQuadraturePointsRhs * d_blockSize,
          0.0);
        d_cellvec2QuadMatrixDevice.resize(d_totalLocallyOwnedCells *
                                            d_numberQuadraturePointsRhs *
                                            d_blockSize,
                                          0.0);
      }
    for (unsigned int iNode = 0; iNode < d_locallyOwnedDofs * d_blockSize;
         iNode++)
      {
        d_psiWithFarFieldBC.data()[iNode] = psiInputVec.data()[iNode];
      }
    d_constraintsMatrixInfoHost.distribute(d_psiWithFarFieldBC, d_blockSize);

    d_psiDevice->updateGhostValues();
    d_constraintsMatrixInfoDevice.distribute(*d_psiDevice, d_blockSize);

    // calling constraints distribute on the input vector
    // TODO change this
    d_constraintsMatrixInfoHost.distribute(*d_psi, d_blockSize);
    d_projectorKetTimesVector =
      &(d_kohnShamDeviceClassPtr
          ->getParallelProjectorKetTimesBlockVectorDevice());

    d_scalarProdVecDevice.resize(d_blockSize * d_locallyOwnedDofs, 0.0);
    d_scalarProdVecQuadDevice.resize(d_blockSize * d_numberQuadraturePointsRhs *
                                       d_totalLocallyOwnedCells,
                                     0.0);
    d_eigenValuesMemSpace.resize(d_blockSize, 0.0);

    for(signed int iBlock = 0 ; iBlock < d_blockSize; iBlock++)
      {
        eigenValues[iBlock] = -1.0*eigenValues[iBlock];
      }
    dftfe::utils::deviceMemcpyH2D(d_eigenValuesDevice.begin(),
                                  &eigenValues[0],
                                  d_blockSize * sizeof(double));

    d_dotProdDevice.resize(d_blockSize, 0.0);
  }

  void
  MultiVectorAdjointProblemDevice::computeMuMatrixDevice(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
                                 &                           inputJxwDevice,
    distributedDeviceVec<double> &psiVecDevice)
  {
    const unsigned int totalLocallyOwnedCells =
      d_matrixFreeDataPtr->n_physical_cells();

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandler->begin_active(),
      endc = d_dofHandler->end();

    const dealii::Quadrature<3> &quadratureRhs =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentRhs);
    const unsigned int numberDofsPerElement =
      d_dofHandler->get_fe().dofs_per_cell;
    const unsigned numberQuadraturePointsRhs = quadratureRhs.size();


    const unsigned int inc  = 1;
    const double       beta = 0.0, alpha = 1.0;
    char               transposeMat      = 'T';
    char               doNotTransposeMat = 'N';

    d_MuMatrixDevice.setValue(0.0);
    std::fill(d_MuMatrix.begin(), d_MuMatrix.end(), 0.0);

    d_cellWaveFunctionMatrixDevice.setValue(0.0);

    dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
      d_blockSize,
      totalLocallyOwnedCells * d_numberDofsPerElement,
      psiVecDevice.data(),
      d_cellWaveFunctionMatrixDevice.begin(),
      d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());


    const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                            scalarCoeffBeta  = dataTypes::number(0.0);
    const unsigned int strideA = d_numberDofsPerElement * d_blockSize;
    const unsigned int strideB = 0;
    const unsigned int strideC = numberQuadraturePointsRhs * d_blockSize;

    dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
      d_kohnShamDeviceClassPtr->getDeviceBlasHandle(),
      dftfe::utils::DEVICEBLAS_OP_N,
      std::is_same<dataTypes::number, std::complex<double>>::value ?
        dftfe::utils::DEVICEBLAS_OP_T :
        dftfe::utils::DEVICEBLAS_OP_N,
      d_blockSize,
      numberQuadraturePointsRhs,
      numberDofsPerElement,
      &alpha,
      d_cellWaveFunctionMatrixDevice.begin(),
      d_blockSize,
      strideA,
      d_shapeFunctionValueDevice.begin(),
      numberDofsPerElement,
      strideB,
      &beta,
      d_cellWaveFunctionQuadMatrixDevice.begin(),
      d_blockSize,
      strideC,
      totalLocallyOwnedCells);

    unsigned int numVec = d_vectorList.size() / 2;
    d_MuMatrixDeviceCellWise.setValue(0.0);

    muMatrixDevice(totalLocallyOwnedCells,
                   numVec,
                   numberQuadraturePointsRhs,
                   d_blockSize,
                   d_effectiveOrbitalOccupancyDevice.begin(),
                   d_vectorListDevice.begin(),
                   d_cellWaveFunctionQuadMatrixDevice.begin(),
                   inputJxwDevice.begin(),
                   d_MuMatrixDeviceCellWise.data());


    dftfe::utils::deviceMemcpyD2H(
      dftfe::utils::makeDataTypeDeviceCompatible(d_MuMatrixCellWise.data()),
      d_MuMatrixDeviceCellWise.data(),
      numVec * totalLocallyOwnedCells * sizeof(dataTypes::number));


    for (unsigned int iVecList = 0; iVecList < numVec; iVecList++)
      {
        unsigned int iVec            = d_vectorList[2 * iVecList];
        unsigned int degenerateVecId = d_vectorList[2 * iVecList + 1];
        for (unsigned int iCell = 0; iCell < totalLocallyOwnedCells; iCell++)
          {
            d_MuMatrix[iVec * d_blockSize + degenerateVecId] +=
              d_MuMatrixCellWise[iVecList + iCell * numVec];
          }
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  &d_MuMatrix[0],
                  d_blockSize * d_blockSize,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);

    dftfe::utils::deviceMemcpyH2D(d_MuMatrixDevice.begin(),
                                  &d_MuMatrix[0],
                                  d_blockSize * d_blockSize * sizeof(double));
  }

  // template<typename T>
  void
  MultiVectorAdjointProblemDevice::computeRMatrixDevice(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &inputJxwDevice)
  {
    const unsigned int totalLocallyOwnedCells =
      d_matrixFreeDataPtr->n_physical_cells();

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandler->begin_active(),
      endc = d_dofHandler->end();

    const dealii::Quadrature<3> &quadratureRhs =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentRhs);
    const unsigned numberQuadraturePointsRhs = quadratureRhs.size();


    rMatrixDevice(totalLocallyOwnedCells,
                  d_numberDofsPerElement,
                  numberQuadraturePointsRhs,
                  d_shapeFunctionValueTransposedDevice.begin(),
                  inputJxwDevice.begin(),
                  d_RMatrixDevice.begin());
  }

  void
  MultiVectorAdjointProblemDevice::distributeX()
  {

    std::vector<double> dotProductHost(blockSize, 0.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      dotProductMemSpace(blockSize, 0.0);
    BLASWrapperPtr->MultiVectorXDot(d_blockSize,
                                    d_onesDevice,
                                    d_blockedXPtr.begin(),
                                    yMemSpace.begin(),
                                    d_onesDevice.begin(),
                                    tempVec.begin(),
                                    dotProductMemSpace.begin(),
                                    mpi_communicator,
                                    dotProductHost.begin());


    std::vector<double> dotProductsFromDevice;
    // distributeX for the GPU vector
    dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
      d_blockSize,
      d_locallyOwnedDofs,
      1.0,
      d_invSqrtMassDeviceVec,
      d_blockedXDevicePtr->begin());

    d_blockedXDevicePtr->updateGhostValues();
    d_constraintsMatrixInfoDevice.distribute(*d_blockedXDevicePtr, d_blockSize);

    computeDotProductQuadDevice(*d_blockedXDevicePtr,
                                *d_psiDevice,
                                dotProductsFromDevice);

    d_dotProdFinalDevice.resize(d_blockSize, 0.0);

    dftfe::utils::deviceMemcpyH2D(d_dotProdFinalDevice.data(),
                                  &dotProductsFromDevice[0],
                                  d_blockSize * sizeof(dataTypes::number));

    stridedBlockScaleForEigen(d_blockSize,
                              d_locallyOwnedDofs,
                              -1.0,
                              d_dotProdFinalDevice.data(),
                              d_psiDevice->data(),
                              d_blockedXDevicePtr->data());

    d_blockedXDevicePtr->updateGhostValues();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template<typename T>
  void
  MultiVectorAdjointLinearSolverProblem<memorySpace>::vmult(dftfe::linearAlgebra::MultiVector<T,
                                                                           memorySpace> &Ax,
                                                            dftfe::linearAlgebra::MultiVector<T,
                                                                                              memorySpace> &x,
                                         unsigned int blockSize)
  {
    Ax.setValue(0.0);
    d_ksOperatorPtr->HX(x,
                        scalarHX,
                        scalarY,
                        scalarX
                        Ax,
                        false); // onlyHPrimePartForFirstOrderDensityMatResponse

    d_basisOperationsPtr->set_zero(x, d_blockSize);
    d_basisOperationsPtr->set_zero(Ax, d_blockSize);

    d_basisOperationsPtr->stridedBlockScaleAndAddColumnWise(
                               d_blockSize,
                               d_locallyOwnd,
      x.data(),
      d_eigenValuesMemSpace.data(),
      Ax.data());
  }


}
