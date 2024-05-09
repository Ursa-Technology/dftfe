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
// @author Vishal Subramanian, Bikash Kanungo
//

#include "inverseDFTSolverFunction.h"
#include <densityCalculatorCPU.h>
#include <densityCalculatorDevice.h>
#include <map>
#include <vector>
#include "dftUtils.h"
#include "NodalData.h"
#include "CompositeData.h"
#include "MPIWriteOnFile.h"

namespace dftfe
{
  namespace
  {
    //    void
    //    pointWiseScaleWithDiagonal(
    //      const distributedCPUVec<double> &diagonal,
    //      const unsigned int              numberFields,
    //      std::vector<dataTypes::number> &fieldsArrayFlattened)
    //    {
    //      const unsigned int numberDofs = fieldsArrayFlattened.size() /
    //      numberFields; const unsigned int inc        = 1;
    //
    //      for (unsigned int i = 0; i < numberDofs; ++i)
    //        {
    //#ifdef USE_COMPLEX
    //          double scalingCoeff = diagonal.local_element(i);
    //          zdscal_(&numberFields,
    //                  &scalingCoeff,
    //                  &fieldsArrayFlattened[i * numberFields],
    //                  &inc);
    //#else
    //          double scalingCoeff = diagonal.local_element(i);
    //          dscal_(&numberFields,
    //                 &scalingCoeff,
    //                 &fieldsArrayFlattened[i * numberFields],
    //                 &inc);
    //#endif
    //        }
    //    }


    void
    evaluateDegeneracyMap(const std::vector<double> &             eigenValues,
                          std::vector<std::vector<unsigned int>> &degeneracyMap,
                          const double                            degeneracyTol)
    {
      const unsigned int N = eigenValues.size();
      degeneracyMap.resize(N, std::vector<unsigned int>(0));
      std::map<unsigned int, std::set<unsigned int>> groupIdToEigenId;
      std::map<unsigned int, unsigned int>           eigenIdToGroupId;
      unsigned int                                   groupIdCount = 0;
      for (unsigned int i = 0; i < N; ++i)
        {
          auto it = eigenIdToGroupId.find(i);
          if (it != eigenIdToGroupId.end())
            {
              const unsigned int groupId = it->second;
              for (unsigned int j = 0; j < N; ++j)
                {
                  if (std::abs(eigenValues[i] - eigenValues[j]) < degeneracyTol)
                    {
                      groupIdToEigenId[groupId].insert(j);
                      eigenIdToGroupId[j] = groupId;
                    }
                }
            }
          else
            {
              groupIdToEigenId[groupIdCount].insert(i);
              eigenIdToGroupId[i] = groupIdCount;
              for (unsigned int j = 0; j < N; ++j)
                {
                  if (std::abs(eigenValues[i] - eigenValues[j]) < degeneracyTol)
                    {
                      groupIdToEigenId[groupIdCount].insert(j);
                      eigenIdToGroupId[j] = groupIdCount;
                    }
                }

              groupIdCount++;
            }
        }

      for (unsigned int i = 0; i < N; ++i)
        {
          const unsigned int     groupId       = eigenIdToGroupId[i];
          std::set<unsigned int> degenerateIds = groupIdToEigenId[groupId];
          degeneracyMap[i].resize(degenerateIds.size());
          std::copy(degenerateIds.begin(),
                    degenerateIds.end(),
                    degeneracyMap[i].begin());
        }
    }

  } // namespace

  template <typename T>
  inverseDFTSolverFunction<T>::inverseDFTSolverFunction(
    const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain,
    const MPI_Comm &mpi_comm_interpool,
    const MPI_Comm &mpi_comm_interband)
    : d_mpi_comm_parent(mpi_comm_parent)
    , d_mpi_comm_domain(mpi_comm_domain)
    , d_mpi_comm_interpool(mpi_comm_interpool)
    , d_mpi_comm_interband(mpi_comm_interband)
    , d_multiVectorAdjointProblem(mpi_comm_parent, mpi_comm_domain)
    , d_multiVectorLinearMINRESSolver(mpi_comm_parent, mpi_comm_domain)
    ,
#ifdef DFTFE_WITH_DEVICE
    d_multiVectorAdjointProblemDevice(mpi_comm_parent, mpi_comm_domain)
    , d_multiVectorLinearMINRESSolverDevice(mpi_comm_parent, mpi_comm_domain)
    ,
#endif
    pcout(std::cout,
          (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0))
  {
    d_resizeGPUVecDuringInterpolation = true;
    d_resizeCPUVecDuringInterpolation = true;
    d_lossPreviousIteration = 0.0;
  }

  template <typename T>
  void
  inverseDFTSolverFunction<T>::preComputeChildShapeFunction()
  {
    // Quadrature for AX multiplication will FEOrderElectro+1
    const dealii::Quadrature<3> &quadratureRuleChild =
      d_matrixFreeDataChild->get_quadrature(d_matrixFreeQuadratureComponentPot);
    const unsigned int numQuadraturePointsPerCellChild =
      quadratureRuleChild.size();
    const unsigned int numTotalQuadraturePointsChild =
      d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild;

    dealii::FEValues<3> fe_valuesChild(d_dofHandlerChild->get_fe(),
                                       quadratureRuleChild,
                                       dealii::update_values |
                                         dealii::update_JxW_values);

    const unsigned int numberDofsPerElement =
      d_dofHandlerChild->get_fe().dofs_per_cell;

    //
    // resize data members
    //

    d_childCellJxW.resize(numTotalQuadraturePointsChild);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell             = d_dofHandlerChild->begin_active(),
      endc             = d_dofHandlerChild->end();
    unsigned int iElem = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_valuesChild.reinit(cell);
          if (iElem == 0)
            {
              // For the reference cell initalize the shape function values
              d_childCellShapeFunctionValue.resize(
                numberDofsPerElement * numQuadraturePointsPerCellChild);

              for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                   ++iNode)
                {
                  for (unsigned int q_point = 0;
                       q_point < numQuadraturePointsPerCellChild;
                       ++q_point)
                    {
                      d_childCellShapeFunctionValue
                        [numQuadraturePointsPerCellChild * iNode + q_point] =
                          fe_valuesChild.shape_value(iNode, q_point);
                    }
                }
            }

          for (unsigned int q_point = 0;
               q_point < numQuadraturePointsPerCellChild;
               ++q_point)
            {
              d_childCellJxW[(iElem * numQuadraturePointsPerCellChild) +
                             q_point] = fe_valuesChild.JxW(q_point);
            }
          iElem++;
        }
  }

  template <typename T>
  void
  inverseDFTSolverFunction<T>::preComputeParentJxW()
  {
    // Quadrature for AX multiplication will FEOrderElectro+1
    const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
        d_matrixFreeQuadratureComponentAdjointRhs);
    const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();
    const unsigned int numTotalQuadraturePointsParent =
      d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;

    dealii::FEValues<3> fe_valuesParent(d_dofHandlerParent->get_fe(),
                                        quadratureRuleParent,
                                        dealii::update_values |
                                          dealii::update_JxW_values);

    const unsigned int numberDofsPerElement =
      d_dofHandlerParent->get_fe().dofs_per_cell;

    //
    // resize data members
    //

    d_parentCellJxW.resize(numTotalQuadraturePointsParent);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell             = d_dofHandlerParent->begin_active(),
      endc             = d_dofHandlerParent->end();
    unsigned int iElem = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_valuesParent.reinit(cell);

          if (iElem == 0)
            {
              // For the reference cell initalize the shape function values
              d_shapeFunctionValueParent.resize(
                numberDofsPerElement * numQuadraturePointsPerCellParent);

              for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                   ++iNode)
                {
                  for (unsigned int q_point = 0;
                       q_point < numQuadraturePointsPerCellParent;
                       ++q_point)
                    {
                      d_shapeFunctionValueParent[iNode +
                                                 q_point *
                                                   numberDofsPerElement] =
                        fe_valuesParent.shape_value(iNode, q_point);
                    }
                }
            }
          for (unsigned int q_point = 0;
               q_point < numQuadraturePointsPerCellParent;
               ++q_point)
            {
              d_parentCellJxW[(iElem * numQuadraturePointsPerCellParent) +
                              q_point] = fe_valuesParent.JxW(q_point);
            }
          iElem++;
        }
  }
  template <typename T>
  void
  inverseDFTSolverFunction<T>::reinit(
    const std::vector<std::vector<std::vector<double>>> &rhoTargetQuadData,
    const std::vector<std::vector<std::vector<double>>> &weightQuadData,
    const std::vector<std::vector<std::vector<double>>> &potBaseQuadData,
    dftBase *                                            dft,
    const dealii::MatrixFree<3, double> &                matrixFreeDataParent,
    const dealii::MatrixFree<3, double> &                matrixFreeDataChild,
    const dealii::AffineConstraints<double>
      &constraintMatrixHomogeneousPsi, // assumes that the constraint matrix has
                                       // homogenous BC
    const dealii::AffineConstraints<double>
      &constraintMatrixHomogeneousAdjoint, // assumes that the constraint matrix
                                           // has homogenous BC
    const dealii::AffineConstraints<double> &constraintMatrixPot,
    operatorDFTClass &                       kohnShamClass,
#ifdef DFTFE_WITH_DEVICE
    operatorDFTDeviceClass &kohnShamDeviceClass,
#endif
    std::shared_ptr<TransferDataBetweenMeshesBase> & inverseDFTDoFManagerObjPtr,
    const std::vector<double> &kpointWeights,
    const unsigned int         numSpins,
    const unsigned int         numEigenValues,
    const unsigned int         matrixFreePsiVectorComponent,
    const unsigned int         matrixFreeAdjointVectorComponent,
    const unsigned int         matrixFreePotVectorComponent,
    const unsigned int         matrixFreeQuadratureComponentAdjointRhs,
    const unsigned int         matrixFreeQuadratureComponentPot,
    const bool                 isComputeDiagonalA,
    const bool                 isComputeShapeFunction,
    const dftParameters &      dftParams)
  {
    d_rhoTargetQuadData                  = rhoTargetQuadData;
    d_weightQuadData                     = weightQuadData;
    d_potBaseQuadData                    = potBaseQuadData;
    d_dft                                = dft;
    d_matrixFreeDataParent               = &matrixFreeDataParent;
    d_matrixFreeDataChild                = &matrixFreeDataChild;
    d_constraintMatrixHomogeneousPsi     = &constraintMatrixHomogeneousPsi;
    d_constraintMatrixHomogeneousAdjoint = &constraintMatrixHomogeneousAdjoint;
    d_constraintMatrixPot                = &constraintMatrixPot;
    d_kohnShamClass                      = &kohnShamClass;
#ifdef DFTFE_WITH_DEVICE
    d_kohnShamDeviceClass = &kohnShamDeviceClass;
#endif
    d_inverseDFTDoFManagerObjPtr          = inverseDFTDoFManagerObjPtr;
    d_kpointWeights                    = kpointWeights;
    d_numSpins                         = numSpins;
    d_numKPoints                       = d_kpointWeights.size();
    d_numEigenValues                   = numEigenValues;
    d_matrixFreePsiVectorComponent     = matrixFreePsiVectorComponent;
    d_matrixFreeAdjointVectorComponent = matrixFreeAdjointVectorComponent;
    d_matrixFreePotVectorComponent     = matrixFreePotVectorComponent;
    d_matrixFreeQuadratureComponentAdjointRhs =
      matrixFreeQuadratureComponentAdjointRhs;
    d_matrixFreeQuadratureComponentPot = matrixFreeQuadratureComponentPot;
    d_isComputeDiagonalA               = isComputeDiagonalA;
    d_isComputeShapeFunction           = isComputeShapeFunction;
    d_dftParams                        = &dftParams;
    d_getForceCounter                  = 0;

    d_numElectrons = d_dft->getNumElectrons();

    d_elpaScala               = d_dft->getElpaScalaManager();
    d_subspaceIterationSolver = d_dft->getSubspaceIterationSolver();

#ifdef DFTFE_WITH_DEVICE
    d_subspaceIterationSolverDevice = d_dft->getSubspaceIterationSolverDevice();
#endif

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParams->useDevice)
      {
        d_multiVectorAdjointProblemDevice.reinit(
          matrixFreeDataParent,
          constraintMatrixHomogeneousPsi,
          *d_kohnShamDeviceClass,
          matrixFreePsiVectorComponent,
          matrixFreeQuadratureComponentAdjointRhs,
          isComputeDiagonalA,
          isComputeShapeFunction);
      }
#endif
    if (!d_dftParams->useDevice)
      {
        d_multiVectorAdjointProblem.reinit(
          matrixFreeDataParent,
          constraintMatrixHomogeneousPsi,
          constraintMatrixHomogeneousPsi,
          kohnShamClass,
          matrixFreePsiVectorComponent,
          matrixFreePsiVectorComponent,
          matrixFreeQuadratureComponentAdjointRhs,
          isComputeDiagonalA,
          isComputeShapeFunction);
      }



    d_adjointTol             = d_dftParams->inverseAdjointInitialTol;
    d_adjointMaxIterations   = d_dftParams->inverseAdjointMaxIterations;
    d_maxChebyPasses         = 100; // This is hard coded
    d_fractionalOccupancyTol = d_dftParams->inverseFractionOccTol;

    //
    // TODO Hard-coded for now. Read from params
    //
    d_degeneracyTol               = d_dftParams->inverseDegeneracyTol;
    const unsigned int numKPoints = d_kpointWeights.size();
    d_wantedLower.resize(numSpins * numKPoints);
    d_unwantedUpper.resize(numSpins * numKPoints);
    d_unwantedLower.resize(numSpins * numKPoints);
    d_eigenValues.resize(numKPoints,
                         std::vector<double>(d_numSpins * d_numEigenValues,
                                             0.0));
    d_residualNormWaveFunctions.resize(numSpins * numKPoints,
                                       std::vector<double>(d_numEigenValues,
                                                           0.0));

    d_fractionalOccupancy.resize(
      d_numKPoints, std::vector<double>(d_numSpins * d_numEigenValues, 0.0));

    d_dofHandlerParent =
      &d_matrixFreeDataParent->get_dof_handler(d_matrixFreePsiVectorComponent);
    typename DoFHandler<3>::active_cell_iterator cellPtr =
      d_dofHandlerParent->begin_active();
    typename DoFHandler<3>::active_cell_iterator endcellPtr =
      d_dofHandlerParent->end();
    d_numLocallyOwnedCellsParent = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            d_numLocallyOwnedCellsParent++;
          }
      }

    d_dofHandlerChild =
      &d_matrixFreeDataChild->get_dof_handler(d_matrixFreePotVectorComponent);
    cellPtr                     = d_dofHandlerChild->begin_active();
    endcellPtr                  = d_dofHandlerChild->end();
    d_numLocallyOwnedCellsChild = 0;
    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            d_numLocallyOwnedCellsChild++;
          }
      }

    if (isComputeShapeFunction)
      {
        preComputeChildShapeFunction();
        preComputeParentJxW();
      }

    distributedCPUVec<double> dummyPotVec;
    vectorTools::createDealiiVector<double>(
      d_matrixFreeDataChild->get_vector_partitioner(
        d_matrixFreePotVectorComponent),
      1,
      dummyPotVec);
    dummyPotVec = 0.0;

    d_constraintsMatrixDataInfoPot.initialize(
      d_matrixFreeDataChild->get_vector_partitioner(
        d_matrixFreePotVectorComponent),
      *d_constraintMatrixPot);

    d_constraintsMatrixDataInfoPot.precomputeMaps(
      d_matrixFreeDataChild->get_vector_partitioner(
        d_matrixFreePotVectorComponent),
      dummyPotVec.get_partitioner(),
      1); // blockSize

    dealii::IndexSet locally_relevant_dofs_;
    dealii::DoFTools::extract_locally_relevant_dofs(*d_dofHandlerParent,
                                                    locally_relevant_dofs_);

    const dealii::IndexSet &locally_owned_dofs_ =
      d_dofHandlerParent->locally_owned_dofs();
    dealii::IndexSet ghost_indices_ = locally_relevant_dofs_;
    ghost_indices_.subtract_set(locally_owned_dofs_);

    distributedCPUVec<double> tempVec =
      distributedCPUVec<double>(locally_owned_dofs_,
                                ghost_indices_,
                                d_mpi_comm_domain);


    d_solutionPotVecForWritingInParentNodesMFVec.resize(d_numSpins);
    d_solutionPotVecForWritingInParentNodes.resize(d_numSpins);
    for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
      {
        vectorTools::createDealiiVector<double>(
          d_matrixFreeDataParent->get_vector_partitioner(
            d_matrixFreePsiVectorComponent),
          1,
          d_solutionPotVecForWritingInParentNodesMFVec[iSpin]);
        d_solutionPotVecForWritingInParentNodes[iSpin].reinit(tempVec);
      }
  }

  template <typename T>
  void
  inverseDFTSolverFunction<T>::dotProduct(const distributedCPUVec<double> &vec1,
                                          const distributedCPUVec<double> &vec2,
                                          unsigned int         blockSize,
                                          std::vector<double> &outputDot)
  {
    outputDot.resize(blockSize);
    std::fill(outputDot.begin(), outputDot.end(), 0.0);
    for (unsigned int iNode = 0; iNode < vec1.local_size(); iNode++)
      {
        outputDot[iNode % blockSize] +=
          vec1.local_element(iNode) * vec2.local_element(iNode);
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  &outputDot[0],
                  blockSize,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);
  }

  template <typename T>
  void
  inverseDFTSolverFunction<T>::computeDotProduct(
    distributedCPUMultiVec<double> &adjointOutput,
    distributedCPUMultiVec<double> &eigenVectors,
    std::vector<double> &           dotProductOutput,
    unsigned int                    blockSizeInput)
  {
    dotProductOutput.resize(blockSizeInput);
    std::fill(dotProductOutput.begin(), dotProductOutput.end(), 0.0);

    const unsigned int totalLocallyOwnedCells =
      d_matrixFreeDataParent->n_physical_cells();

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerParent->begin_active(),
      endc = d_dofHandlerParent->end();

    const dealii::Quadrature<3> &quadratureRhs =
      d_matrixFreeDataParent->get_quadrature(
        d_matrixFreeQuadratureComponentAdjointRhs);
    const unsigned int numberDofsPerElement =
      d_dofHandlerParent->get_fe().dofs_per_cell;
    const unsigned numberQuadraturePointsRhs = quadratureRhs.size();

    const unsigned int inc  = 1;
    const double       beta = 0.0, alpha = 1.0;
    char               transposeMat      = 'T';
    char               doNotTransposeMat = 'N';

    std::vector<std::vector<dealii::types::global_dof_index>>
      flattenedArrayPsiMacroCellLocalProcIndexIdMap,
      flattenedArrayPsiCellLocalProcIndexIdMap,
      flattenedArrayOutMacroCellLocalProcIndexIdMap,
      flattenedArrayOutCellLocalProcIndexIdMap;

    vectorTools::computeCellLocalIndexSetMap(
      eigenVectors.getMPIPatternP2P(),
      *d_matrixFreeDataParent,
      d_matrixFreePsiVectorComponent,
      blockSizeInput,
      flattenedArrayPsiMacroCellLocalProcIndexIdMap,
      flattenedArrayPsiCellLocalProcIndexIdMap);

    vectorTools::computeCellLocalIndexSetMap(
      adjointOutput.getMPIPatternP2P(),
      *d_matrixFreeDataParent,
      d_matrixFreeAdjointVectorComponent,
      blockSizeInput,
      flattenedArrayOutMacroCellLocalProcIndexIdMap,
      flattenedArrayOutCellLocalProcIndexIdMap);


    std::vector<double> cellLevelJxW;
    cellLevelJxW.resize(numberQuadraturePointsRhs);

    std::vector<double> cellLevelPsi, cellLevelOutput;
    cellLevelPsi.resize(numberDofsPerElement * blockSizeInput);
    cellLevelOutput.resize(numberDofsPerElement * blockSizeInput);

    std::vector<double> cellLevelPsiQuad, cellLevelOutputQuad;
    unsigned int        iElem = 0;
    cell                      = d_dofHandlerParent->begin_active();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cellLevelPsi.resize(numberDofsPerElement * blockSizeInput);
          cellLevelPsiQuad.resize(numberQuadraturePointsRhs * blockSizeInput);

          cellLevelOutput.resize(numberDofsPerElement * blockSizeInput);
          cellLevelOutputQuad.resize(numberQuadraturePointsRhs *
                                     blockSizeInput);


          for (unsigned int dofId = 0; dofId < numberDofsPerElement; dofId++)
            {
              dealii::types::global_dof_index localNodeIdInputEigen =
                flattenedArrayPsiCellLocalProcIndexIdMap[iElem][dofId];

              dcopy_(&blockSizeInput,
                     eigenVectors.data() + localNodeIdInputEigen,
                     &inc,
                     &cellLevelPsi[blockSizeInput * dofId],
                     &inc);

              dealii::types::global_dof_index localNodeIdInputAdjoint =
                flattenedArrayOutCellLocalProcIndexIdMap[iElem][dofId];

              dcopy_(&blockSizeInput,
                     adjointOutput.data() + localNodeIdInputAdjoint,
                     &inc,
                     &cellLevelOutput[blockSizeInput * dofId],
                     &inc);
            }

          dcopy_(&numberQuadraturePointsRhs,
                 &d_parentCellJxW[iElem * numberQuadraturePointsRhs],
                 &inc,
                 &cellLevelJxW[0],
                 &inc);

          dgemm_(&doNotTransposeMat,
                 &doNotTransposeMat,
                 &blockSizeInput,
                 &numberQuadraturePointsRhs,
                 &numberDofsPerElement,
                 &alpha,
                 &cellLevelPsi[0],
                 &blockSizeInput,
                 &d_shapeFunctionValueParent[0],
                 &numberDofsPerElement,
                 &beta,
                 &cellLevelPsiQuad[0],
                 &blockSizeInput);

          dgemm_(&doNotTransposeMat,
                 &doNotTransposeMat,
                 &blockSizeInput,
                 &numberQuadraturePointsRhs,
                 &numberDofsPerElement,
                 &alpha,
                 &cellLevelOutput[0],
                 &blockSizeInput,
                 &d_shapeFunctionValueParent[0],
                 &numberDofsPerElement,
                 &beta,
                 &cellLevelOutputQuad[0],
                 &blockSizeInput);
          for (unsigned int iQuad = 0; iQuad < numberQuadraturePointsRhs;
               iQuad++)
            {
              for (unsigned int iBlock = 0; iBlock < blockSizeInput; iBlock++)
                {
                  dotProductOutput[iBlock] +=
                    cellLevelOutputQuad[iQuad * blockSizeInput + iBlock] *
                    cellLevelPsiQuad[iQuad * blockSizeInput + iBlock] *
                    cellLevelJxW[iQuad];
                }
            }

          iElem++;
        }

    MPI_Allreduce(MPI_IN_PLACE,
                  &dotProductOutput[0],
                  blockSizeInput,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);
  }

  template <typename T>
  void
  inverseDFTSolverFunction<T>::writeVxcDataToFile(
    std::vector<distributedCPUVec<double>> &pot,
    unsigned int                            counter)
  {
    const unsigned int poolId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interpool);
    const unsigned int bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interband);
    const unsigned int minPoolId =
      dealii::Utilities::MPI::min(poolId, d_mpi_comm_interpool);
    const unsigned int minBandGroupId =
      dealii::Utilities::MPI::min(bandGroupId, d_mpi_comm_interband);

    if (poolId == minPoolId && bandGroupId == minBandGroupId)
      {
        auto local_range = pot[0].locally_owned_elements();
        std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
          dof_coord_child;
        dealii::DoFTools::map_dofs_to_support_points<3, 3>(
          dealii::MappingQ1<3, 3>(), *d_dofHandlerChild, dof_coord_child);
        dealii::types::global_dof_index numberDofsChild =
          d_dofHandlerChild->n_dofs();

        std::vector<std::shared_ptr<dftUtils::CompositeData>> data(0);

        const std::string filename = d_dftParams->vxcDataFolder + "/" +
                                     d_dftParams->fileNameWriteVxcPostFix +
                                     "_" + std::to_string(counter);
        for (dealii::types::global_dof_index iNode = 0; iNode < numberDofsChild;
             iNode++)
          {
            if (local_range.is_element(iNode))
              {
                std::vector<double> nodeVals(0);
                nodeVals.push_back(iNode);
                nodeVals.push_back(dof_coord_child[iNode][0]);
                nodeVals.push_back(dof_coord_child[iNode][1]);
                nodeVals.push_back(dof_coord_child[iNode][2]);

                nodeVals.push_back(pot[0][iNode]);
                if (d_numSpins == 2)
                  {
                    nodeVals.push_back(pot[1][iNode]);
                  }
                data.push_back(std::make_shared<dftUtils::NodalData>(nodeVals));
              }
          }
        std::vector<dftUtils::CompositeData *> dataRawPtrs(data.size());
        for (unsigned int i = 0; i < data.size(); ++i)
          dataRawPtrs[i] = data[i].get();
        dftUtils::MPIWriteOnFile().writeData(dataRawPtrs,
                                             filename,
                                             d_mpi_comm_domain);
      }
    //    if(counter == 1000)
    //      {
    //        auto local_range = pot[0].locally_owned_elements();
    //        std::map< dealii::types::global_dof_index,
    //        dealii::Point<3,double>>dof_coord_child ;
    //        dealii::DoFTools::map_dofs_to_support_points<3,3>(dealii::MappingQ1<3,3>()
    //        ,*d_dofHandlerChild,dof_coord_child);
    //        dealii::types::global_dof_index numberDofsChild =
    //        d_dofHandlerChild->n_dofs();
    //
    //        std::vector<std::shared_ptr<dftUtils::CompositeData> >
    //        dataNode(0);
    //
    //        const std::string filenameGlobalNode = d_dftParams->vxcDataFolder
    //        + "/testFile_withNodes"; for( dealii::types::global_dof_index
    //        iNode = 0; iNode < numberDofsChild; iNode++ )
    //          {
    //            if(local_range.is_element(iNode))
    //              {
    //                std::vector<double> nodeValsGlobal(0);
    //                nodeValsGlobal.push_back(iNode);
    //                nodeValsGlobal.push_back(dof_coord_child[iNode][0]);
    //                nodeValsGlobal.push_back(dof_coord_child[iNode][1]);
    //                nodeValsGlobal.push_back(dof_coord_child[iNode][2]);
    //
    //                nodeValsGlobal.push_back(iNode);
    //                dataNode.push_back(std::make_shared<dftUtils::NodalData>(nodeValsGlobal));
    //              }
    //          }
    //        std::vector<dftUtils::CompositeData *>
    //        dataRawPtrsGlobalNode(dataNode.size()); for(unsigned int i = 0; i
    //        < dataNode.size(); ++i)
    //          dataRawPtrsGlobalNode[i] = dataNode[i].get();
    //        dftUtils::MPIWriteOnFile().writeData(dataRawPtrsGlobalNode,
    //        filenameGlobalNode, d_mpi_comm_domain);
    //      }
  }

  template <typename T>
  void
  inverseDFTSolverFunction<T>::getForceVector(
    std::vector<distributedCPUVec<double>> &pot,
    std::vector<distributedCPUVec<double>> &force,
    std::vector<double> &                   loss)
  {
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParams->useDevice)
      {
        dealii::TimerOutput computingTimerStandard(
          d_kohnShamDeviceClass->getMPICommunicator(),
          pcout,
          d_dftParams->reproducible_output || d_dftParams->verbosity < 2 ?
            dealii::TimerOutput::never :
            dealii::TimerOutput::every_call,
          dealii::TimerOutput::wall_times);
        computingTimerStandard.enter_subsection("getForceVectorGPU on GPU");
        getForceVectorGPU(pot, force, loss);
        computingTimerStandard.leave_subsection("getForceVectorGPU on GPU");
      }
#endif

    if (!d_dftParams->useDevice)
      {
        dealii::TimerOutput computingTimerStandard(
          d_kohnShamClass->getMPICommunicator(),
          pcout,
          d_dftParams->reproducible_output || d_dftParams->verbosity < 2 ?
            dealii::TimerOutput::never :
            dealii::TimerOutput::every_call,
          dealii::TimerOutput::wall_times);
        computingTimerStandard.enter_subsection("getForceVectorCPU on CPU");
        getForceVectorCPU(pot, force, loss);
        computingTimerStandard.leave_subsection("getForceVectorCPU on CPU");
      }
  }

#ifdef DFTFE_WITH_DEVICE

  template <typename T>
  void
  inverseDFTSolverFunction<T>::solveEigenGPU(
    const std::vector<distributedCPUVec<double>> &pot)
  {
    dealii::TimerOutput computingTimerStandard(
      d_kohnShamDeviceClass->getMPICommunicator(),
      pcout,
      d_dftParams->reproducible_output || d_dftParams->verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);

    pcout << "Inside solve eigen\n";

    double potL2Norm = 0.0;

    for (unsigned int i = 0; i < pot[0].locally_owned_size(); i++)
      {
        potL2Norm += pot[0].local_element(i) * pot[0].local_element(i);
      }

    MPI_Allreduce(
      MPI_IN_PLACE, &potL2Norm, 1, MPI_DOUBLE, MPI_SUM, d_mpi_comm_domain);

    pcout << " norm2 of input pot = " << potL2Norm << "\n";
    const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
        d_matrixFreeQuadratureComponentAdjointRhs);
    const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();
    const unsigned int numTotalQuadraturePoints =
      numQuadraturePointsPerCellParent * d_numLocallyOwnedCellsParent;

    bool isFirstFilteringPass = (d_getForceCounter == 0) ? true : false;
    std::vector<std::vector<std::vector<double>>> residualNorms(
      d_numSpins,
      std::vector<std::vector<double>>(
        d_numKPoints, std::vector<double>(d_numEigenValues, 0.0)));

    double       maxResidual = 0.0;
    unsigned int iPass       = 0;
    const double chebyTol    = d_dftParams->chebyshevTolerance;
    if (d_getForceCounter > 3)
      {
        double tolPreviousIter = d_tolForChebFiltering;
        d_tolForChebFiltering =
          std::min(chebyTol, d_lossPreviousIteration / 10.0);
        d_tolForChebFiltering =
          std::min(d_tolForChebFiltering, tolPreviousIter);
      }
    else
      {
        d_tolForChebFiltering = chebyTol;
      }

    pcout << " Tol for the eigen solve = " << d_tolForChebFiltering << "\n";
    std::vector<dftfe::utils::MemoryStorage<dataTypes::number,
                                            dftfe::utils::MemorySpace::HOST>> potParentQuadData(
        d_numSpins, dftfe::utils::MemoryStorage<dataTypes::number,
                                                dftfe::utils::MemorySpace::HOST>(numTotalQuadraturePoints, 0.0));

    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        computingTimerStandard.enter_subsection(
          "interpolate child data to parent quad on CPU");
        d_inverseDFTDoFManagerObjPtr->interpolateMesh2DataToMesh1QuadPoints(
          pot[iSpin], 1, potParentQuadData[iSpin],d_resizeGPUVecDuringInterpolation);
        computingTimerStandard.leave_subsection(
          "interpolate child data to parent quad on CPU");
        std::vector<std::vector<double>> potKSQuadData(
          d_numLocallyOwnedCellsParent,
          std::vector<double>(numQuadraturePointsPerCellParent, 0.0));
        pcout << "before cell loop before compute veff eigen\n";

        double inputToHamil = 0.0;
        for (unsigned int iCell = 0; iCell < d_numLocallyOwnedCellsParent;
             ++iCell)
          {
            for (unsigned int iQuad = 0;
                 iQuad < numQuadraturePointsPerCellParent;
                 ++iQuad)
              {
                potKSQuadData[iCell][iQuad] =
                  d_potBaseQuadData[iSpin][iCell][iQuad] +
                  potParentQuadData[iSpin]
                                   [iCell * numQuadraturePointsPerCellParent +
                                    iQuad];

                inputToHamil +=
                  potKSQuadData[iCell][iQuad] * potKSQuadData[iCell][iQuad];
              }
          }

        MPI_Allreduce(MPI_IN_PLACE,
                      &inputToHamil,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpi_comm_domain);

        pcout << " norm2 of input to hamil = " << inputToHamil << "\n";
        computingTimerStandard.enter_subsection("computeVeff inverse on CPU");
        d_kohnShamDeviceClass->computeVEff(potKSQuadData);
        computingTimerStandard.leave_subsection("computeVeff inverse on CPU");

        computingTimerStandard.enter_subsection(
          "computeHamiltonianMatrix on CPU");
        d_kohnShamDeviceClass->computeHamiltonianMatricesAllkpt(iSpin);
        computingTimerStandard.leave_subsection(
          "computeHamiltonianMatrix on CPU");

        double hamilNorm = d_kohnShamDeviceClass->computeNormOfHamiltonian();
        pcout << " norm2 of hamil matrix = " << hamilNorm << "\n";
      }

    do
      {
        pcout << " inside iPass of chebFil = " << iPass << "\n";
        for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
          {
            for (unsigned int iKpoint = 0; iKpoint < d_numKPoints; ++iKpoint)
              {
                pcout
                  << "before kohnShamEigenSpaceCompute in k point loop eigen\n";
                const unsigned int kpointSpinId =
                  iSpin * d_numKPoints + iKpoint;
                computingTimerStandard.enter_subsection(
                  "reinitkPointSpinIndex on CPU");
                d_kohnShamDeviceClass->reinitkPointSpinIndex(iKpoint, iSpin);
                computingTimerStandard.leave_subsection(
                  "reinitkPointSpinIndex on CPU");

                computingTimerStandard.enter_subsection(
                  "kohnShamEigenSpaceCompute inverse on CPU");
                d_dft->kohnShamEigenSpaceCompute(
                  iSpin,
                  iKpoint,
                  *d_kohnShamDeviceClass,
                  *d_elpaScala,
                  *d_subspaceIterationSolverDevice,
                  residualNorms[iSpin][iKpoint],
                  true,  // compute residual
                  false, // spectrum splitting
                  false, // mixed precision
                  false  // is first SCF
                );
                computingTimerStandard.leave_subsection(
                  "kohnShamEigenSpaceCompute inverse on CPU");
              }
          }

        const std::vector<std::vector<double>> &eigenValues =
          d_dft->getEigenValues();
        d_dft->compute_fermienergy(eigenValues, d_numElectrons);
        const double fermiEnergy = d_dft->getFermiEnergy();
        maxResidual              = 0.0;
        for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
          {
            for (unsigned int iKpoint = 0; iKpoint < d_numKPoints; ++iKpoint)
              {
                pcout << "compute partial occupancy eigen\n";
                for (unsigned int iEig = 0; iEig < d_numEigenValues; ++iEig)
                  {
                    const double eigenValue =
                      eigenValues[iKpoint][d_numEigenValues * iSpin + iEig];
                    d_fractionalOccupancy[iKpoint][d_numEigenValues * iSpin +
                                                   iEig] =
                      dftUtils::getPartialOccupancy(eigenValue,
                                                    fermiEnergy,
                                                    C_kb,
                                                    d_dftParams->TVal);
                    if (d_fractionalOccupancy[iKpoint]
                                             [d_numEigenValues * iSpin + iEig] >
                        d_fractionalOccupancyTol)
                      {
                        if (residualNorms[iSpin][iKpoint][iEig] > maxResidual)
                          maxResidual = residualNorms[iSpin][iKpoint][iEig];
                      }
                  }
              }
          }
        iPass++;
    }
    while (maxResidual > d_tolForChebFiltering && iPass < d_maxChebyPasses);

    pcout << " maxRes = " << maxResidual << " iPass = " << iPass << "\n";
  }

  template <typename T>
  void
  inverseDFTSolverFunction<T>::getForceVectorGPU(
    std::vector<distributedCPUVec<double>> &pot,
    std::vector<distributedCPUVec<double>> &force,
    std::vector<double> &                   loss)
  {
    dealii::TimerOutput computingTimerStandard(
      d_kohnShamDeviceClass->getMPICommunicator(),
      pcout,
      d_dftParams->reproducible_output || d_dftParams->verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);



    pcout << "Inside force vector \n";
    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        d_constraintsMatrixDataInfoPot.distribute(pot[iSpin], 1);
        //      d_constraintMatrixPot->distribute(pot[iSpin]);
        //      pot[iSpin].update_ghost_values();
      }
    computingTimerStandard.enter_subsection(
      "SolveEigen in inverse call on CPU");
    this->solveEigenGPU(pot);
    computingTimerStandard.leave_subsection(
      "SolveEigen in inverse call on CPU");
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::DEVICE>
                                           *eigenVectorsDevice = d_dft->getEigenVectorsDevice();
    const std::vector<std::vector<double>> &eigenValues =
      d_dft->getEigenValues();
    const double fermiEnergy = d_dft->getFermiEnergy();
    unsigned int numLocallyOwnedDofs =
      d_dofHandlerParent->n_locally_owned_dofs();
    unsigned int numDofsPerCell = d_dofHandlerParent->get_fe().dofs_per_cell;

    const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
        d_matrixFreeQuadratureComponentAdjointRhs);
    const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();

    const dealii::Quadrature<3> &quadratureRuleChild =
      d_matrixFreeDataChild->get_quadrature(d_matrixFreeQuadratureComponentPot);
    const unsigned int numQuadraturePointsPerCellChild =
      quadratureRuleChild.size();
    const unsigned int numTotalQuadraturePointsChild =
      d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild;

    std::map<dealii::CellId, std::vector<double>> rhoValues;
    std::map<dealii::CellId, std::vector<double>> gradRhoValues;
    std::map<dealii::CellId, std::vector<double>> rhoValuesSpinPolarized;
    std::map<dealii::CellId, std::vector<double>> gradRhoValuesSpinPolarized;

    typename DoFHandler<3>::active_cell_iterator cellPtr =
      d_dofHandlerParent->begin_active();
    typename DoFHandler<3>::active_cell_iterator endcellPtr =
      d_dofHandlerParent->end();
    std::vector<double> rhoCellDummy(numQuadraturePointsPerCellParent, 0.0);
    std::vector<double> rhoSpinPolarizedCellDummy(
      d_numSpins * numQuadraturePointsPerCellParent, 0.0);
    std::vector<double> gradRhoCellDummy(3 * numQuadraturePointsPerCellParent,
                                         0.0);
    std::vector<double> gradRhoSpinPolarizedCellDummy(
      d_numSpins * 3 * numQuadraturePointsPerCellParent, 0.0);

    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const dealii::CellId cellId        = cellPtr->id();
            rhoValues[cellId]                  = rhoCellDummy;
            rhoValuesSpinPolarized[cellId]     = rhoSpinPolarizedCellDummy;
            gradRhoValues[cellId]              = gradRhoCellDummy;
            gradRhoValuesSpinPolarized[cellId] = gradRhoSpinPolarizedCellDummy;
          }
      }

    //
    // @note: Assumes:
    // 1. No constraint magnetization and hence the fermi energy up and down are
    // set to the same value (=fermi energy)
    // 2. No spectrum splitting
    //
    pcout << "computeRhoFromPSIGPU\n";
    Device::computeRhoFromPSI<T>(eigenVectorsDevice->data(),
                                 eigenVectorsDevice->data(),
                                 d_numEigenValues,
                                 d_numEigenValues,
                                 numLocallyOwnedDofs,
                                 eigenValues,
                                 fermiEnergy,
                                 fermiEnergy, // fermi energy up
                                 fermiEnergy, // fermi energy down
                                 *d_kohnShamDeviceClass,
                                 *d_dofHandlerParent,
                                 d_numLocallyOwnedCellsParent,
                                 numDofsPerCell,
                                 numQuadraturePointsPerCellParent,
                                 d_kpointWeights,
                                 &rhoValues,
                                 &gradRhoValues,
                                 &rhoValuesSpinPolarized,
                                 &gradRhoValuesSpinPolarized,
                                 true, // evaluate grad rho
                                 d_mpi_comm_parent,
                                 d_mpi_comm_interpool,
                                 d_mpi_comm_interband,
                                 *d_dftParams,
                                 false, // spectrum splitting,
                                 false  // useFEOrderRhoPlusOneGLQuad
    );

    force.resize(d_numSpins);

    std::vector<std::vector<double>> partialOccupancies(
      d_numKPoints, std::vector<double>(d_numSpins * d_numEigenValues, 0.0));

    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex)
      {
        for (unsigned int kPoint = 0; kPoint < d_numKPoints; ++kPoint)
          {
            pcout << " before partial occupancy " << kPoint << " \n";
            for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
              {
                const double eigenValue =
                  eigenValues[kPoint][d_numEigenValues * spinIndex + iWave];
                partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                           iWave] =
                  dftUtils::getPartialOccupancy(eigenValue,
                                                fermiEnergy,
                                                C_kb,
                                                d_dftParams->TVal);

                if (d_dftParams->constraintMagnetization)
                  {
                    partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                               iWave] = 1.0;
                    if (spinIndex == 0)
                      {
                        if (eigenValue > fermiEnergy) // fermi energy up
                          partialOccupancies[kPoint]
                                            [d_numEigenValues * spinIndex +
                                             iWave] = 0.0;
                      }
                    else if (spinIndex == 1)
                      {
                        if (eigenValue > fermiEnergy) // fermi energy down
                          partialOccupancies[kPoint]
                                            [d_numEigenValues * spinIndex +
                                             iWave] = 0.0;
                      }
                  }
              }
          }
      }


    std::vector<std::vector<std::vector<double>>> rhoDiff(
      d_numSpins,
      std::vector<std::vector<double>>(
        d_numLocallyOwnedCellsParent,
        std::vector<double>(numQuadraturePointsPerCellParent, 0.0)));

    //
    // @note: d_rhoTargetQuadData for spin unploarized case stores only the
    // rho_up (which is also equals tp rho_down) and not the total rho.
    // Accordingly, while taking the difference with the KS rho, we use half of
    // the total KS rho
    //

    cellPtr    = d_dofHandlerParent->begin_active();
    endcellPtr = d_dofHandlerParent->end();
    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        if (d_numSpins == 1)
          {
            unsigned int iCell = 0;
            for (; cellPtr != endcellPtr; ++cellPtr)
              {
                if (cellPtr->is_locally_owned())
                  {
                    for (unsigned int iQuad = 0;
                         iQuad < numQuadraturePointsPerCellParent;
                         ++iQuad)
                      {
                        const dealii::CellId cellId = cellPtr->id();
                        rhoDiff[iSpin][iCell][iQuad] =
                          (d_rhoTargetQuadData[iSpin][iCell][iQuad] -
                           0.5 * rhoValues[cellId][iQuad]);
                      }
                    iCell++;
                  }
              }
          }
        else
          {
            unsigned int iCell = 0;
            for (; cellPtr != endcellPtr; ++cellPtr)
              {
                if (cellPtr->is_locally_owned())
                  {
                    for (unsigned int iQuad = 0;
                         iQuad < numQuadraturePointsPerCellParent;
                         ++iQuad)
                      {
                        const dealii::CellId cellId = cellPtr->id();
                        rhoDiff[iSpin][iCell][iQuad] =
                          (d_rhoTargetQuadData[iSpin][iCell][iQuad] -
                           rhoValuesSpinPolarized[cellId]
                                                 [iQuad * d_numSpins + iSpin]);
                      }
                    iCell++;
                  }
              }
          }
      }

    std::vector<double>              lossUnWeighted(d_numSpins, 0.0);
    std::vector<double>              errorInVxc(d_numSpins, 0.0);
    std::vector<dftfe::utils::MemoryStorage<dataTypes::number,
                                            dftfe::utils::MemorySpace::HOST>> potParentQuadData(
        d_numSpins,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::HOST>(numQuadraturePointsPerCellParent *
                                                                       d_numLocallyOwnedCellsParent,
                                                                     0.0));

    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        vectorTools::createDealiiVector<double>(
          d_matrixFreeDataChild->get_vector_partitioner(
            d_matrixFreePotVectorComponent),
          1,
          force[iSpin]);

        force[iSpin] = 0.0;
        std::vector<double> sumPsiAdjointChildQuadData(
          numTotalQuadraturePointsChild, 0.0);
        std::vector<double> sumPsiAdjointChildQuadDataPartial(
          numTotalQuadraturePointsChild, 0.0);

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          sumPsiAdjointChildQuadDataDevice(numTotalQuadraturePointsChild, 0.0);

        // u = rhoTarget - rhoKS
        std::vector<std::vector<double>> uVals(
          d_numLocallyOwnedCellsParent,
          std::vector<double>(numQuadraturePointsPerCellParent));



        loss[iSpin]       = 0.0;
        errorInVxc[iSpin] = 0.0;
        d_inverseDFTDoFManagerObjPtr->interpolateMesh2DataToMesh1QuadPoints(
          pot[iSpin], 1, potParentQuadData[iSpin],d_resizeGPUVecDuringInterpolation);
        for (unsigned int iCell = 0; iCell < d_numLocallyOwnedCellsParent;
             iCell++)
          {
            for (unsigned int iQuad = 0;
                 iQuad < numQuadraturePointsPerCellParent;
                 ++iQuad)
              {
                uVals[iCell][iQuad] = rhoDiff[iSpin][iCell][iQuad] *
                                      d_weightQuadData[iSpin][iCell][iQuad];

                lossUnWeighted[iSpin] +=
                  rhoDiff[iSpin][iCell][iQuad] * rhoDiff[iSpin][iCell][iQuad] *
                  d_parentCellJxW[iCell * numQuadraturePointsPerCellParent +
                                  iQuad];

                loss[iSpin] +=
                  rhoDiff[iSpin][iCell][iQuad] * rhoDiff[iSpin][iCell][iQuad] *
                  d_weightQuadData[iSpin][iCell][iQuad] *
                  d_parentCellJxW[iCell * numQuadraturePointsPerCellParent +
                                  iQuad];

                double vxcDiff =
                  (potParentQuadData[iSpin]
                                    [iCell * numQuadraturePointsPerCellParent +
                                     iQuad] -
                   d_targetPotValuesParentQuadData[iSpin][iCell][iQuad]);
                errorInVxc[iSpin] +=
                  vxcDiff * vxcDiff *
                  d_parentCellJxW[iCell * numQuadraturePointsPerCellParent +
                                  iQuad];
              }
          }


        const unsigned int defaultBlockSize = d_dftParams->chebyWfcBlockSize;


        for (unsigned int iKPoint = 0; iKPoint < d_numKPoints; ++iKPoint)
          {
            pcout << " kpoint loop before adjoint " << iKPoint
                  << " forceVector\n";
            d_kohnShamDeviceClass->reinitkPointSpinIndex(iKPoint, iSpin);
            unsigned int jvec              = 0;
            unsigned int previousBlockSize = defaultBlockSize;
            while (jvec < d_numEigenValues)
              {
                pcout << " jvec = " << jvec << "\n";
                unsigned int currentBlockSize =
                  std::min(defaultBlockSize, d_numEigenValues - jvec);

                bool acceptCurrentBlockSize = false;

                while (!acceptCurrentBlockSize)
                  {
                    //
                    // check if last vector of this block and first vector of
                    // next block are degenerate
                    //
                    unsigned int idThisBlockLastVec =
                      jvec + currentBlockSize - 1;
                    if (idThisBlockLastVec + 1 != d_numEigenValues)
                      {
                        const double diffEigen = std::abs(
                          eigenValues[iKPoint][d_numEigenValues * iSpin +
                                               idThisBlockLastVec] -
                          eigenValues[iKPoint][d_numEigenValues * iSpin +
                                               idThisBlockLastVec + 1]);
                        if (diffEigen < d_degeneracyTol)
                          {
                            currentBlockSize--;
                          }
                        else
                          {
                            acceptCurrentBlockSize = true;
                          }
                      }
                    else
                      {
                        acceptCurrentBlockSize = true;
                      }
                  }

                pcout << " curr block size = " << currentBlockSize << "\n";
                if (currentBlockSize != previousBlockSize || jvec == 0)
                  {
                    dftfe::linearAlgebra::
                      createMultiVectorFromDealiiPartitioner(
                        d_matrixFreeDataParent->get_vector_partitioner(
                          d_matrixFreePsiVectorComponent),
                        currentBlockSize,
                        psiBlockVec);

                    dftfe::linearAlgebra::
                      createMultiVectorFromDealiiPartitioner(
                        d_matrixFreeDataParent->get_vector_partitioner(
                          d_matrixFreePsiVectorComponent),
                        currentBlockSize,
                        psiBlockVecDevice);

                    dftfe::linearAlgebra::
                      createMultiVectorFromDealiiPartitioner(
                        d_matrixFreeDataParent->get_vector_partitioner(
                          d_matrixFreePsiVectorComponent),
                        currentBlockSize,
                        multiVectorAdjointOutputWithPsiConstraints);

                    dftfe::linearAlgebra::
                      createMultiVectorFromDealiiPartitioner(
                        d_matrixFreeDataParent->get_vector_partitioner(
                          d_matrixFreePsiVectorComponent),
                        currentBlockSize,
                        multiVectorAdjointOutputWithPsiConstraintsDevice);

                    adjointInhomogenousDirichletValues.reinit(
                      multiVectorAdjointOutputWithPsiConstraints);

                    constraintsMatrixPsiDataInfo.initialize(
                      d_matrixFreeDataParent->get_vector_partitioner(
                        d_matrixFreePsiVectorComponent),
                      *d_constraintMatrixHomogeneousPsi);

                    constraintsMatrixPsiDataInfo.precomputeMaps(
                      psiBlockVec.getMPIPatternP2P(), currentBlockSize);

                    dftfe::linearAlgebra::
                      createMultiVectorFromDealiiPartitioner(
                        d_matrixFreeDataParent->get_vector_partitioner(
                          d_matrixFreeAdjointVectorComponent),
                        currentBlockSize,
                        multiVectorAdjointOutputWithAdjointConstraints);

                    dftfe::linearAlgebra::
                      createMultiVectorFromDealiiPartitioner(
                        d_matrixFreeDataParent->get_vector_partitioner(
                          d_matrixFreeAdjointVectorComponent),
                        currentBlockSize,
                        multiVectorAdjointOutputWithAdjointConstraintsDevice);

                    constraintsMatrixAdjointDataInfoDevice.initialize(
                      d_matrixFreeDataParent->get_vector_partitioner(
                        d_matrixFreeAdjointVectorComponent),
                      *d_constraintMatrixHomogeneousAdjoint);

                    constraintsMatrixAdjointDataInfoDevice.precomputeMaps(
                      multiVectorAdjointOutputWithAdjointConstraints
                        .getMPIPatternP2P(),
                      currentBlockSize);


                    constraintsMatrixAdjointDataInfo.initialize(
                      d_matrixFreeDataParent->get_vector_partitioner(
                        d_matrixFreeAdjointVectorComponent),
                      *d_constraintMatrixHomogeneousAdjoint);

                    constraintsMatrixAdjointDataInfo.precomputeMaps(
                      multiVectorAdjointOutputWithAdjointConstraints
                        .getMPIPatternP2P(),
                      currentBlockSize);

                    vectorTools::computeCellLocalIndexSetMap(
                      psiBlockVec.getMPIPatternP2P(),
                      *d_matrixFreeDataParent,
                      d_matrixFreePsiVectorComponent,
                      currentBlockSize,
                      fullFlattenedArrayCellLocalProcIndexIdMapPsi);

                    vectorTools::computeCellLocalIndexSetMap(
                      multiVectorAdjointOutputWithAdjointConstraints
                        .getMPIPatternP2P(),
                      *d_matrixFreeDataParent,
                      d_matrixFreeAdjointVectorComponent,
                      currentBlockSize,
                      fullFlattenedArrayCellLocalProcIndexIdMapAdjoint);

                    fullFlattenedArrayCellLocalProcIndexIdMapPsiDevice.resize(
                      fullFlattenedArrayCellLocalProcIndexIdMapPsi.size());
                    fullFlattenedArrayCellLocalProcIndexIdMapPsiDevice.copyFrom(
                      fullFlattenedArrayCellLocalProcIndexIdMapPsi);

                    fullFlattenedArrayCellLocalProcIndexIdMapAdjointDevice
                      .resize(fullFlattenedArrayCellLocalProcIndexIdMapAdjoint
                                .size());
                    fullFlattenedArrayCellLocalProcIndexIdMapAdjointDevice
                      .copyFrom(
                        fullFlattenedArrayCellLocalProcIndexIdMapAdjoint);
                  }

                //
                // @note We assume that there is only homogenous Dirichlet BC
                //
                adjointInhomogenousDirichletValues.setValue(0.0);

                multiVectorAdjointOutputWithPsiConstraints.setValue(0.0);

                multiVectorAdjointOutputWithPsiConstraintsDevice.setValue(0.0);


                multiVectorAdjointOutputWithAdjointConstraintsDevice.setValue(
                  0.0);

                dftfe::utils::deviceKernelsGeneric::
                  stridedCopyToBlockConstantStride(
                    currentBlockSize,
                    d_numEigenValues,
                    numLocallyOwnedDofs,
                    jvec,
                    eigenVectorsDevice->begin() +
                      (d_numSpins * iKPoint + iSpin) * d_numEigenValues,
                    psiBlockVecDevice.begin());

                dftfe::utils::deviceMemcpyD2H(
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    psiBlockVec.data()),
                  psiBlockVecDevice.data(),
                  numLocallyOwnedDofs * currentBlockSize * sizeof(double));


                std::vector<double> effectiveOrbitalOccupancy;
                std::vector<std::vector<unsigned int>> degeneracyMap(0);
                effectiveOrbitalOccupancy.resize(currentBlockSize);
                degeneracyMap.resize(currentBlockSize);
                std::vector<double> shiftValues;
                shiftValues.resize(currentBlockSize);

                for (unsigned int iBlock = 0; iBlock < currentBlockSize;
                     iBlock++)
                  {
                    shiftValues[iBlock] =
                      eigenValues[iKPoint]
                                 [d_numEigenValues * iSpin + iBlock + jvec];

                    effectiveOrbitalOccupancy[iBlock] =
                      partialOccupancies[iKPoint][d_numEigenValues * iSpin +
                                                  iBlock + jvec] *
                      d_kpointWeights[iKPoint];
                  }

                evaluateDegeneracyMap(shiftValues,
                                      degeneracyMap,
                                      d_degeneracyTol);

                d_multiVectorAdjointProblemDevice.updateInputPsi(
                  psiBlockVec,
                  psiBlockVecDevice,
                  effectiveOrbitalOccupancy,
                  degeneracyMap,
                  shiftValues,
                  currentBlockSize);

                double adjoinTolForThisIteration = 5.0 * d_tolForChebFiltering;
                d_adjointTol =
                  std::min(d_adjointTol, adjoinTolForThisIteration);
                pcout << " Tol for adjoint problem = " << d_adjointTol << "\n";
                d_multiVectorLinearMINRESSolverDevice.solveNew(
                  d_multiVectorAdjointProblemDevice,
                  uVals,
                  multiVectorAdjointOutputWithPsiConstraints,
                  multiVectorAdjointOutputWithPsiConstraintsDevice,
                  adjointInhomogenousDirichletValues,
                  currentBlockSize,
                  d_adjointTol,
                  d_adjointMaxIterations,
                  d_dftParams->verbosity,
                  true); // distributeFlag

                dftfe::utils::deviceKernelsGeneric::
                  copyValueType1ArrToValueType2Arr(
                    currentBlockSize * numLocallyOwnedDofs,
                    multiVectorAdjointOutputWithPsiConstraintsDevice.data(),
                    multiVectorAdjointOutputWithAdjointConstraintsDevice
                      .data());

                multiVectorAdjointOutputWithAdjointConstraintsDevice
                  .updateGhostValues();
                constraintsMatrixAdjointDataInfoDevice.distribute(
                  multiVectorAdjointOutputWithAdjointConstraintsDevice,
                  currentBlockSize);

                //                    if((d_getForceCounter %
                //                    d_dftParams->writeVxcFrequency == 0)
                //                    &&(d_dftParams->writeVxcData))
                //                    {
                //                        distributedCPUVec<double>
                //                        outputPiWithPsiConstraint;
                //                        distributedCPUVec<double>
                //                        outputPiWithHomoConstraint;
                //
                //                        distributedCPUVec<double> inputPsi;
                //
                //                        vectorTools::createDealiiVector<double>(
                //                                d_matrixFreeDataParent->get_vector_partitioner(
                //                                        d_matrixFreePsiVectorComponent),
                //                                1,
                //                                outputPiWithPsiConstraint);
                //
                //                        vectorTools::createDealiiVector<double>(
                //                                d_matrixFreeDataParent->get_vector_partitioner(
                //                                        d_matrixFreePsiVectorComponent),
                //                                1,
                //                                inputPsi);
                //
                //                        vectorTools::createDealiiVector<double>(
                //                                d_matrixFreeDataParent->get_vector_partitioner(
                //                                        d_matrixFreeAdjointVectorComponent),
                //                                1,
                //                                outputPiWithHomoConstraint);
                //
                //                        for ( unsigned int iNode = 0 ; iNode <
                //                        numLocallyOwnedDofs; iNode++)
                //                        {
                //                            outputPiWithHomoConstraint.local_element(iNode)
                //                            =
                //                                    multiVectorAdjointOutputWithAdjointConstraints.data()[iNode*currentBlockSize];
                //
                //                            outputPiWithPsiConstraint.local_element(iNode)
                //                            =
                //                                    multiVectorAdjointOutputWithPsiConstraints.data()[iNode*currentBlockSize]
                //                                    ;
                //
                //                            inputPsi.local_element(iNode) =
                //                                    psiBlockVec.data()[iNode*currentBlockSize]
                //                                    ;
                //                        }
                //
                //                        inputPsi.update_ghost_values();
                //                        outputPiWithHomoConstraint.update_ghost_values();
                //                        outputPiWithPsiConstraint.update_ghost_values();
                //
                //                        if ((d_getForceCounter %
                //                        d_dftParams->writeVxcFrequency  == 0)
                //                        &&(d_dftParams->writeVxcData))
                //                        {
                //                            pcout<<"writing adjoint output\n";
                //                            dealii::DataOut<3,
                //                            dealii::DoFHandler<3>>
                //                            data_out_pi;
                //
                //                            data_out_pi.attach_dof_handler(*d_dofHandlerParent);
                //
                //                            std::string outputVecName1 =
                //                            "adjointOutputWithPsiBC";
                //                            std::string outputVecName2 =
                //                            "adjointOutputWithHomoBC";
                //                            std::string outputVecName3 =
                //                            "inputPsi";
                //                            data_out_pi.add_data_vector(outputPiWithPsiConstraint,
                //                            outputVecName1);
                //                            data_out_pi.add_data_vector(outputPiWithHomoConstraint,
                //                            outputVecName2);
                //                            data_out_pi.add_data_vector(inputPsi,
                //                            outputVecName3);
                //
                //                            data_out_pi.build_patches();
                //                            data_out_pi.write_vtu_with_pvtu_record("./",
                //                            "adjointOutput",
                //                            d_getForceCounter,d_mpi_comm_domain
                //                            ,2, 4);
                //                        }
                //                    }


                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::DEVICE>
                  psiChildQuadDataDevice(numTotalQuadraturePointsChild *
                                           currentBlockSize,
                                         0.0);
                dftfe::utils::MemoryStorage<double,
                                            dftfe::utils::MemorySpace::DEVICE>
                  adjointChildQuadDataDevice(numTotalQuadraturePointsChild *
                                               currentBlockSize,
                                             0.0);

                computingTimerStandard.enter_subsection(
                  "interpolate parent data to child quad on GPU");
                d_inverseDFTDoFManagerObjPtr
                  ->interpolateMesh1DataToMesh2QuadPoints(
                    psiBlockVecDevice,
                    currentBlockSize,
                    fullFlattenedArrayCellLocalProcIndexIdMapPsiDevice,
                    psiChildQuadDataDevice,
                    d_resizeGPUVecDuringInterpolation);

                d_inverseDFTDoFManagerObjPtr
                  ->interpolateMesh1DataToMesh2QuadPoints(
                    multiVectorAdjointOutputWithAdjointConstraintsDevice,
                    currentBlockSize,
                    fullFlattenedArrayCellLocalProcIndexIdMapAdjointDevice,
                    adjointChildQuadDataDevice,
                    d_resizeGPUVecDuringInterpolation);
                computingTimerStandard.leave_subsection(
                  "interpolate parent data to child quad on GPU");

                sumPsiAdjointChildQuadDataDevice.setValue(0.0);
                dftfe::utils::deviceKernelsGeneric::addVecOverContinuousIndex(
                  numTotalQuadraturePointsChild,
                  currentBlockSize,
                  psiChildQuadDataDevice.data(),
                  adjointChildQuadDataDevice.data(),
                  sumPsiAdjointChildQuadDataDevice.data());

                dftfe::utils::deviceMemcpyD2H(
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    sumPsiAdjointChildQuadDataPartial.data()),
                  sumPsiAdjointChildQuadDataDevice.data(),
                  numTotalQuadraturePointsChild * sizeof(double));

                for (unsigned int iQuad = 0;
                     iQuad < numTotalQuadraturePointsChild;
                     ++iQuad)
                  {
                    sumPsiAdjointChildQuadData[iQuad] +=
                      sumPsiAdjointChildQuadDataPartial[iQuad];
                  }
                jvec += currentBlockSize;

                previousBlockSize = currentBlockSize;

              } // block loop
          }     // kpoint loop

        // Assumes the block size is 1
        // if that changes, change the d_flattenedArrayCellChildCellMap

        integrateWithShapeFunctionsForChildData(force[iSpin],
                                                sumPsiAdjointChildQuadData);
        pcout << "force norm = " << force[iSpin].l2_norm() << "\n";
        if ((d_getForceCounter % d_dftParams->writeVxcFrequency == 0) &&
            (d_dftParams->writeVxcData))
          {
            pcout << "writing force output\n";
            d_constraintMatrixPot->distribute(force[iSpin]);
            force[iSpin].update_ghost_values();

            /*
            dealii::DataOut<3, dealii::DoFHandler<3>> data_out_force;

                        data_out_force.attach_dof_handler(*d_dofHandlerChild);

                        std::string outputVecNameForce = "ForceOutput";

                        data_out_force.add_data_vector(force[iSpin],
            outputVecNameForce); data_out_force.write_vtu_with_pvtu_record("./",
            "forceOutput", d_getForceCounter,d_mpi_comm_domain ,2, 4);
                    */
          }
        d_constraintMatrixPot->set_zero(force[iSpin]);
        force[iSpin].zero_out_ghosts();
      } // spin loop

    if ((d_getForceCounter % d_dftParams->writeVxcFrequency == 0) &&
        (d_dftParams->writeVxcData))
      {
        std::cout << " Norm of vxc written = " << pot[0].l2_norm() << "\n";
        writeVxcDataToFile(pot, d_getForceCounter);
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  &loss[0],
                  d_numSpins,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);

    MPI_Allreduce(MPI_IN_PLACE,
                  &errorInVxc[0],
                  d_numSpins,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);

    MPI_Allreduce(MPI_IN_PLACE,
                  &lossUnWeighted[0],
                  d_numSpins,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);

    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        pcout << " iter = " << d_getForceCounter
              << " loss unweighted = " << lossUnWeighted[iSpin] << "\n";
        pcout << " iter = " << d_getForceCounter
              << " vxc norm = " << pot[iSpin].l2_norm() << "\n";
        pcout << " iter = " << d_getForceCounter
              << " error In Vxc = " << errorInVxc[iSpin] << "\n";
      }

    if ((d_getForceCounter % d_dftParams->writeVxcFrequency == 0) &&
        (d_dftParams->writeVxcData))
      {

        pcout<<"writing solution";
        dealii::DataOut<3, dealii::DoFHandler<3>> data_out;

        data_out.attach_dof_handler(*d_dofHandlerChild);

        std::string outputVecName = "solution";
        data_out.add_data_vector(pot[0], outputVecName);

        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record("./", "inverseVxc",
                                            d_getForceCounter,d_mpi_comm_domain ,2, 4);

      }

    d_lossPreviousIteration = loss[0];
    if (d_numSpins == 2)
      {
        d_lossPreviousIteration = std::min(d_lossPreviousIteration, loss[1]);
      }

    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        d_constraintMatrixPot->set_zero(pot[iSpin]);
        pot[iSpin].zero_out_ghosts();
      }
    d_getForceCounter++;
    d_resizeGPUVecDuringInterpolation = false;
  }
#endif

  template <typename T>
  void
  inverseDFTSolverFunction<T>::getForceVectorCPU(
    std::vector<distributedCPUVec<double>> &pot,
    std::vector<distributedCPUVec<double>> &force,
    std::vector<double> &                   loss)
  {
    dealii::TimerOutput computingTimerStandard(
      d_kohnShamClass->getMPICommunicator(),
      pcout,
      d_dftParams->reproducible_output || d_dftParams->verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);


    pcout << "Inside force vector \n";
    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        d_constraintsMatrixDataInfoPot.distribute(pot[iSpin], 1);
        //      d_constraintMatrixPot->distribute(pot[iSpin]);
        //      pot[iSpin].update_ghost_values();
      }
    computingTimerStandard.enter_subsection(
      "SolveEigen in inverse call on CPU");
    this->solveEigenCPU(pot);
    computingTimerStandard.leave_subsection(
      "SolveEigen in inverse call on CPU");
    const std::vector<std::vector<T>> *eigenVectors = d_dft->getEigenVectors();
    const std::vector<std::vector<double>> &eigenValues =
      d_dft->getEigenValues();
    const double fermiEnergy = d_dft->getFermiEnergy();
    unsigned int numLocallyOwnedDofs =
      d_dofHandlerParent->n_locally_owned_dofs();
    unsigned int numDofsPerCell = d_dofHandlerParent->get_fe().dofs_per_cell;

    const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
        d_matrixFreeQuadratureComponentAdjointRhs);
    const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();

    const dealii::Quadrature<3> &quadratureRuleChild =
      d_matrixFreeDataChild->get_quadrature(d_matrixFreeQuadratureComponentPot);
    const unsigned int numQuadraturePointsPerCellChild =
      quadratureRuleChild.size();
    const unsigned int numTotalQuadraturePointsChild =
      d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild;

    std::map<dealii::CellId, std::vector<double>> rhoValues;
    std::map<dealii::CellId, std::vector<double>> gradRhoValues;
    std::map<dealii::CellId, std::vector<double>> rhoValuesSpinPolarized;
    std::map<dealii::CellId, std::vector<double>> gradRhoValuesSpinPolarized;

    typename DoFHandler<3>::active_cell_iterator cellPtr =
      d_dofHandlerParent->begin_active();
    typename DoFHandler<3>::active_cell_iterator endcellPtr =
      d_dofHandlerParent->end();
    std::vector<double> rhoCellDummy(numQuadraturePointsPerCellParent, 0.0);
    std::vector<double> rhoSpinPolarizedCellDummy(
      d_numSpins * numQuadraturePointsPerCellParent, 0.0);
    std::vector<double> gradRhoCellDummy(3 * numQuadraturePointsPerCellParent,
                                         0.0);
    std::vector<double> gradRhoSpinPolarizedCellDummy(
      d_numSpins * 3 * numQuadraturePointsPerCellParent, 0.0);

    for (; cellPtr != endcellPtr; ++cellPtr)
      {
        if (cellPtr->is_locally_owned())
          {
            const dealii::CellId cellId        = cellPtr->id();
            rhoValues[cellId]                  = rhoCellDummy;
            rhoValuesSpinPolarized[cellId]     = rhoSpinPolarizedCellDummy;
            gradRhoValues[cellId]              = gradRhoCellDummy;
            gradRhoValuesSpinPolarized[cellId] = gradRhoSpinPolarizedCellDummy;
          }
      }

    //
    // @note: Assumes:
    // 1. No constraint magnetization and hence the fermi energy up and down are
    // set to the same value (=fermi energy)
    // 2. No spectrum splitting
    //
    pcout << "computeRhoFromPSICPU\n";
    computeRhoFromPSICPU<T>(*eigenVectors,
                            *eigenVectors,
                            d_numEigenValues,
                            d_numEigenValues,
                            numLocallyOwnedDofs,
                            eigenValues,
                            fermiEnergy,
                            fermiEnergy, // fermi energy up
                            fermiEnergy, // fermi energy down
                            *d_kohnShamClass,
                            *d_dofHandlerParent,
                            d_numLocallyOwnedCellsParent,
                            numDofsPerCell,
                            numQuadraturePointsPerCellParent,
                            d_kpointWeights,
                            &rhoValues,
                            &gradRhoValues,
                            &rhoValuesSpinPolarized,
                            &gradRhoValuesSpinPolarized,
                            true, // evaluate grad rho
                            d_mpi_comm_parent,
                            d_mpi_comm_interpool,
                            d_mpi_comm_interband,
                            *d_dftParams,
                            false, // spectrum splitting,
                            false  // useFEOrderRhoPlusOneGLQuad
    );

    force.resize(d_numSpins);

    std::vector<std::vector<double>> partialOccupancies(
      d_numKPoints, std::vector<double>(d_numSpins * d_numEigenValues, 0.0));

    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex)
      {
        for (unsigned int kPoint = 0; kPoint < d_numKPoints; ++kPoint)
          {
            pcout << " before partial occupancy " << kPoint << " \n";
            for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
              {
                const double eigenValue =
                  eigenValues[kPoint][d_numEigenValues * spinIndex + iWave];
                partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                           iWave] =
                  dftUtils::getPartialOccupancy(eigenValue,
                                                fermiEnergy,
                                                C_kb,
                                                d_dftParams->TVal);

                if (d_dftParams->constraintMagnetization)
                  {
                    partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                               iWave] = 1.0;
                    if (spinIndex == 0)
                      {
                        if (eigenValue > fermiEnergy) // fermi energy up
                          partialOccupancies[kPoint]
                                            [d_numEigenValues * spinIndex +
                                             iWave] = 0.0;
                      }
                    else if (spinIndex == 1)
                      {
                        if (eigenValue > fermiEnergy) // fermi energy down
                          partialOccupancies[kPoint]
                                            [d_numEigenValues * spinIndex +
                                             iWave] = 0.0;
                      }
                  }
              }
          }
      }


    std::vector<std::vector<std::vector<double>>> rhoDiff(
      d_numSpins,
      std::vector<std::vector<double>>(
        d_numLocallyOwnedCellsParent,
        std::vector<double>(numQuadraturePointsPerCellParent, 0.0)));

    //
    // @note: d_rhoTargetQuadData for spin unploarized case stores only the
    // rho_up (which is also equals tp rho_down) and not the total rho.
    // Accordingly, while taking the difference with the KS rho, we use half of
    // the total KS rho
    //

    cellPtr    = d_dofHandlerParent->begin_active();
    endcellPtr = d_dofHandlerParent->end();
    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        if (d_numSpins == 1)
          {
            unsigned int iCell = 0;
            for (; cellPtr != endcellPtr; ++cellPtr)
              {
                if (cellPtr->is_locally_owned())
                  {
                    for (unsigned int iQuad = 0;
                         iQuad < numQuadraturePointsPerCellParent;
                         ++iQuad)
                      {
                        const dealii::CellId cellId = cellPtr->id();
                        rhoDiff[iSpin][iCell][iQuad] =
                          (d_rhoTargetQuadData[iSpin][iCell][iQuad] -
                           0.5 * rhoValues[cellId][iQuad]);
                      }
                    iCell++;
                  }
              }
          }
        else
          {
            unsigned int iCell = 0;
            for (; cellPtr != endcellPtr; ++cellPtr)
              {
                if (cellPtr->is_locally_owned())
                  {
                    for (unsigned int iQuad = 0;
                         iQuad < numQuadraturePointsPerCellParent;
                         ++iQuad)
                      {
                        const dealii::CellId cellId = cellPtr->id();
                        rhoDiff[iSpin][iCell][iQuad] =
                          (d_rhoTargetQuadData[iSpin][iCell][iQuad] -
                           rhoValuesSpinPolarized[cellId]
                                                 [iQuad * d_numSpins + iSpin]);
                      }
                    iCell++;
                  }
              }
          }
      }

    std::vector<double>              lossUnWeighted(d_numSpins, 0.0);
    std::vector<double>              errorInVxc(d_numSpins, 0.0);
    std::vector<dftfe::utils::MemoryStorage<dataTypes::number,
                                            dftfe::utils::MemorySpace::HOST>> potParentQuadData(
        d_numSpins,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::HOST>(numQuadraturePointsPerCellParent *
                                                                       d_numLocallyOwnedCellsParent,
                                                                     0.0));

    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        vectorTools::createDealiiVector<double>(
          d_matrixFreeDataChild->get_vector_partitioner(
            d_matrixFreePotVectorComponent),
          1,
          force[iSpin]);

        force[iSpin] = 0.0;
        std::vector<double> sumPsiAdjointChildQuadData(
          numTotalQuadraturePointsChild, 0.0);

        // u = rhoTarget - rhoKS
        std::vector<std::vector<double>> uVals(
          d_numLocallyOwnedCellsParent,
          std::vector<double>(numQuadraturePointsPerCellParent));



        loss[iSpin]       = 0.0;
        errorInVxc[iSpin] = 0.0;
        d_inverseDFTDoFManagerObjPtr->interpolateMesh2DataToMesh1QuadPoints(
          pot[iSpin], 1, potParentQuadData[iSpin],d_resizeCPUVecDuringInterpolation);
        for (unsigned int iCell = 0; iCell < d_numLocallyOwnedCellsParent;
             iCell++)
          {
            for (unsigned int iQuad = 0;
                 iQuad < numQuadraturePointsPerCellParent;
                 ++iQuad)
              {
                uVals[iCell][iQuad] = rhoDiff[iSpin][iCell][iQuad] *
                                      d_weightQuadData[iSpin][iCell][iQuad];

                lossUnWeighted[iSpin] +=
                  rhoDiff[iSpin][iCell][iQuad] * rhoDiff[iSpin][iCell][iQuad] *
                  d_parentCellJxW[iCell * numQuadraturePointsPerCellParent +
                                  iQuad];

                loss[iSpin] +=
                  rhoDiff[iSpin][iCell][iQuad] * rhoDiff[iSpin][iCell][iQuad] *
                  d_weightQuadData[iSpin][iCell][iQuad] *
                  d_parentCellJxW[iCell * numQuadraturePointsPerCellParent +
                                  iQuad];

                double vxcDiff =
                  (potParentQuadData[iSpin]
                                    [iCell * numQuadraturePointsPerCellParent +
                                     iQuad] -
                   d_targetPotValuesParentQuadData[iSpin][iCell][iQuad]);
                errorInVxc[iSpin] +=
                  vxcDiff * vxcDiff *
                  d_parentCellJxW[iCell * numQuadraturePointsPerCellParent +
                                  iQuad];
              }
          }


        const unsigned int defaultBlockSize = d_dftParams->chebyWfcBlockSize;

        for (unsigned int iKPoint = 0; iKPoint < d_numKPoints; ++iKPoint)
          {
            pcout << " kpoint loop before adjoint " << iKPoint
                  << " forceVector\n";
            d_kohnShamClass->reinitkPointSpinIndex(iKPoint, iSpin);

            const std::vector<T> &psiAtKPoint =
              (*eigenVectors)[d_numSpins * iKPoint + iSpin];

            unsigned int jvec              = 0;
            unsigned int previousBlockSize = defaultBlockSize;
            while (jvec < d_numEigenValues)
              {
                pcout << " jvec = " << jvec << "\n";
                unsigned int currentBlockSize =
                  std::min(defaultBlockSize, d_numEigenValues - jvec);

                bool acceptCurrentBlockSize = false;

                while (!acceptCurrentBlockSize)
                  {
                    //
                    // check if last vector of this block and first vector of
                    // next block are degenerate
                    //
                    unsigned int idThisBlockLastVec =
                      jvec + currentBlockSize - 1;
                    if (idThisBlockLastVec + 1 != d_numEigenValues)
                      {
                        const double diffEigen = std::abs(
                          eigenValues[iKPoint][d_numEigenValues * iSpin +
                                               idThisBlockLastVec] -
                          eigenValues[iKPoint][d_numEigenValues * iSpin +
                                               idThisBlockLastVec + 1]);
                        if (diffEigen < d_degeneracyTol)
                          {
                            currentBlockSize--;
                          }
                        else
                          {
                            acceptCurrentBlockSize = true;
                          }
                      }
                    else
                      {
                        acceptCurrentBlockSize = true;
                      }
                  }

                pcout << " curr block size = " << currentBlockSize << "\n";
                if (currentBlockSize != previousBlockSize || jvec == 0)
                  {
                    dftfe::linearAlgebra::
                      createMultiVectorFromDealiiPartitioner(
                        d_matrixFreeDataParent->get_vector_partitioner(
                          d_matrixFreePsiVectorComponent),
                        currentBlockSize,
                        psiBlockVec);

                    dftfe::linearAlgebra::
                      createMultiVectorFromDealiiPartitioner(
                        d_matrixFreeDataParent->get_vector_partitioner(
                          d_matrixFreePsiVectorComponent),
                        currentBlockSize,
                        multiVectorAdjointOutputWithPsiConstraints);

                    adjointInhomogenousDirichletValues.reinit(
                      multiVectorAdjointOutputWithPsiConstraints);

                    constraintsMatrixPsiDataInfo.initialize(
                      d_matrixFreeDataParent->get_vector_partitioner(
                        d_matrixFreePsiVectorComponent),
                      *d_constraintMatrixHomogeneousPsi);

                    constraintsMatrixPsiDataInfo.precomputeMaps(
                      psiBlockVec.getMPIPatternP2P(), currentBlockSize);

                    dftfe::linearAlgebra::
                      createMultiVectorFromDealiiPartitioner(
                        d_matrixFreeDataParent->get_vector_partitioner(
                          d_matrixFreeAdjointVectorComponent),
                        currentBlockSize,
                        multiVectorAdjointOutputWithAdjointConstraints);

                    constraintsMatrixAdjointDataInfo.initialize(
                      d_matrixFreeDataParent->get_vector_partitioner(
                        d_matrixFreeAdjointVectorComponent),
                      *d_constraintMatrixHomogeneousAdjoint);

                    constraintsMatrixAdjointDataInfo.precomputeMaps(
                      multiVectorAdjointOutputWithAdjointConstraints
                        .getMPIPatternP2P(),
                      currentBlockSize);

                    vectorTools::computeCellLocalIndexSetMap(
                      psiBlockVec.getMPIPatternP2P(),
                      *d_matrixFreeDataParent,
                      d_matrixFreePsiVectorComponent,
                      currentBlockSize,
                      fullFlattenedArrayCellLocalProcIndexIdMapPsi);

                    vectorTools::computeCellLocalIndexSetMap(
                      multiVectorAdjointOutputWithAdjointConstraints
                        .getMPIPatternP2P(),
                      *d_matrixFreeDataParent,
                      d_matrixFreeAdjointVectorComponent,
                      currentBlockSize,
                      fullFlattenedArrayCellLocalProcIndexIdMapAdjoint);
                  }

                //
                // @note We assume that there is only homogenous Dirichlet BC
                //
                adjointInhomogenousDirichletValues.setValue(0.0);

                multiVectorAdjointOutputWithPsiConstraints.setValue(0.0);

                multiVectorAdjointOutputWithAdjointConstraints.setValue(0.0);

                for (unsigned int iNode = 0; iNode < numLocallyOwnedDofs;
                     ++iNode)
                  {
                    for (unsigned int iWave = 0; iWave < currentBlockSize;
                         ++iWave)
                      {
                        psiBlockVec.data()[iNode * currentBlockSize + iWave] =
                          psiAtKPoint[iNode * d_numEigenValues + jvec + iWave];
                      }
                  }

                constraintsMatrixPsiDataInfo.distribute(psiBlockVec,
                                                        currentBlockSize);

                std::vector<double> effectiveOrbitalOccupancy;
                std::vector<std::vector<unsigned int>> degeneracyMap(0);
                effectiveOrbitalOccupancy.resize(currentBlockSize);
                degeneracyMap.resize(currentBlockSize);
                std::vector<double> shiftValues;
                shiftValues.resize(currentBlockSize);

                for (unsigned int iBlock = 0; iBlock < currentBlockSize;
                     iBlock++)
                  {
                    shiftValues[iBlock] =
                      eigenValues[iKPoint]
                                 [d_numEigenValues * iSpin + iBlock + jvec];

                    effectiveOrbitalOccupancy[iBlock] =
                      partialOccupancies[iKPoint][d_numEigenValues * iSpin +
                                                  iBlock + jvec] *
                      d_kpointWeights[iKPoint];
                  }

                evaluateDegeneracyMap(shiftValues,
                                      degeneracyMap,
                                      d_degeneracyTol);

                d_multiVectorAdjointProblem.updateInputPsi(
                  psiBlockVec,
                  effectiveOrbitalOccupancy,
                  degeneracyMap,
                  shiftValues,
                  currentBlockSize);

                double adjoinTolForThisIteration = 5.0 * d_tolForChebFiltering;
                d_adjointTol =
                  std::min(d_adjointTol, adjoinTolForThisIteration);
                pcout << " Tol for adjoint problem = " << d_adjointTol << "\n";

                d_multiVectorLinearMINRESSolver.solveNew(
                  d_multiVectorAdjointProblem,
                  uVals,
                  multiVectorAdjointOutputWithPsiConstraints,
                  adjointInhomogenousDirichletValues,
                  currentBlockSize,
                  d_adjointTol,
                  d_adjointMaxIterations,
                  d_dftParams->verbosity,
                  true); // distributeFlag



                for (unsigned int iNode = 0;
                     iNode < currentBlockSize * numLocallyOwnedDofs;
                     iNode++)
                  {
                    multiVectorAdjointOutputWithAdjointConstraints
                      .data()[iNode] =
                      multiVectorAdjointOutputWithPsiConstraints.data()[iNode];
                  }
                constraintsMatrixAdjointDataInfo.distribute(
                  multiVectorAdjointOutputWithAdjointConstraints,
                  currentBlockSize);


                if ((d_getForceCounter % d_dftParams->writeVxcFrequency == 0) &&
                    (d_dftParams->writeVxcData))
                  {
                    distributedCPUVec<double> outputPiWithPsiConstraint;
                    distributedCPUVec<double> outputPiWithHomoConstraint;

                    distributedCPUVec<double> inputPsi;

                    vectorTools::createDealiiVector<double>(
                      d_matrixFreeDataParent->get_vector_partitioner(
                        d_matrixFreePsiVectorComponent),
                      1,
                      outputPiWithPsiConstraint);

                    vectorTools::createDealiiVector<double>(
                      d_matrixFreeDataParent->get_vector_partitioner(
                        d_matrixFreePsiVectorComponent),
                      1,
                      inputPsi);

                    vectorTools::createDealiiVector<double>(
                      d_matrixFreeDataParent->get_vector_partitioner(
                        d_matrixFreeAdjointVectorComponent),
                      1,
                      outputPiWithHomoConstraint);

                    for (unsigned int iNode = 0; iNode < numLocallyOwnedDofs;
                         iNode++)
                      {
                        outputPiWithHomoConstraint.local_element(iNode) =
                          multiVectorAdjointOutputWithAdjointConstraints
                            .data()[iNode * currentBlockSize];

                        outputPiWithPsiConstraint.local_element(iNode) =
                          multiVectorAdjointOutputWithPsiConstraints
                            .data()[iNode * currentBlockSize];

                        inputPsi.local_element(iNode) =
                          psiBlockVec.data()[iNode * currentBlockSize];
                      }

                    inputPsi.update_ghost_values();
                    outputPiWithHomoConstraint.update_ghost_values();
                    outputPiWithPsiConstraint.update_ghost_values();

                    if (d_getForceCounter % d_dftParams->writeVxcFrequency == 0)
                      {
                        /*
                            pcout<<"writing adjoint output\n";
                            dealii::DataOut<3, dealii::DoFHandler<3>>
                           data_out_pi;

                            data_out_pi.attach_dof_handler(*d_dofHandlerParent);

                            std::string outputVecName1 =
                           "adjointOutputWithPsiBC"; std::string outputVecName2
                           = "adjointOutputWithHomoBC"; std::string
                           outputVecName3 = "inputPsi";
                            data_out_pi.add_data_vector(outputPiWithPsiConstraint,
                           outputVecName1);
                            data_out_pi.add_data_vector(outputPiWithHomoConstraint,
                           outputVecName2);
                            data_out_pi.add_data_vector(inputPsi,
                           outputVecName3);

                            data_out_pi.build_patches();
                            data_out_pi.write_vtu_with_pvtu_record("./",
                           "adjointOutput", d_getForceCounter,d_mpi_comm_domain
                           ,2, 4);
                      */
                      }
                  }


                dftfe::utils::MemoryStorage<dataTypes::number,
                                            dftfe::utils::MemorySpace::HOST> psiChildQuadData(
                    numTotalQuadraturePointsChild * currentBlockSize, 0.0);
                dftfe::utils::MemoryStorage<dataTypes::number,
                                            dftfe::utils::MemorySpace::HOST> adjointChildQuadData(
                    numTotalQuadraturePointsChild * currentBlockSize, 0.0);

                computingTimerStandard.enter_subsection(
                  "interpolate parent data to child quad on CPU");
                d_inverseDFTDoFManagerObjPtr
                  ->interpolateMesh1DataToMesh2QuadPoints(
                    psiBlockVec,
                    currentBlockSize,
                    fullFlattenedArrayCellLocalProcIndexIdMapPsi,
                    psiChildQuadData,
                    d_resizeCPUVecDuringInterpolation);

                d_inverseDFTDoFManagerObjPtr
                  ->interpolateMesh1DataToMesh2QuadPoints(
                    multiVectorAdjointOutputWithAdjointConstraints,
                    currentBlockSize,
                    fullFlattenedArrayCellLocalProcIndexIdMapAdjoint,
                    adjointChildQuadData,
                    d_resizeCPUVecDuringInterpolation);
                computingTimerStandard.leave_subsection(
                  "interpolate parent data to child quad on CPU");

                for (unsigned int iQuad = 0;
                     iQuad < numTotalQuadraturePointsChild;
                     ++iQuad)
                  {
                    for (unsigned int iVec = 0; iVec < currentBlockSize; ++iVec)
                      {
                        sumPsiAdjointChildQuadData[iQuad] +=
                          psiChildQuadData[iQuad * currentBlockSize + iVec] *
                          adjointChildQuadData[iQuad * currentBlockSize + iVec];
                      }
                  }

                jvec += currentBlockSize;

                previousBlockSize = currentBlockSize;

              } // block loop
          }     // kpoint loop

        // Assumes the block size is 1
        // if that changes, change the d_flattenedArrayCellChildCellMap

        integrateWithShapeFunctionsForChildData(force[iSpin],
                                                sumPsiAdjointChildQuadData);
        pcout << "force norm = " << force[iSpin].l2_norm() << "\n";
        if ((d_getForceCounter % d_dftParams->writeVxcFrequency == 0) &&
            (d_dftParams->writeVxcData))
          {
            /*
                  pcout<<"writing force output\n";
                  d_constraintMatrixPot->distribute(force[iSpin]);
                  force[iSpin].update_ghost_values();
                  dealii::DataOut<3, dealii::DoFHandler<3>> data_out_force;

                  data_out_force.attach_dof_handler(*d_dofHandlerChild);

                  std::string outputVecNameForce = "ForceOutput";

                  data_out_force.add_data_vector(force[iSpin],
               outputVecNameForce);
                  data_out_force.write_vtu_with_pvtu_record("./", "forceOutput",
               d_getForceCounter,d_mpi_comm_domain ,2, 4);
                */
          }
        d_constraintMatrixPot->set_zero(force[iSpin]);
        force[iSpin].zero_out_ghosts();
      } // spin loop

    if ((d_getForceCounter % d_dftParams->writeVxcFrequency == 0) &&
        (d_dftParams->writeVxcData))
      {
        std::cout << " Norm of vxc written = " << pot[0].l2_norm() << "\n";
        writeVxcDataToFile(pot, d_getForceCounter);
      }
    MPI_Allreduce(MPI_IN_PLACE,
                  &loss[0],
                  d_numSpins,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);

    MPI_Allreduce(MPI_IN_PLACE,
                  &errorInVxc[0],
                  d_numSpins,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);

    MPI_Allreduce(MPI_IN_PLACE,
                  &lossUnWeighted[0],
                  d_numSpins,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);

    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        pcout << " iter = " << d_getForceCounter
              << " loss unweighted = " << lossUnWeighted[iSpin] << "\n";
        pcout << " iter = " << d_getForceCounter
              << " vxc norm = " << pot[iSpin].l2_norm() << "\n";
        pcout << " iter = " << d_getForceCounter
              << " error In Vxc = " << errorInVxc[iSpin] << "\n";
      }

    if (d_getForceCounter % d_dftParams->writeVxcFrequency == 0)
      {

        pcout<<"writing solution";
        dealii::DataOut<3, dealii::DoFHandler<3>> data_out;

        data_out.attach_dof_handler(*d_dofHandlerChild);

        std::string outputVecName = "solution";
        data_out.add_data_vector(pot[0], outputVecName);

        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record("./", "inverseVxc",
                                            d_getForceCounter,d_mpi_comm_domain ,2, 4);

      }

    d_lossPreviousIteration = loss[0];
    if (d_numSpins == 2)
      {
        d_lossPreviousIteration = std::min(d_lossPreviousIteration, loss[1]);
      }

    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        d_constraintMatrixPot->set_zero(pot[iSpin]);
        pot[iSpin].zero_out_ghosts();
      }
    d_getForceCounter++;
    d_resizeCPUVecDuringInterpolation = false;
  }



  template <typename T>
  void
  inverseDFTSolverFunction<T>::integrateWithShapeFunctionsForChildData(
    distributedCPUVec<double> &outputVec,
    const std::vector<double> &quadInputData)
  {
    const dealii::Quadrature<3> &quadratureRuleChild =
      d_matrixFreeDataChild->get_quadrature(d_matrixFreeQuadratureComponentPot);
    const unsigned int numQuadraturePointsPerCellChild =
      quadratureRuleChild.size();
    const unsigned int numTotalQuadraturePointsChild =
      d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild;

    const double       alpha          = 1.0;
    const double       beta           = 0.0;
    const unsigned int inc            = 1;
    const unsigned int blockSizeInput = 1;
    char               doNotTans      = 'N';
    pcout << " inside integrateWithShapeFunctionsForChildData doNotTans = "
          << doNotTans << "\n";

    const unsigned int numberDofsPerElement =
      d_dofHandlerChild->get_fe().dofs_per_cell;

    std::vector<double> cellLevelNodalOutput(numberDofsPerElement);
    std::vector<double> cellLevelQuadInput(numQuadraturePointsPerCellChild);
    std::vector<dealii::types::global_dof_index> localDofIndices(
      numberDofsPerElement);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell             = d_dofHandlerChild->begin_active(),
      endc             = d_dofHandlerChild->end();
    unsigned int iElem = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(localDofIndices);
          std::fill(cellLevelNodalOutput.begin(),
                    cellLevelNodalOutput.end(),
                    0.0);

          std::copy(quadInputData.begin() +
                      (iElem * numQuadraturePointsPerCellChild),
                    quadInputData.begin() +
                      ((iElem + 1) * numQuadraturePointsPerCellChild),
                    cellLevelQuadInput.begin());

          for (unsigned int q_point = 0;
               q_point < numQuadraturePointsPerCellChild;
               ++q_point)
            {
              cellLevelQuadInput[q_point] =
                cellLevelQuadInput[q_point] *
                d_childCellJxW[(iElem * numQuadraturePointsPerCellChild) +
                               q_point];
            }

          dgemm_(&doNotTans,
                 &doNotTans,
                 &blockSizeInput,
                 &numberDofsPerElement,
                 &numQuadraturePointsPerCellChild,
                 &alpha,
                 &cellLevelQuadInput[0],
                 &blockSizeInput,
                 &d_childCellShapeFunctionValue[0],
                 &numQuadraturePointsPerCellChild,
                 &beta,
                 &cellLevelNodalOutput[0],
                 &blockSizeInput);

          d_constraintMatrixPot->distribute_local_to_global(
            cellLevelNodalOutput, localDofIndices, outputVec);

          iElem++;
        }
    outputVec.compress(dealii::VectorOperation::add);
  }


  template <typename T>
  void
  inverseDFTSolverFunction<T>::solveEigenCPU(
    const std::vector<distributedCPUVec<double>> &pot)
  {
    dealii::TimerOutput computingTimerStandard(
      d_kohnShamClass->getMPICommunicator(),
      pcout,
      d_dftParams->reproducible_output || d_dftParams->verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);

    pcout << "Inside solve eigen\n";
    double potL2Norm = 0.0;
    for (unsigned int i = 0; i < pot[0].locally_owned_size(); i++)
      {
        potL2Norm += pot[0].local_element(i) * pot[0].local_element(i);
      }

    MPI_Allreduce(
      MPI_IN_PLACE, &potL2Norm, 1, MPI_DOUBLE, MPI_SUM, d_mpi_comm_domain);

    pcout << " norm2 of input pot = " << potL2Norm << "\n";
    const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
        d_matrixFreeQuadratureComponentAdjointRhs);
    const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();
    const unsigned int numTotalQuadraturePoints =
      numQuadraturePointsPerCellParent * d_numLocallyOwnedCellsParent;

    bool isFirstFilteringPass = (d_getForceCounter == 0) ? true : false;
    std::vector<std::vector<std::vector<double>>> residualNorms(
      d_numSpins,
      std::vector<std::vector<double>>(
        d_numKPoints, std::vector<double>(d_numEigenValues, 0.0)));

    double       maxResidual = 0.0;
    unsigned int iPass       = 0;
    // const double chebyTol = d_dftParams->chebyshevTolerance;
    const double chebyTol = 1e-10;
    if (d_getForceCounter > 3)
      {
        double tolPreviousIter = d_tolForChebFiltering;
        d_tolForChebFiltering =
          std::min(chebyTol, d_lossPreviousIteration / 10.0);
        d_tolForChebFiltering =
          std::min(d_tolForChebFiltering, tolPreviousIter);
      }
    else
      {
        d_tolForChebFiltering = chebyTol;
      }

    pcout << " Tol for the eigen solve = " << d_tolForChebFiltering << "\n";
    std::vector< dftfe::utils::MemoryStorage<dataTypes::number,
                                            dftfe::utils::MemorySpace::HOST>> potParentQuadData(
        d_numSpins, dftfe::utils::MemoryStorage<dataTypes::number,
                                                dftfe::utils::MemorySpace::HOST>(numTotalQuadraturePoints, 0.0));

    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
      {
        computingTimerStandard.enter_subsection(
          "interpolate child data to parent quad on CPU");
        d_inverseDFTDoFManagerObjPtr->interpolateMesh2DataToMesh1QuadPoints(
          pot[iSpin], 1, potParentQuadData[iSpin],
          d_resizeCPUVecDuringInterpolation);
        computingTimerStandard.leave_subsection(
          "interpolate child data to parent quad on CPU");
        std::vector<std::vector<double>> potKSQuadData(
          d_numLocallyOwnedCellsParent,
          std::vector<double>(numQuadraturePointsPerCellParent, 0.0));
        pcout << "before cell loop before compute veff eigen\n";
        double inputToHamil = 0.0;
        for (unsigned int iCell = 0; iCell < d_numLocallyOwnedCellsParent;
             ++iCell)
          {
            for (unsigned int iQuad = 0;
                 iQuad < numQuadraturePointsPerCellParent;
                 ++iQuad)
              {
                potKSQuadData[iCell][iQuad] =
                  d_potBaseQuadData[iSpin][iCell][iQuad] +
                  potParentQuadData[iSpin]
                                   [iCell * numQuadraturePointsPerCellParent +
                                    iQuad];

                inputToHamil +=
                  potKSQuadData[iCell][iQuad] * potKSQuadData[iCell][iQuad];
              }
          }

        MPI_Allreduce(MPI_IN_PLACE,
                      &inputToHamil,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      d_mpi_comm_domain);

        pcout << " norm2 of input to hamil = " << inputToHamil << "\n";

        computingTimerStandard.enter_subsection("computeVeff inverse on CPU");
        d_kohnShamClass->computeVEff(potKSQuadData);
        computingTimerStandard.leave_subsection("computeVeff inverse on CPU");
        for (unsigned int iKpoint = 0; iKpoint < d_numKPoints; ++iKpoint)
          {
            computingTimerStandard.enter_subsection(
              "reinitkPointSpinIndex on CPU");
            d_kohnShamClass->reinitkPointSpinIndex(iKpoint, iSpin);
            computingTimerStandard.leave_subsection(
              "reinitkPointSpinIndex on CPU");

            computingTimerStandard.enter_subsection(
              "computeHamiltonianMatrix on CPU");
            d_kohnShamClass->computeHamiltonianMatrix(iKpoint, iSpin);
            computingTimerStandard.leave_subsection(
              "computeHamiltonianMatrix on CPU");
            double hamilNorm =
              d_kohnShamClass->computeNormOfHamiltonian(iKpoint, iSpin);
            pcout << " norm2 of hamil matrix = " << hamilNorm << "\n";
          }
      }

    do
      {
        pcout << " inside iPass of chebFil = " << iPass << "\n";
        for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
          {
            for (unsigned int iKpoint = 0; iKpoint < d_numKPoints; ++iKpoint)
              {
                pcout
                  << "before kohnShamEigenSpaceCompute in k point loop eigen\n";
                const unsigned int kpointSpinId =
                  iSpin * d_numKPoints + iKpoint;
                computingTimerStandard.enter_subsection(
                  "reinitkPointSpinIndex on CPU");
                d_kohnShamClass->reinitkPointSpinIndex(iKpoint, iSpin);
                computingTimerStandard.leave_subsection(
                  "reinitkPointSpinIndex on CPU");

                computingTimerStandard.enter_subsection(
                  "kohnShamEigenSpaceCompute inverse on CPU");
                d_dft->kohnShamEigenSpaceCompute(iSpin,
                                                 iKpoint,
                                                 *d_kohnShamClass,
                                                 *d_elpaScala,
                                                 *d_subspaceIterationSolver,
                                                 residualNorms[iSpin][iKpoint],
                                                 true,  // compute residual
                                                 false, // spectrum splitting
                                                 false, // mixed precision
                                                 false  // is first SCF
                );
                computingTimerStandard.leave_subsection(
                  "kohnShamEigenSpaceCompute inverse on CPU");
              }
          }

        const std::vector<std::vector<double>> &eigenValues =
          d_dft->getEigenValues();
        d_dft->compute_fermienergy(eigenValues, d_numElectrons);
        const double fermiEnergy = d_dft->getFermiEnergy();
        maxResidual              = 0.0;
        for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
          {
            for (unsigned int iKpoint = 0; iKpoint < d_numKPoints; ++iKpoint)
              {
                pcout << "compute partial occupancy eigen\n";
                for (unsigned int iEig = 0; iEig < d_numEigenValues; ++iEig)
                  {
                    const double eigenValue =
                      eigenValues[iKpoint][d_numEigenValues * iSpin + iEig];
                    d_fractionalOccupancy[iKpoint][d_numEigenValues * iSpin +
                                                   iEig] =
                      dftUtils::getPartialOccupancy(eigenValue,
                                                    fermiEnergy,
                                                    C_kb,
                                                    d_dftParams->TVal);
                    if (d_fractionalOccupancy[iKpoint]
                                             [d_numEigenValues * iSpin + iEig] >
                        d_fractionalOccupancyTol)
                      {
                        if (residualNorms[iSpin][iKpoint][iEig] > maxResidual)
                          maxResidual = residualNorms[iSpin][iKpoint][iEig];
                      }
                  }
              }
          }
        iPass++;
    }
    while (maxResidual > d_tolForChebFiltering && iPass < d_maxChebyPasses);

    pcout << " maxRes = " << maxResidual << " iPass = " << iPass << "\n";
  }

  // TODO changed for debugging purposes
  template <typename T>
  void
  inverseDFTSolverFunction<T>::setInitialGuess(
    const std::vector<distributedCPUVec<double>> &pot,
    const std::vector<std::vector<std::vector<double>>>
      &targetPotValuesParentQuadData)
  {
    d_pot = pot;
    const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
        d_matrixFreeQuadratureComponentAdjointRhs);
    const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();

    d_targetPotValuesParentQuadData.resize(d_numSpins);
    for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
      {
        d_targetPotValuesParentQuadData[iSpin].resize(
          d_numLocallyOwnedCellsParent);
        for (unsigned int iCell = 0; iCell < d_numLocallyOwnedCellsParent;
             iCell++)
          {
            d_targetPotValuesParentQuadData[iSpin][iCell].resize(
              numQuadraturePointsPerCellParent, 0.0);
            for (unsigned int iQuad = 0;
                 iQuad < numQuadraturePointsPerCellParent;
                 ++iQuad)
              {
                d_targetPotValuesParentQuadData[iSpin][iCell][iQuad] =
                  targetPotValuesParentQuadData[iSpin][iCell][iQuad];
              }
          }
      }

    std::vector<distributedCPUVec<double>> potTest;
    potTest.resize(1);
    potTest[0].reinit(d_pot[0], false);
    potTest[0] = 0.0;

    double potTestNorm  = potTest[0].l2_norm();
    double potInputNorm = d_pot[0].l2_norm();

    pcout << " norm of input pot = " << potInputNorm << "\n";
    pcout << " norm of trial vec initial = " << potTestNorm << "\n";


    //    writeVxcDataToFile(d_pot,1000);

    pcout<<"writing solution";
    dealii::DataOut<3, dealii::DoFHandler<3>> data_out;

    data_out.attach_dof_handler(*d_dofHandlerChild);

    std::string outputVecName = "solution";
    data_out.add_data_vector(d_pot[0], outputVecName);

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record("./", "inverseVxc_start",
                                        1000,d_mpi_comm_domain ,2, 4);
  }

  template <typename T>
  std::vector<distributedCPUVec<double>>
  inverseDFTSolverFunction<T>::getInitialGuess() const
  {
    return d_pot;
  }

  template <typename T>
  void
  inverseDFTSolverFunction<T>::setSolution(
    const std::vector<distributedCPUVec<double>> &pot)
  {
    d_pot = pot;
  }

  template class inverseDFTSolverFunction<dataTypes::number>;
  //  template class inverseDFTSolverFunction<float>;
  //  template class inverseDFTSolverFunction<double>;
  //  template class inverseDFTSolverFunction<std::complex<float>>;
  //  template class inverseDFTSolverFunction<std::complex<double>>;


} // end of namespace dftfe
