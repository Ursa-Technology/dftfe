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

#include "transferDataBetweenMeshesIncompatiblePartitioning.h"
#include <inverseDFTDoFManager.h>
#include "inverseDFT.h"
#include "BFGSInverseDFTSolver.h"
#include "inverseDFTSolverFunction.h"
#include <densityCalculatorCPU.h>
#include <gaussianFunctionManager.h>
namespace dftfe
{
  namespace
  {
    double
    realPart(const double x)
    {
      return x;
    }

    double
    realPart(const std::complex<double> x)
    {
      return x.real();
    }

    double
    complexConj(const double x)
    {
      return x;
    }

    std::complex<double>
    complexConj(const std::complex<double> x)
    {
      return std::conj(x);
    }

    float
    realPart(const float x)
    {
      return x;
    }

    float
    realPart(const std::complex<float> x)
    {
      return x.real();
    }

    float
    complexConj(const float x)
    {
      return x;
    }

    std::complex<float>
    complexConj(const std::complex<float> x)
    {
      return std::conj(x);
    }

    struct coordinateValues
    {
      double iNode;
      double xcoord;
      double ycoord;
      double zcoord;
      double value0;
      double value1;
    };

    struct less_than_key
    {
      inline bool
      operator()(const coordinateValues &lhs, const coordinateValues &rhs)
      {
        double tol = 1e-6;
        if (lhs.iNode - rhs.iNode < -tol)
          {
            return true;
          }
        return false;


        double xdiff = lhs.xcoord - rhs.xcoord;
        double ydiff = lhs.ycoord - rhs.ycoord;
        double zdiff = lhs.zcoord - rhs.zcoord;
        if (xdiff < -tol)
          return true;
        if (xdiff > tol)
          return false;

        if (ydiff < -tol)
          return true;
        if (ydiff > tol)
          return false;

        if (zdiff < -tol)
          return true;
        if (zdiff > tol)
          return false;

        return false;
        // AssertThrow(
        //   (std::abs(xdiff) > tol) || (std::abs(ydiff) > tol) ||
        //   (std::abs(zdiff) > tol), ExcMessage(
        //     "DFT-FE error:  coordinates of two different vertices in Vxc are
        //     close to tol`"));
      }
    };

    //    auto comp = [](const coordinateValues& lhs, const coordinateValues&
    //    rhs){
    //      double tol = 1e-6;
    //
    //      double xdiff = lhs.xcoord - rhs.xcoord;
    //      double ydiff = lhs.ycoord - rhs.ycoord;
    //      double zdiff = lhs.zcoord - rhs.zcoord;
    //      if(xdiff < -tol)
    //        return true;
    //      if(xdiff > tol)
    //        return false;
    //
    //          if(ydiff < -tol)
    //            return true;
    //          if(ydiff > tol)
    //            return false;
    //
    //              if(zdiff < -tol)
    //                return true;
    //              if(zdiff > tol)
    //                return false;
    //
    //              AssertThrow(
    //                (std::abs(xdiff) > tol) || (std::abs(ydiff) > tol) ||
    //                (std::abs(zdiff) > tol), ExcMessage(
    //                  "DFT-FE error:  coordinates of two different vertices in
    //                  Vxc are close to tol`"));
    //    };


  } // namespace

  inverseDFT::inverseDFT(dftBase &       dft,
                         dftParameters & dftParams,
                         const MPI_Comm &mpi_comm_parent,
                         const MPI_Comm &mpi_comm_domain,
                         const MPI_Comm &mpi_comm_bandgroup,
                         const MPI_Comm &mpi_comm_interpool)
    : d_mpiComm_domain(mpi_comm_domain)
    , d_mpiComm_parent(mpi_comm_parent)
    , d_mpiComm_bandgroup(mpi_comm_bandgroup)
    , d_mpiComm_interpool(mpi_comm_interpool)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_dftParams(dftParams)
    , d_triaManagerVxc(mpi_comm_parent,
                       mpi_comm_domain,
                       mpi_comm_interpool,
                       mpi_comm_bandgroup,
                       dftParams)
    , d_dofHandlerTriaVxc()
    , d_gaussQuadVxc(2) // TODO this hard coded to Gauss 2x2x2 rule which is
                        // sufficient as the vxc mesh is taken to be linear FE.
                        // Read from params file for generality
    , d_rhoTargetTolForConstraints(
        1e-6) // // TODO this hard coded. Is this correct
  {

    d_dftBaseClass = &dft;

    d_dftMatrixFreeData = &(d_dftBaseClass->getMatrixFreeData());

    d_dftDensityDoFHandlerIndex = d_dftBaseClass->getDensityDofHandlerIndex();

    d_dftQuadIndex     = d_dftBaseClass->getDensityQuadratureId();
    d_gaussQuadAdjoint = &(d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex));

    d_numSpins       = 1 + d_dftParams.spinPolarized;
    d_kpointWeights  = d_dftBaseClass->getKPointWeights();
    d_numEigenValues = d_dftBaseClass->getNumEigenValues();
    d_numKPoints     = d_kpointWeights.size();
    // get the triangulation manager of the DFTClass
    d_dftTriaManager = d_dftBaseClass->getTriangulationManager();

    d_dofHandlerDFTClass =
      &d_dftMatrixFreeData->get_dof_handler(d_dftDensityDoFHandlerIndex);
    d_constraintDFTClass = d_dftBaseClass->getDensityConstraint();

    d_dftMatrixFreeDataElectro  = &(d_dftBaseClass->getMatrixFreeDataElectro());
    d_dftElectroDoFHandlerIndex = d_dftBaseClass->getElectroDofHandlerIndex();
    d_dofHandlerElectroDFTClass =
      &d_dftMatrixFreeDataElectro->get_dof_handler(d_dftElectroDoFHandlerIndex);
    d_dftElectroRhsQuadIndex = d_dftBaseClass->getElectroQuadratureRhsId();
    d_dftElectroAxQuadIndex  = d_dftBaseClass->getElectroQuadratureAxId();
  }

  inverseDFT::~inverseDFT()
  {
    // delete d_triaManagerVxcPtr;
  }
  void
  inverseDFT::createParentChildDofManager()
  {
    /*

        dftParameters dftParamsVxc(dftParams);

        dftParamsVxc.innerAtomBallRadius = dftParams.VxcInnerDomain;
        dftParamsVxc.meshSizeInnerBall = dftParams.VxcInnerMeshSize;
        dftParamsVxc.meshSizeOuterDomain = dftParams.meshSizeOuterDomain;
        dftParamsVxc.outerAtomBallRadius = dftParams.outerAtomBallRadius;
        dftParamsVxc.meshSizeOuterBall = dftParams.meshSizeOuterBall;

        triangulationManager dftTriaManagerVxc(d_mpiComm_parent,
            d_mpiComm_domain,
            d_mpiComm_interpool,
            d_mpiComm_bandgroup,
            1,
            dftParamsVxc);

        dftTriaManagerVxc.generateSerialUnmovedAndParallelMovedUnmovedMesh(
            atomLocations,
            :q);
            */
    const parallel::distributed::Triangulation<3> &parallelMeshUnmoved =
      d_dftTriaManager->getParallelMeshUnmoved();
    const parallel::distributed::Triangulation<3> &parallelMeshMoved =
      d_dftTriaManager->getParallelMeshMoved();
    /*
              d_triaManagerVxcPtr = new triangulationManagerVxc(d_mpiComm_parent,
                           d_mpiComm_domain,
                           d_mpiComm_interpool,
                           d_mpiComm_bandgroup,
                           dftParamsVxc,
                           dealii::parallel::distributed::Triangulation<3, 3>::default_setting); // set this to
                                                                                                 // dealii::parallel::distributed::Triangulation<3, 3>::no_automatic_repartitioning
                                                                                                 // If you want no repartitioning
    */



    // TODO does not assume periodic BCs.
    std::vector<std::vector<double>> atomLocations =
      d_dftBaseClass->getAtomLocationsCart();

    MPI_Barrier(d_mpiComm_domain);
    double meshStart = MPI_Wtime();
    //
    // @note This is compatible with only non-periodic boundary conditions as imageAtomLocations is not considered
    //
    d_triaManagerVxc.generateParallelUnmovedMeshVxc(
      d_dftParams.meshSizeInnerBall, // TODO Read value from params  (This is
                                     // the uniform mesh size for the vxc mesh)
      parallelMeshUnmoved,
      atomLocations,
      *d_dftTriaManager);

    MPI_Barrier(d_mpiComm_domain);
    double meshEnd = MPI_Wtime();
    // TODO this function has been commented out
    d_triaManagerVxc.generateParallelMovedMeshVxc(parallelMeshUnmoved,
                                                  parallelMeshMoved);

    //parallelMeshMoved = parallelMeshUnmoved;
    MPI_Barrier(d_mpiComm_domain);
    double                                         meshMoveEnd = MPI_Wtime();
    const parallel::distributed::Triangulation<3> &parallelMeshMovedVxc =
      d_triaManagerVxc.getParallelMovedMeshVxc();

    const parallel::distributed::Triangulation<3> &parallelMeshUnmovedVxc =
      d_triaManagerVxc.getParallelUnmovedMeshVxc();

    d_dofHandlerTriaVxc.reinit(parallelMeshMovedVxc);

    // TODO this hard coded to linear FE (which should be the usual case).
    // Read it from params file for generality
    const dealii::FE_Q<3> finite_elementVxc(1);

    d_dofHandlerTriaVxc.distribute_dofs(finite_elementVxc);

    dealii::IndexSet locallyRelevantDofsVxc;


    dealii::DoFTools::extract_locally_relevant_dofs(d_dofHandlerTriaVxc,
                                                    locallyRelevantDofsVxc);


    d_constraintMatrixVxc.clear();
    d_constraintMatrixVxc.reinit(locallyRelevantDofsVxc);
    dealii::DoFTools::make_hanging_node_constraints(d_dofHandlerTriaVxc,
                                                    d_constraintMatrixVxc);
    d_constraintMatrixVxc.close();

    typename MatrixFree<3>::AdditionalData additional_data;
    // comment this if using deal ii version 9
    // additional_data.mpi_communicator = d_mpiCommParent;
    additional_data.tasks_parallel_scheme =
      MatrixFree<3>::AdditionalData::partition_partition;

    //    additional_data.mapping_update_flags =
    //      update_values | update_JxW_values;

    additional_data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;

    d_matrixFreeDataVxc.reinit(dealii::MappingQ1<3, 3>(),
                               d_dofHandlerTriaVxc,
                               d_constraintMatrixVxc,
                               d_gaussQuadVxc,
                               additional_data);

    d_dofHandlerVxcIndex = 0;
    d_quadVxcIndex       = 0;

    /*
        unsigned int maxRelativeRefinement = 0;
        d_triaManagerVxc.computeMapBetweenParentAndChildMesh(
          parallelMeshMoved,
          parallelMeshMovedVxc,
          d_mapParentCellsToChild,
          d_mapParentCellToChildCellsIter,
          d_mapChildCellsToParent,
          maxRelativeRefinement);

        std::cout << " max relative refinement = " << maxRelativeRefinement << "\n";
    */
    MPI_Barrier(d_mpiComm_domain);
    double constraintsEnd = MPI_Wtime();
    /*
        d_inverseDftDoFManagerObjPtr = std::make_shared<TransferDataBetweenMeshesCompatiblePartitioning>(*d_dftMatrixFreeData,
                                         d_dftDensityDoFHandlerIndex,
                                         d_dftQuadIndex,
                                         d_matrixFreeDataVxc,
                                         d_dofHandlerVxcIndex,
                                         d_quadVxcIndex,
                                         d_mapParentCellsToChild,
                                         d_mapParentCellToChildCellsIter,
                                         d_mapChildCellsToParent,
                                         maxRelativeRefinement,
                                         d_dftParams.useDevice);
    */

    d_inverseDftDoFManagerObjPtr = std::make_shared<TransferDataBetweenMeshesIncompatiblePartitioning>(*d_dftMatrixFreeData,
                                                                                                       d_dftDensityDoFHandlerIndex,
                                                                                                       d_dftQuadIndex,
                                                                                                       d_matrixFreeDataVxc,
                                                                                                       d_dofHandlerVxcIndex,
                                                                                                       d_quadVxcIndex,
                                                                                                       d_mpiComm_domain,
                                                                                                       d_dftParams.useDevice);
    MPI_Barrier(d_mpiComm_domain);
    double createMapEnd = MPI_Wtime();

    std::cout << " time for mesh generation = " << meshEnd - meshStart
              << " move mesh = " << meshMoveEnd - meshEnd
              << " constraints = " << constraintsEnd - meshMoveEnd
              << " map gen = " << createMapEnd - constraintsEnd << "\n";
  }

  template <typename T>
  void
  inverseDFT::setInitialPot()
  {
    // get the eigen vectors from dftClass
    const std::vector<std::vector<T>> *eigenVectors =
      d_dftBaseClass->getEigenVectors();
    const std::vector<std::vector<double>> &eigenValues =
      d_dftBaseClass->getEigenValues();
    const double fermiEnergy = d_dftBaseClass->getFermiEnergy();



    unsigned int totalLocallyOwnedCellsVxc =
      d_matrixFreeDataVxc.n_physical_cells();

    const unsigned int numQuadPointsPerCellInVxc = d_gaussQuadVxc.size();



    int isSpinPolarized;
    if (d_dftParams.spinPolarized == 1)
      {
        isSpinPolarized = XC_POLARIZED;
      }
    else
      {
        isSpinPolarized = XC_UNPOLARIZED;
      }

    excWavefunctionBaseClass *excFunctionalPtr;


    xc_func_type *funcX = d_dftBaseClass->getfuncX();
    xc_func_type *funcC = d_dftBaseClass->getfuncC();

    excManager::createExcClassObj(d_dftParams.xc_id,
                                  isSpinPolarized,
                                  0.0,   // exx factor
                                  false, // scale exchange
                                  1.0,   // scale exchange factor
                                  false, // computeCorrelation
                                  funcX,
                                  funcC,
                                  excFunctionalPtr);

    std::vector<std::vector<double>> partialOccupancies(
      d_numKPoints, std::vector<double>(d_numSpins * d_numEigenValues, 0.0));

    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex)
      for (unsigned int kPoint = 0; kPoint < d_numKPoints; ++kPoint)
        for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
          {
            const double eigenValue =
              eigenValues[kPoint][d_numEigenValues * spinIndex + iWave];
            partialOccupancies[kPoint][d_numEigenValues * spinIndex + iWave] =
              dftUtils::getPartialOccupancy(eigenValue,
                                            fermiEnergy,
                                            C_kb,
                                            d_dftParams.TVal);

            if (d_dftParams.constraintMagnetization)
              {
                partialOccupancies[kPoint]
                                  [d_numEigenValues * spinIndex + iWave] = 1.0;
                if (spinIndex == 0)
                  {
                    if (eigenValue > fermiEnergy) // fermi energy up
                      partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                                 iWave] = 0.0;
                  }
                else if (spinIndex == 1)
                  {
                    if (eigenValue > fermiEnergy) // fermi energy down
                      partialOccupancies[kPoint][d_numEigenValues * spinIndex +
                                                 iWave] = 0.0;
                  }
              }
          }

    std::vector<distributedCPUVec<double>> vxcInitialGuess;

    std::vector<double> rhoInput;

    unsigned int locallyOwnedDofs =
      d_dofHandlerDFTClass->n_locally_owned_dofs();

    rhoInput.resize(d_numSpins * locallyOwnedDofs);
    std::fill(rhoInput.begin(), rhoInput.end(), 0.0);
    vxcInitialGuess.resize(d_numSpins);
    std::vector<dftfe::utils::MemoryStorage<dataTypes::number,
                                            dftfe::utils::MemorySpace::HOST>> initialPotValuesChildQuad;
    initialPotValuesChildQuad.resize(d_numSpins);
    d_vxcInitialChildNodes.resize(d_numSpins);
    std::vector<std::map<dealii::CellId, std::vector<double>>>
      initialPotValuesChildQuadDealiiMap;
    initialPotValuesChildQuadDealiiMap.resize(d_numSpins);
    const double spinFactor = (d_numSpins == 2) ? 1.0 : 2.0;
    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex)
      {
        for (unsigned int kPoint = 0; kPoint < d_numKPoints; ++kPoint)
          {
            for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
              {
                for (unsigned int iNode = 0; iNode < locallyOwnedDofs; iNode++)
                  {
                    T eigenVectorValue =
                      (*eigenVectors)[d_numSpins * kPoint + spinIndex]
                                     [iWave + iNode * d_numEigenValues];
                    rhoInput[iNode * d_numSpins + spinIndex] +=
                      spinFactor *
                      partialOccupancies[kPoint]
                                        [d_numEigenValues * spinIndex + iWave] *
                      d_kpointWeights[kPoint] *
                      realPart(complexConj(eigenVectorValue) *
                               eigenVectorValue);
                  }
              }
          }
      }

    // allocate storage for exchange potential
    std::vector<double> exchangePotentialVal(d_numSpins * locallyOwnedDofs,
                                             0.0);
    std::vector<double> corrPotentialVal(d_numSpins * locallyOwnedDofs, 0.0);
    std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

    std::map<VeffOutputDataAttributes, std::vector<double> *>
      outputDerExchangeEnergy;
    std::map<VeffOutputDataAttributes, std::vector<double> *>
      outputDerCorrEnergy;

    rhoData[rhoDataAttributes::values] = &rhoInput;

    outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
      &exchangePotentialVal;

    outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
      &corrPotentialVal;

    excFunctionalPtr->computeDensityBasedVxc(locallyOwnedDofs,
                                             rhoData,
                                             outputDerExchangeEnergy,
                                             outputDerCorrEnergy);

    dftUtils::constraintMatrixInfo constraintsMatrixDataInfoPsi;
    constraintsMatrixDataInfoPsi.initialize(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      *d_constraintDFTClass);


    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex)
      {
        vectorTools::createDealiiVector<double>(
          d_dftMatrixFreeData->get_vector_partitioner(
            d_dftDensityDoFHandlerIndex),
          1,
          vxcInitialGuess[spinIndex]);
        vxcInitialGuess[spinIndex] = 0.0;

        std::vector<dealii::types::global_dof_index>
          fullFlattenedArrayCellLocalProcIndexIdMapVxcParent;
        vectorTools::computeCellLocalIndexSetMap(
          vxcInitialGuess[spinIndex].get_partitioner(),
          *d_dftMatrixFreeData,
          d_dftDensityDoFHandlerIndex,
          1,
          fullFlattenedArrayCellLocalProcIndexIdMapVxcParent);

        constraintsMatrixDataInfoPsi.precomputeMaps(
          d_dftMatrixFreeData->get_vector_partitioner(
            d_dftDensityDoFHandlerIndex),
          vxcInitialGuess[spinIndex].get_partitioner(),
          1); // blockSize

        vectorTools::createDealiiVector<double>(
          d_matrixFreeDataVxc.get_vector_partitioner(d_dofHandlerVxcIndex),
          1,
          d_vxcInitialChildNodes[spinIndex]);
        d_vxcInitialChildNodes[spinIndex] = 0.0;

        initialPotValuesChildQuad[spinIndex].resize(totalLocallyOwnedCellsVxc *
                                                    numQuadPointsPerCellInVxc);
        std::fill(initialPotValuesChildQuad[spinIndex].begin(),
                  initialPotValuesChildQuad[spinIndex].end(),
                  0.0);
        for (unsigned int iNode = 0; iNode < locallyOwnedDofs; iNode++)
          {
            vxcInitialGuess[spinIndex].local_element(iNode) =
              exchangePotentialVal[iNode * d_numSpins + spinIndex];
            // + corrPotentialVal[iNode*d_numSpins + spinIndex];
          }

        //	  d_constraintDFTClass->distribute(vxcInitialGuess[spinIndex]);
        constraintsMatrixDataInfoPsi.distribute(vxcInitialGuess[spinIndex], 1);
        //    vxcInitialGuess[spinIndex].update_ghost_values();

        d_inverseDftDoFManagerObjPtr->interpolateMesh1DataToMesh2QuadPoints(
          vxcInitialGuess[spinIndex],
          1, // blockSize
          fullFlattenedArrayCellLocalProcIndexIdMapVxcParent,
          initialPotValuesChildQuad[spinIndex],
          true);

        dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerTriaVxc
                                                             .begin_active(),
                                                    endc =
                                                      d_dofHandlerTriaVxc.end();
        unsigned int iElem = 0;
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              std::vector<double> cellLevelQuadInput;
              cellLevelQuadInput.resize(numQuadPointsPerCellInVxc);
              for (unsigned int iQuad = 0; iQuad < numQuadPointsPerCellInVxc;
                   iQuad++)
                {
                  cellLevelQuadInput[iQuad] = initialPotValuesChildQuad
                    [spinIndex][iElem * numQuadPointsPerCellInVxc + iQuad];
                }
              initialPotValuesChildQuadDealiiMap[spinIndex][cell->id()] =
                cellLevelQuadInput;
              iElem++;
            }

        d_dftBaseClass->l2ProjectionQuadToNodal(
          d_matrixFreeDataVxc,
          d_constraintMatrixVxc,
          d_dofHandlerVxcIndex,
          d_quadVxcIndex,
          initialPotValuesChildQuadDealiiMap[spinIndex],
          d_vxcInitialChildNodes[spinIndex]);

        d_constraintMatrixVxc.set_zero(d_vxcInitialChildNodes[spinIndex]);
        //    d_vxcInitialChildNodes[spinIndex].zero_out_ghosts();
      }

    delete (excFunctionalPtr);

    distributedCPUVec<double> rhoInputTotal;
    vectorTools::createDealiiVector<double>(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      1,
      rhoInputTotal);
    rhoInputTotal = 0.0;

    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex)
      {
        for (unsigned int iNode = 0; iNode < locallyOwnedDofs; iNode++)
          {
            rhoInputTotal.local_element(iNode) +=
              rhoInput[iNode * d_numSpins + spinIndex];
          }
      }

    constraintsMatrixDataInfoPsi.distribute(rhoInputTotal, 1);
    //	d_constraintDFTClass->distribute(rhoInputTotal);
    //  rhoInputTotal.update_ghost_values();
    setAdjointBoundaryCondition(rhoInputTotal);
  }

  //  void inverseDFT::computeMomentOfInertia(const
  //  std::vector<std::vector<double>> &density,
  //                               const std::vector<double> &coordinates,
  //                               const std::vector<double> &JxWValues,
  //                               std::vector<double> &I_density)
  //  {
  //      I_density.resize(9);
  //      std::fill(I_density.begin(),I_density.end(),0.0);
  //      unsigned int numCoord = JxWValues.size();
  //      for( unsigned int i = 0 ; i < numCoord; i++)
  //      {
  //          double xcoord = coordinates[3*i +0];
  //          double ycoord = coordinates[3*i +1];
  //          double zcoord = coordinates[3*i +2];
  //
  //          //TODO how to handle spin density
  //          I_density[0] += density[iSpin]*(ycoord*ycoord +
  //          zcoord*zcoord)*JxWValues[i]; I_density[4] +=
  //          density[iSpin]*(xcoord*xcoord + zcoord*zcoord)*JxWValues[i];
  //          I_density[8] += density[iSpin]*(xcoord*xcoord +
  //          ycoord*ycoord)*JxWValues[i];
  //
  //          I_density[1] -= density[iSpin]*(xcoord*ycoord)*JxWValues[i];
  //          I_density[2] -= density[iSpin]*(xcoord*zcoord)*JxWValues[i]; //
  //          TODO check if this is a typo I_density[5] -=
  //          density[iSpin]*(ycoord*zcoord)*JxWValues[i];
  //      }
  //      MPI_Allreduce(MPI_IN_PLACE,
  //                    &I_density,
  //                    9,
  //                    MPI_DOUBLE,
  //                    MPI_SUM,
  //                    d_mpiComm_domain);
  //
  //      I_density[3] = I_density[1];
  //      I_density[6] = I_density[2];
  //      I_density[7] = I_density[5];
  //  }

  void
  inverseDFT::setInitialDensityFromGaussian(
    const std::vector<std::vector<double>> &rhoValuesFeSpin)
  {
    // Quadrature for AX multiplication will FEOrderElectro+1
    const dealii::Quadrature<3> &quadratureRuleParent =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
    const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();
    unsigned int totalLocallyOwnedCellsParent =
      d_dftMatrixFreeData->n_physical_cells();

    const unsigned int numTotalQuadraturePointsParent =
      totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;

    const dealii::DoFHandler<3> *dofHandlerParent =
      &d_dftMatrixFreeData->get_dof_handler(d_dftDensityDoFHandlerIndex);
    dealii::FEValues<3> fe_valuesParent(dofHandlerParent->get_fe(),
                                        quadratureRuleParent,
                                        dealii::update_JxW_values |
                                          dealii::update_quadrature_points);

    const unsigned int numberDofsPerElement =
      dofHandlerParent->get_fe().dofs_per_cell;

    //
    // resize data members
    //

    std::vector<double> quadJxWValues(numTotalQuadraturePointsParent, 0.0);
    std::vector<double> quadCoordinates(numTotalQuadraturePointsParent * 3,
                                        0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell                = dofHandlerParent->begin_active(),
      endc                = dofHandlerParent->end();
    unsigned int iElem    = 0;
    unsigned int quadPtNo = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_valuesParent.reinit(cell);

          for (unsigned int q_point = 0;
               q_point < numQuadraturePointsPerCellParent;
               ++q_point)
            {
              quadJxWValues[(iElem * numQuadraturePointsPerCellParent) +
                            q_point] = fe_valuesParent.JxW(q_point);
              dealii::Point<3, double> qPointVal =
                fe_valuesParent.quadrature_point(q_point);
              unsigned int qPointCoordIndex =
                ((iElem * numQuadraturePointsPerCellParent) + q_point) * 3;
              quadCoordinates[qPointCoordIndex + 0] = qPointVal[0];
              quadCoordinates[qPointCoordIndex + 1] = qPointVal[1];
              quadCoordinates[qPointCoordIndex + 2] = qPointVal[2];
            }
          iElem++;
        }

    std::vector<std::string> densityMatPrimaryFileNames;
    densityMatPrimaryFileNames.push_back(
      d_dftParams.densityMatPrimaryFileNameSpinUp);
    if (d_numSpins == 2)
      {
        densityMatPrimaryFileNames.push_back(
          d_dftParams.densityMatPrimaryFileNameSpinDown);
      }

    gaussianFunctionManager gaussianFuncManPrimaryObj(
      densityMatPrimaryFileNames,      // densityMatFilenames
      d_dftParams.gaussianAtomicCoord, // atomicCoordsFilename
      'A',                             // unit
      d_mpiComm_parent,
      d_mpiComm_domain);

    unsigned int gaussQuadIndex = 0;
    gaussianFuncManPrimaryObj.evaluateForQuad(
      &quadCoordinates[0],
      &quadJxWValues[0],
      numTotalQuadraturePointsParent,
      true,  // evalBasis,
      false, // evalBasisDerivatives,
      false, // evalBasisDoubleDerivatives,
      true,  // evalSMat,
      true,  // normalizeBasis,
      gaussQuadIndex,
      d_dftParams.sMatrixName);

    std::vector<std::vector<double>> rhoGaussianPrimary;
    rhoGaussianPrimary.resize(d_numSpins);
    for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
      {
        rhoGaussianPrimary[iSpin].resize(numTotalQuadraturePointsParent, 0.0);
        gaussianFuncManPrimaryObj.getRhoValue(gaussQuadIndex,
                                              iSpin,
                                              &rhoGaussianPrimary[iSpin][0]);
      }



    if (d_numSpins == 1)
      {
        std::vector<double> qpointCoord(3,0.0);
        std::vector<double> gradVal(3,0.0);
        d_sigmaGradRhoTarget.resize(totalLocallyOwnedCellsParent*numQuadraturePointsPerCellParent);

        for( unsigned int iCell = 0; iCell < totalLocallyOwnedCellsParent; iCell++)
          {
            for (unsigned int q_point = 0;
                 q_point < numQuadraturePointsPerCellParent;
                 ++q_point)
              {
                unsigned int qPointId = (iCell * numQuadraturePointsPerCellParent) + q_point;
                unsigned int qPointCoordIndex = qPointId*3;

                qpointCoord[0] = quadCoordinates[qPointCoordIndex + 0];
                qpointCoord[1] = quadCoordinates[qPointCoordIndex + 1];
                qpointCoord[2] = quadCoordinates[qPointCoordIndex + 2];

                gaussianFuncManPrimaryObj.getRhoGradient(&qpointCoord[0],0,gradVal);

                d_sigmaGradRhoTarget[qPointId] = 4.0*(gradVal[0]*gradVal[0] + gradVal[1]*gradVal[1] + gradVal[2]*gradVal[2]);
                if ( d_sigmaGradRhoTarget[qPointId] > 1e5)
                  {
                    std::cout<<" Large value of d_sigmaGradRhoTarget found at "<<qpointCoord[0]<<" "<<qpointCoord[1]<<" "<<qpointCoord[2]<<"\n";
                  }
              }
          }
      }
    if (d_numSpins == 2 )
      {
        std::vector<double> qpointCoord(3,0.0);
        std::vector<double> gradValSpinUp(3,0.0);
        std::vector<double> gradValSpinDown(3,0.0);
        d_sigmaGradRhoTarget.resize(3*totalLocallyOwnedCellsParent*numQuadraturePointsPerCellParent);

        for( unsigned int iCell = 0; iCell < totalLocallyOwnedCellsParent; iCell++)
          {
            for (unsigned int q_point = 0;
                 q_point < numQuadraturePointsPerCellParent;
                 ++q_point)
              {

                unsigned int qPointId = (iCell * numQuadraturePointsPerCellParent) + q_point;
                unsigned int qPointCoordIndex = qPointId*3;

                qpointCoord[0] = quadCoordinates[qPointCoordIndex + 0];
                qpointCoord[1] = quadCoordinates[qPointCoordIndex + 1];
                qpointCoord[2] = quadCoordinates[qPointCoordIndex + 2];

                gaussianFuncManPrimaryObj.getRhoGradient(&qpointCoord[0],0,gradValSpinUp);
                gaussianFuncManPrimaryObj.getRhoGradient(&qpointCoord[0],1,gradValSpinDown);

                d_sigmaGradRhoTarget[3*qPointId+0] = gradValSpinUp[0]*gradValSpinUp[0] +
                                                         gradValSpinUp[1]*gradValSpinUp[1] +
                                                         gradValSpinUp[2]*gradValSpinUp[2];

                d_sigmaGradRhoTarget[3*qPointId+1] = gradValSpinUp[0]*gradValSpinDown[0] +
                                                         gradValSpinUp[1]*gradValSpinDown[1] +
                                                         gradValSpinUp[2]*gradValSpinDown[2];

                d_sigmaGradRhoTarget[3*qPointId+2] = gradValSpinDown[0]*gradValSpinDown[0] +
                                                         gradValSpinDown[1]*gradValSpinDown[1] +
                                                         gradValSpinDown[2]*gradValSpinDown[2];
              }
          }
      }

    auto sigmaGradIt= std::max_element(d_sigmaGradRhoTarget.begin(),d_sigmaGradRhoTarget.end());
    double maxSigmaGradVal = *sigmaGradIt;

    MPI_Allreduce(
      MPI_IN_PLACE, &maxSigmaGradVal, 1, MPI_DOUBLE, MPI_MAX, d_mpiComm_domain);

    pcout<<" Max vlaue of sigmaGradVal = "<<maxSigmaGradVal<<"\n";
    std::vector<std::string> densityMatDFTFileNames;
    densityMatDFTFileNames.push_back(d_dftParams.densityMatDFTFileNameSpinUp);
    if (d_numSpins == 2)
      {
        densityMatDFTFileNames.push_back(
          d_dftParams.densityMatDFTFileNameSpinDown);
      }

    gaussianFunctionManager gaussianFuncManDFTObj(
      densityMatDFTFileNames,          // densityMatFilenames
      d_dftParams.gaussianAtomicCoord, // atomicCoordsFilename
      'A',                             // unit
      d_mpiComm_parent,
      d_mpiComm_domain);

    gaussianFuncManDFTObj.evaluateForQuad(&quadCoordinates[0],
                                          &quadJxWValues[0],
                                          numTotalQuadraturePointsParent,
                                          true,  // evalBasis,
                                          false, // evalBasisDerivatives,
                                          false, // evalBasisDoubleDerivatives,
                                          true,  // evalSMat,
                                          true,  // normalizeBasis,
                                          gaussQuadIndex,
                                          d_dftParams.sMatrixName);

    std::vector<std::vector<double>> rhoGaussianDFT;
    rhoGaussianDFT.resize(d_numSpins);
    for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
      {
        rhoGaussianDFT[iSpin].resize(numTotalQuadraturePointsParent, 0.0);
        gaussianFuncManDFTObj.getRhoValue(gaussQuadIndex,
                                          iSpin,
                                          &rhoGaussianDFT[iSpin][0]);
      }

    d_rhoTarget.resize(d_numSpins);
    for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
      {
        d_rhoTarget[iSpin].resize(totalLocallyOwnedCellsParent);
        cell  = dofHandlerParent->begin_active();
        iElem = 0;
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              d_rhoTarget[iSpin][iElem].resize(numQuadraturePointsPerCellParent,
                                               0.0);
              for (unsigned int iQuad = 0;
                   iQuad < numQuadraturePointsPerCellParent;
                   iQuad++)
                {
                  unsigned int index =
                    iElem * numQuadraturePointsPerCellParent + iQuad;
                  d_rhoTarget[iSpin][iElem][iQuad] =
                    rhoGaussianPrimary[iSpin][index] -
                    rhoGaussianDFT[iSpin][index] +
                    rhoValuesFeSpin[iSpin][index];
                }
              iElem++;
            }
      }



    std::map<dealii::CellId, std::vector<double>> rhoGaussianPrim;
    std::map<dealii::CellId, std::vector<double>> rhoGaussianSecond;
    std::map<dealii::CellId, std::vector<double>> rhoFeLda;
    std::map<dealii::CellId, std::vector<double>> rhoDiff;

    cell                  = dofHandlerParent->begin_active();
    iElem                 = 0;
    double rhoSumGaussian = 0.0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellId = cell->id();
          std::vector<double>  cellLevelRGP, cellLevelRGS, cellLevelRFL,
            cellLevelRhoDiff;
          cellLevelRGP.resize(numQuadraturePointsPerCellParent);
          cellLevelRGS.resize(numQuadraturePointsPerCellParent);
          cellLevelRFL.resize(numQuadraturePointsPerCellParent);
          cellLevelRhoDiff.resize(numQuadraturePointsPerCellParent);
          for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
               iQuad++)
            {
              unsigned int index =
                iElem * numQuadraturePointsPerCellParent + iQuad;
              cellLevelRGP[iQuad] = rhoGaussianPrimary[0][index];
              cellLevelRGS[iQuad] = rhoGaussianDFT[0][index];
              cellLevelRFL[iQuad] = rhoValuesFeSpin[0][index];
              cellLevelRhoDiff[iQuad] =
                rhoGaussianDFT[0][index] - rhoValuesFeSpin[0][index];
              if (d_rhoTarget[0][iElem][iQuad] < 0.0)
                {
                  unsigned int qPointCoordIndex =
                    ((iElem * numQuadraturePointsPerCellParent) + iQuad) * 3;
                  std::cout << " qPoint = ("<<quadCoordinates[qPointCoordIndex + 0]<<","<<quadCoordinates[qPointCoordIndex + 1]<<","<<quadCoordinates[qPointCoordIndex + 2]<<") RHO IS NEGATIVE!!!!!!!!!!\n";
                  std::cout<<"primary = "<<rhoGaussianPrimary[0][index]<<" secondary = "<<
                    rhoGaussianDFT[0][index]<< " Fe = "<< rhoValuesFeSpin[0][index]<<"\n";
                }
              rhoSumGaussian +=
                d_rhoTarget[0][iElem][iQuad] *
                quadJxWValues[(iElem * numQuadraturePointsPerCellParent) +
                              iQuad];
            }
          rhoGaussianPrim[cellId]   = cellLevelRGP;
          rhoGaussianSecond[cellId] = cellLevelRGS;
          rhoFeLda[cellId]          = cellLevelRFL;
          rhoDiff[cellId]           = cellLevelRhoDiff;
          iElem++;
        }
    MPI_Allreduce(
      MPI_IN_PLACE, &rhoSumGaussian, 1, MPI_DOUBLE, MPI_SUM, d_mpiComm_domain);
    pcout << " Sum of all rho target = " << rhoSumGaussian << "\n";

    distributedCPUVec<double> rhoGPVec, rhoGSVec, rhoFLVec, rhoDiffVec;

    vectorTools::createDealiiVector<double>(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      1,
      rhoGPVec);
    rhoGPVec = 0.0;
    rhoGSVec.reinit(rhoGPVec);
    rhoFLVec.reinit(rhoGPVec);
    rhoDiffVec.reinit(rhoGPVec);

    d_dftBaseClass->l2ProjectionQuadToNodal(*d_dftMatrixFreeData,
                                            *d_constraintDFTClass,
                                            d_dftDensityDoFHandlerIndex,
                                            d_dftQuadIndex,
                                            rhoGaussianPrim,
                                            rhoGPVec);

    d_constraintDFTClass->distribute(rhoGPVec);
    rhoGPVec.update_ghost_values();

    d_dftBaseClass->l2ProjectionQuadToNodal(*d_dftMatrixFreeData,
                                            *d_constraintDFTClass,
                                            d_dftDensityDoFHandlerIndex,
                                            d_dftQuadIndex,
                                            rhoGaussianSecond,
                                            rhoGSVec);

    d_constraintDFTClass->distribute(rhoGSVec);
    rhoGSVec.update_ghost_values();

    d_dftBaseClass->l2ProjectionQuadToNodal(*d_dftMatrixFreeData,
                                            *d_constraintDFTClass,
                                            d_dftDensityDoFHandlerIndex,
                                            d_dftQuadIndex,
                                            rhoFeLda,
                                            rhoFLVec);

    d_constraintDFTClass->distribute(rhoFLVec);
    rhoFLVec.update_ghost_values();

    d_dftBaseClass->l2ProjectionQuadToNodal(*d_dftMatrixFreeData,
                                            *d_constraintDFTClass,
                                            d_dftDensityDoFHandlerIndex,
                                            d_dftQuadIndex,
                                            rhoDiff,
                                            rhoDiffVec);

    d_constraintDFTClass->distribute(rhoDiffVec);
    rhoDiffVec.update_ghost_values();
    /*
       dealii::DataOut<3, dealii::DoFHandler<3>> data_out_rho;

       data_out_rho.attach_dof_handler(*dofHandlerParent);

       std::string outputVecName1 = "rho Gaussian primary";
       std::string outputVecName2 = "rho Gaussian secondary";
       std::string outputVecName3 = "rho fe lda";
       std::string outputVecName4 = "rho diff";
       data_out_rho.add_data_vector(rhoGPVec,outputVecName1);
       data_out_rho.add_data_vector(rhoGSVec,outputVecName2);
       data_out_rho.add_data_vector(rhoFLVec,outputVecName3);
       data_out_rho.add_data_vector(rhoDiffVec,outputVecName4);

       data_out_rho.build_patches();
       data_out_rho.write_vtu_with_pvtu_record("./", "inputRhoData",
       0,d_mpiComm_domain,2, 4);
    */
  }

  template <typename T>
  void
  inverseDFT::setInitialPotL2Proj()
  {
    unsigned int totalLocallyOwnedCellsVxc =
      d_matrixFreeDataVxc.n_physical_cells();

    const unsigned int numQuadPointsPerCellInVxc = d_gaussQuadVxc.size();


    double spinFactor = (d_dftParams.spinPolarized == 1) ? 1.0 : 2.0;

    int isSpinPolarized;
    if (d_dftParams.spinPolarized == 1)
      {
        isSpinPolarized = XC_POLARIZED;
      }
    else
      {
        isSpinPolarized = XC_UNPOLARIZED;
      }

    excWavefunctionBaseClass *excFunctionalPtrLDA, *excFunctionalPtrGGA ;
    xc_func_type              funcX_LDA, funcC_LDA;
    xc_func_type              funcX_GGA, funcC_GGA;
    // xc_func_type * funcX = d_dftBaseClass->getfuncX();
    // xc_func_type * funcC = d_dftBaseClass->getfuncC();
    excManager::createExcClassObj(d_dftParams.xc_id,
                                  // isSpinPolarized,
                                  (d_dftParams.spinPolarized == 1) ? true :
                                                                     false,
                                  0.0,   // exx factor
                                  false, // scale exchange
                                  1.0,   // scale exchange factor
                                  true,  // computeCorrelation
                                  &funcX_LDA,
                                  &funcC_LDA,
                                  excFunctionalPtrLDA);

    excManager::createExcClassObj(6, // X - LB , C = PBE
                                     // isSpinPolarized,
                                  (d_dftParams.spinPolarized == 1) ? true :
                                                                     false,
                                  0.0,   // exx factor
                                  false, // scale exchange
                                  1.0,   // scale exchange factor
                                  true,  // computeCorrelation
                                  &funcX_GGA,
                                  &funcC_GGA,
                                  excFunctionalPtrGGA);

    std::vector<distributedCPUVec<double>> vxcInitialGuess;


    unsigned int locallyOwnedDofs =
      d_dofHandlerDFTClass->n_locally_owned_dofs();

    vxcInitialGuess.resize(d_numSpins);
    std::vector<dftfe::utils::MemoryStorage<dataTypes::number,
                                            dftfe::utils::MemorySpace::HOST>> initialPotValuesChildQuad;
    initialPotValuesChildQuad.resize(d_numSpins);
    d_vxcInitialChildNodes.resize(d_numSpins);
    std::vector<std::map<dealii::CellId, std::vector<double>>>
      initialPotValuesChildQuadDealiiMap;
    initialPotValuesChildQuadDealiiMap.resize(d_numSpins);

    std::vector<std::map<dealii::CellId, std::vector<double>>>
      initialPotValuesParentQuadData;
    initialPotValuesParentQuadData.resize(d_numSpins);

    std::vector<std::map<dealii::CellId, std::vector<double>>>
      exactPotValuesParentQuadData;
    exactPotValuesParentQuadData.resize(d_numSpins);


    d_targetPotValuesParentQuadData.resize(d_numSpins);

    unsigned int totalOwnedCellsPsi = d_dftMatrixFreeData->n_physical_cells();

    const dealii::Quadrature<3> &quadratureRulePsi =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);

    unsigned int        numQuadPointsPerPsiCell = quadratureRulePsi.size();
    std::vector<double> rhoSpinFlattened(d_numSpins * totalOwnedCellsPsi *
                                           numQuadPointsPerPsiCell,
                                         0.0);
    for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
      {
        for (unsigned int iCell = 0; iCell < totalOwnedCellsPsi; iCell++)
          {
            for (unsigned int iQuad = 0; iQuad < numQuadPointsPerPsiCell;
                 iQuad++)
              {
                rhoSpinFlattened[(iCell * numQuadPointsPerPsiCell + iQuad) *
                                   d_numSpins +
                                 iSpin] =
                  spinFactor * d_rhoTarget[iSpin][iCell][iQuad];
              }
          }
      }

    distributedCPUVec<double> rhoInputTotal;
    vectorTools::createDealiiVector<double>(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      1,
      rhoInputTotal);
    rhoInputTotal = 0.0;


    unsigned int totalLocallyOwnedCellsPsi =
      d_dftMatrixFreeData->n_physical_cells();

    unsigned int numLocallyOwnedDofsPsi =
      d_dofHandlerDFTClass->n_locally_owned_dofs();
    unsigned int numDofsPerCellPsi =
      d_dofHandlerDFTClass->get_fe().dofs_per_cell;

    std::map<dealii::CellId, std::vector<double>> rhoValues;

    typename DoFHandler<3>::active_cell_iterator cellPsiPtr =
      d_dofHandlerDFTClass->begin_active();
    typename DoFHandler<3>::active_cell_iterator endcellPsiPtr =
      d_dofHandlerDFTClass->end();

    unsigned int iElem      = 0;
    unsigned int spinIndex1 = 0;
    unsigned int spinIndex2 = 0;
    if (d_numSpins == 2)
      {
        spinIndex2 = 1;
      }
    for (; cellPsiPtr != endcellPsiPtr; ++cellPsiPtr)
      {
        if (cellPsiPtr->is_locally_owned())
          {
            const dealii::CellId cellId = cellPsiPtr->id();
            std::vector<double>  cellLevelRho;
            cellLevelRho.resize(numQuadPointsPerPsiCell);
            for (unsigned int iQuad = 0; iQuad < numQuadPointsPerPsiCell;
                 iQuad++)
              {
                cellLevelRho[iQuad] = d_rhoTarget[spinIndex1][iElem][iQuad] +
                                      d_rhoTarget[spinIndex2][iElem][iQuad];
              }
            rhoValues[cellId] = cellLevelRho;
            iElem++;
          }
      }

    std::map<dealii::CellId, std::vector<double>> hartreeQuadData;
    computeHartreePotOnParentQuad(hartreeQuadData);

    // allocate storage for exchange potential
    std::vector<double> exchangePotentialVal(d_numSpins * totalOwnedCellsPsi *
                                               numQuadPointsPerPsiCell,
                                             0.0);

    std::vector<double> exchangePotentialValDummy(d_numSpins * totalOwnedCellsPsi *
                                                    numQuadPointsPerPsiCell,
                                                  0.0);

    std::vector<double> corrPotentialVal(d_numSpins * totalOwnedCellsPsi *
                                           numQuadPointsPerPsiCell,
                                         0.0);

    std::vector<double> corrPotentialValDummy(d_numSpins * totalOwnedCellsPsi *
                                                numQuadPointsPerPsiCell,
                                              0.0);

    std::vector<double> derExchEnergyWithSigmaValDummy(d_sigmaGradRhoTarget.size(),0.0);

    std::vector<double> derCorrEnergyWithSigmaValDummy(d_sigmaGradRhoTarget.size(),0.0);

    std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

    std::map<VeffOutputDataAttributes, std::vector<double> *>
      outputDerExchangeEnergy;


    std::map<VeffOutputDataAttributes, std::vector<double> *>
      outputDerExchangeEnergyDummy;

    std::map<VeffOutputDataAttributes, std::vector<double> *>
      outputDerCorrEnergy, outputDerCorrEnergyDummy;

    rhoData[rhoDataAttributes::values] = &rhoSpinFlattened;

    rhoData[rhoDataAttributes::sigmaGradValue] =
      &d_sigmaGradRhoTarget;

    outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
      &exchangePotentialVal;

    outputDerExchangeEnergy[VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
      &derExchEnergyWithSigmaValDummy;


    outputDerCorrEnergyDummy[VeffOutputDataAttributes::derEnergyWithDensity] =
      &corrPotentialValDummy;

    outputDerCorrEnergyDummy[VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
      &derCorrEnergyWithSigmaValDummy;

    outputDerExchangeEnergyDummy[VeffOutputDataAttributes::derEnergyWithDensity] =
      &exchangePotentialValDummy;

    outputDerCorrEnergy[VeffOutputDataAttributes::derEnergyWithDensity] =
      &corrPotentialVal;

    excFunctionalPtrGGA->computeDensityBasedVxc(totalOwnedCellsPsi *
                                                  numQuadPointsPerPsiCell,
                                                rhoData,
                                                outputDerExchangeEnergy,
                                                outputDerCorrEnergyDummy);


    excFunctionalPtrLDA->computeDensityBasedVxc(totalOwnedCellsPsi *
                                                  numQuadPointsPerPsiCell,
                                                rhoData,
                                                outputDerExchangeEnergyDummy,
                                                outputDerCorrEnergy);

    dftUtils::constraintMatrixInfo constraintsMatrixDataInfoPsi;
    constraintsMatrixDataInfoPsi.initialize(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      *d_constraintDFTClass);

    unsigned int numElectrons = d_dftBaseClass->getNumElectrons();

    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex)
      {
        vectorTools::createDealiiVector<double>(
          d_dftMatrixFreeData->get_vector_partitioner(
            d_dftDensityDoFHandlerIndex),
          1,
          vxcInitialGuess[spinIndex]);
        vxcInitialGuess[spinIndex] = 0.0;

        constraintsMatrixDataInfoPsi.precomputeMaps(
          d_dftMatrixFreeData->get_vector_partitioner(
            d_dftDensityDoFHandlerIndex),
          vxcInitialGuess[spinIndex].get_partitioner(),
          1); // blockSize


        vectorTools::createDealiiVector<double>(
          d_matrixFreeDataVxc.get_vector_partitioner(d_dofHandlerVxcIndex),
          1,
          d_vxcInitialChildNodes[spinIndex]);
        d_vxcInitialChildNodes[spinIndex] = 0.0;

        d_targetPotValuesParentQuadData[spinIndex].resize(totalOwnedCellsPsi);

        dealii::DoFHandler<3>::active_cell_iterator
          cellPsi             = d_dofHandlerDFTClass->begin_active(),
          endcPsi             = d_dofHandlerDFTClass->end();
        unsigned int iElemPsi = 0;
        for (; cellPsi != endcPsi; ++cellPsi)
          if (cellPsi->is_locally_owned())
            {
              d_targetPotValuesParentQuadData[spinIndex][iElemPsi].resize(
                numQuadPointsPerPsiCell, 0.0);
              std::vector<double> cellLevelQuadInput,
                cellLevelQuadInputExactVxc;
              cellLevelQuadInput.resize(numQuadPointsPerPsiCell);
              cellLevelQuadInputExactVxc.resize(numQuadPointsPerPsiCell);

              const std::vector<double> &hartreeCellLevelQuad =
                hartreeQuadData.find(cellPsi->id())->second;
              for (unsigned int iQuad = 0; iQuad < numQuadPointsPerPsiCell;
                   iQuad++)
                {
                  double tau = d_dftParams.inverseTauForVxBc;
                  double preFactor =
                    rhoSpinFlattened[(iElemPsi * numQuadPointsPerPsiCell +
                                      iQuad) *
                                       d_numSpins +
                                     spinIndex] /
                    (rhoSpinFlattened[(iElemPsi * numQuadPointsPerPsiCell +
                                       iQuad) *
                                        d_numSpins +
                                      spinIndex] +
                     tau);
                  double exchangeValue = exchangePotentialVal
                    [(iElemPsi * numQuadPointsPerPsiCell + iQuad) * d_numSpins +
                     spinIndex];
                  double exchangeCorrValue =
                    exchangePotentialVal[(iElemPsi * numQuadPointsPerPsiCell +
                                          iQuad) *
                                           d_numSpins +
                                         spinIndex] +
                    corrPotentialVal[(iElemPsi * numQuadPointsPerPsiCell +
                                      iQuad) *
                                       d_numSpins +
                                     spinIndex];

                  cellLevelQuadInput[iQuad] =
                    ((1.0 - preFactor) * exchangeValue +
                     (preFactor)*exchangeCorrValue);

                  if (d_dftParams.fermiAmaldiBC)
                    {
                      double tauBC = d_dftParams.inverseTauForFABC;
                      double preFactorBC =
                        rhoSpinFlattened[(iElemPsi * numQuadPointsPerPsiCell +
                                          iQuad) *
                                           d_numSpins +
                                         spinIndex] /
                        (rhoSpinFlattened[(iElemPsi * numQuadPointsPerPsiCell +
                                           iQuad) *
                                            d_numSpins +
                                          spinIndex] +
                         tauBC);

                      cellLevelQuadInput[iQuad] =
                        preFactorBC * cellLevelQuadInput[iQuad] +
                        (1.0 - preFactorBC) * (-1.0 / numElectrons) *
                          hartreeCellLevelQuad[iQuad];
                    }
                  d_targetPotValuesParentQuadData[spinIndex][iElemPsi][iQuad] =
                    exchangePotentialVal[(iElemPsi * numQuadPointsPerPsiCell +
                                          iQuad) *
                                           d_numSpins +
                                         spinIndex] +
                    corrPotentialVal[(iElemPsi * numQuadPointsPerPsiCell +
                                      iQuad) *
                                       d_numSpins +
                                     spinIndex];
                  cellLevelQuadInputExactVxc[iQuad] =
                    exchangePotentialVal[(iElemPsi * numQuadPointsPerPsiCell +
                                          iQuad) *
                                           d_numSpins +
                                         spinIndex] +
                    corrPotentialVal[(iElemPsi * numQuadPointsPerPsiCell +
                                      iQuad) *
                                       d_numSpins +
                                     spinIndex];
                }
              initialPotValuesParentQuadData[spinIndex][cellPsi->id()] =
                cellLevelQuadInput;
              exactPotValuesParentQuadData[spinIndex][cellPsi->id()] =
                cellLevelQuadInputExactVxc;
              iElemPsi++;
            }

        d_dftBaseClass->l2ProjectionQuadToNodal(
          *d_dftMatrixFreeData,
          *d_constraintDFTClass,
          d_dftDensityDoFHandlerIndex,
          d_dftQuadIndex,
          initialPotValuesParentQuadData[spinIndex],
          vxcInitialGuess[spinIndex]);

        constraintsMatrixDataInfoPsi.distribute(vxcInitialGuess[spinIndex], 1);

        distributedCPUVec<double> exactVxcTestParent;
        exactVxcTestParent.reinit(vxcInitialGuess[spinIndex]);

        d_dftBaseClass->l2ProjectionQuadToNodal(
          *d_dftMatrixFreeData,
          *d_constraintDFTClass,
          d_dftDensityDoFHandlerIndex,
          d_dftQuadIndex,
          exactPotValuesParentQuadData[spinIndex],
          exactVxcTestParent);

        constraintsMatrixDataInfoPsi.distribute(exactVxcTestParent, 1);

        //        d_constraintDFTClass->distribute(vxcInitialGuess[spinIndex]);
        //        vxcInitialGuess[spinIndex].update_ghost_values();

        std::vector<dealii::types::global_dof_index>
          fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent;
        vectorTools::computeCellLocalIndexSetMap(
          vxcInitialGuess[spinIndex].get_partitioner(),
          *d_dftMatrixFreeData,
          d_dftDensityDoFHandlerIndex,
          1,
          fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent);

        std::vector<dftfe::utils::MemoryStorage<dataTypes::number,
                                                dftfe::utils::MemorySpace::HOST>> exactPotValuesChildQuad;
        exactPotValuesChildQuad.resize(d_numSpins);
        std::vector<std::map<dealii::CellId, std::vector<double>>>
          exactPotValuesChildQuadDealiiMap;
        exactPotValuesChildQuadDealiiMap.resize(d_numSpins);
        exactPotValuesChildQuad[spinIndex].resize(totalLocallyOwnedCellsVxc *
                                                  numQuadPointsPerCellInVxc);
        std::fill(exactPotValuesChildQuad[spinIndex].begin(),
                  exactPotValuesChildQuad[spinIndex].end(),
                  0.0);
        d_inverseDftDoFManagerObjPtr->interpolateMesh1DataToMesh2QuadPoints(
          exactVxcTestParent,
          1, // blockSize
          fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent,
          exactPotValuesChildQuad[spinIndex],
          true);


        initialPotValuesChildQuad[spinIndex].resize(totalLocallyOwnedCellsVxc *
                                                    numQuadPointsPerCellInVxc);
        std::fill(initialPotValuesChildQuad[spinIndex].begin(),
                  initialPotValuesChildQuad[spinIndex].end(),
                  0.0);

        d_inverseDftDoFManagerObjPtr->interpolateMesh1DataToMesh2QuadPoints(
          vxcInitialGuess[spinIndex],
          1, // blockSize
          fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent,
          initialPotValuesChildQuad[spinIndex],
          true);

        dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerTriaVxc
                                                             .begin_active(),
                                                    endc =
                                                      d_dofHandlerTriaVxc.end();
        unsigned int iElem = 0;
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              std::vector<double> cellLevelQuadInput;
              cellLevelQuadInput.resize(numQuadPointsPerCellInVxc);

              std::vector<double> cellLevelQuadInputExact;
              cellLevelQuadInputExact.resize(numQuadPointsPerCellInVxc);
              for (unsigned int iQuad = 0; iQuad < numQuadPointsPerCellInVxc;
                   iQuad++)
                {
                  cellLevelQuadInput[iQuad] = initialPotValuesChildQuad
                    [spinIndex][iElem * numQuadPointsPerCellInVxc + iQuad];
                  cellLevelQuadInputExact[iQuad] =
                    exactPotValuesChildQuad[spinIndex]
                                           [iElem * numQuadPointsPerCellInVxc +
                                            iQuad];
                }
              initialPotValuesChildQuadDealiiMap[spinIndex][cell->id()] =
                cellLevelQuadInput;
              exactPotValuesChildQuadDealiiMap[spinIndex][cell->id()] =
                cellLevelQuadInputExact;
              iElem++;
            }

        d_dftBaseClass->l2ProjectionQuadToNodal(
          d_matrixFreeDataVxc,
          d_constraintMatrixVxc,
          d_dofHandlerVxcIndex,
          d_quadVxcIndex,
          initialPotValuesChildQuadDealiiMap[spinIndex],
          d_vxcInitialChildNodes[spinIndex]);

        d_constraintMatrixVxc.set_zero(d_vxcInitialChildNodes[spinIndex]);
        //    d_vxcInitialChildNodes[spinIndex].zero_out_ghosts();

        distributedCPUVec<double> exactVxcTestChild;
        exactVxcTestChild.reinit(d_vxcInitialChildNodes[spinIndex]);
        d_dftBaseClass->l2ProjectionQuadToNodal(
          d_matrixFreeDataVxc,
          d_constraintMatrixVxc,
          d_dofHandlerVxcIndex,
          d_quadVxcIndex,
          exactPotValuesChildQuadDealiiMap[spinIndex],
          exactVxcTestChild);

        d_constraintMatrixVxc.distribute(exactVxcTestChild);
        exactVxcTestChild.update_ghost_values();

        /*
              pcout<<"writing exact vxc output\n";
              dealii::DataOut<3, dealii::DoFHandler<3>> data_out_vxc;

              data_out_vxc.attach_dof_handler(d_dofHandlerTriaVxc);

              std::string outputVecName1 = "exact vxc";
              data_out_vxc.add_data_vector(exactVxcTestChild, outputVecName1);

              data_out_vxc.build_patches();
              data_out_vxc.write_vtu_with_pvtu_record("./", "exactVxc",
           0,d_mpiComm_domain ,2, 4);
      */


        pcout<<"writing initial vxc guess\n";
        dealii::DataOut<3, dealii::DoFHandler<3>> data_out_vxc;

        data_out_vxc.attach_dof_handler(*d_dofHandlerDFTClass);

        std::string outputVecName1 = "initial vxc";
        data_out_vxc.add_data_vector(vxcInitialGuess[0], outputVecName1);

        data_out_vxc.build_patches();
        data_out_vxc.write_vtu_with_pvtu_record("./", "initiaVxcGuess",
                                                0,d_mpiComm_domain ,2, 4);


      }

    delete (excFunctionalPtrLDA);
    delete (excFunctionalPtrGGA);

    d_dftBaseClass->l2ProjectionQuadToNodal(*d_dftMatrixFreeData,
                                            *d_constraintDFTClass,
                                            d_dftDensityDoFHandlerIndex,
                                            d_dftQuadIndex,
                                            rhoValues,
                                            rhoInputTotal);

    constraintsMatrixDataInfoPsi.distribute(rhoInputTotal, 1);

    d_constraintDFTClass->distribute(rhoInputTotal);
    rhoInputTotal.update_ghost_values();

    setAdjointBoundaryCondition(rhoInputTotal);
  }

  void
  inverseDFT::setAdjointBoundaryCondition(distributedCPUVec<double> &rhoTarget)
  {
    /*
    pcout<<"writing rho target";
    dealii::DataOut<3, dealii::DoFHandler<3>> data_out;

    data_out.attach_dof_handler(*d_dofHandlerDFTClass);

    std::string outputVecName = "rhoTarget";
    data_out.add_data_vector(rhoTarget, outputVecName);

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record("./", "rhoTarget", 0,d_mpiComm_domain
    ,2, 4);
*/
    dealii::IndexSet localSet = d_dofHandlerDFTClass->locally_owned_dofs();

    dealii::IndexSet locallyRelevantDofsAdjoint;


    dealii::DoFTools::extract_locally_relevant_dofs(*d_dofHandlerDFTClass,
                                                    locallyRelevantDofsAdjoint);

    distributedCPUVec<double> rhoTargetFullVector;
    rhoTargetFullVector.reinit(localSet,
                               locallyRelevantDofsAdjoint,
                               d_mpiComm_domain);

    unsigned int locallyOwnedDofs =
      d_dofHandlerDFTClass->n_locally_owned_dofs();
    for (unsigned int iNode = 0; iNode < locallyOwnedDofs; iNode++)
      {
        rhoTargetFullVector.local_element(iNode) =
          rhoTarget.local_element(iNode);
      }
    rhoTargetFullVector.update_ghost_values();

    d_constraintMatrixAdjoint.clear();
    d_constraintMatrixAdjoint.reinit(locallyRelevantDofsAdjoint);
    dealii::DoFTools::make_hanging_node_constraints(*d_dofHandlerDFTClass,
                                                    d_constraintMatrixAdjoint);

    const unsigned int dofs_per_cell =
      d_dofHandlerDFTClass->get_fe().dofs_per_cell;
    const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const unsigned int dofs_per_face =
      d_dofHandlerDFTClass->get_fe().dofs_per_face;

    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      dofs_per_cell);
    std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
      dofs_per_face);

    std::vector<bool> dofs_touched(d_dofHandlerDFTClass->n_dofs(), false);

    dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerDFTClass
                                                         ->begin_active(),
                                                endc =
                                                  d_dofHandlerDFTClass->end();
    unsigned int adjointConstraiedNodes = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          cell->get_dof_indices(cellGlobalDofIndices);
          for (unsigned int iDof = 0; iDof < dofs_per_cell; iDof++)
            {
              const dealii::types::global_dof_index nodeId =
                cellGlobalDofIndices[iDof];
              if (dofs_touched[nodeId])
                continue;
              dofs_touched[nodeId] = true;
              if (rhoTargetFullVector[nodeId] < d_rhoTargetTolForConstraints)
                {
                  if (!d_constraintMatrixAdjoint.is_constrained(nodeId))
                    {
                      if (rhoTargetFullVector.in_local_range(nodeId))
                        {
                          adjointConstraiedNodes++;
                        }
                      d_constraintMatrixAdjoint.add_line(nodeId);
                      d_constraintMatrixAdjoint.set_inhomogeneity(nodeId, 0.0);
                    } // non-hanging node check
                }
            }
          //          for (unsigned int iFace = 0; iFace < faces_per_cell;
          //          ++iFace)
          //            {
          //              const unsigned int boundaryId =
          //              cell->face(iFace)->boundary_id(); if (boundaryId == 0)
          //                {
          //                  cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
          //                  for (unsigned int iFaceDof = 0; iFaceDof <
          //                  dofs_per_face;
          //                       ++iFaceDof)
          //                    {
          //                      const dealii::types::global_dof_index nodeId =
          //                        iFaceGlobalDofIndices[iFaceDof];
          //                      if (dofs_touched[nodeId])
          //                        continue;
          //                      dofs_touched[nodeId] = true;
          //                      if
          //                      (!d_constraintMatrixAdjoint.is_constrained(nodeId))
          //                        {
          //                          d_constraintMatrixAdjoint.add_line(nodeId);
          //                          d_constraintMatrixAdjoint.set_inhomogeneity(nodeId,
          //                          0.0);
          //                        } // non-hanging node check
          //                    }     // Face dof loop
          //                }         // non-periodic boundary id
          //            }
        }

    d_constraintMatrixAdjoint.close();

    MPI_Allreduce(MPI_IN_PLACE,
                  &adjointConstraiedNodes,
                  1,
                  dftfe::dataTypes::mpi_type_id(&adjointConstraiedNodes),
                  MPI_SUM,
                  d_mpiComm_domain);

    pcout << " no of constrained adjoint from manual addition  = "
          << adjointConstraiedNodes << "\n";

    std::cout << " num adjoint constraints iProc = " << this_mpi_process
              << "size = " << d_constraintMatrixAdjoint.n_constraints() << "\n";

    IndexSet locally_active_dofs;

    DoFTools::extract_locally_active_dofs(*d_dofHandlerDFTClass,
                                          locally_active_dofs);

    bool consistentConstraints =
      d_constraintMatrixAdjoint.is_consistent_in_parallel(
        Utilities::MPI::all_gather(d_mpiComm_domain,
                                   d_dofHandlerDFTClass->locally_owned_dofs()),
        locally_active_dofs,
        d_mpiComm_domain,
        true);

    pcout << " Are the constraints consistent across partitoners = "
          << consistentConstraints << "\n";

    typename MatrixFree<3>::AdditionalData additional_data;
    // comment this if using deal ii version 9
    // additional_data.mpi_communicator = d_mpiCommParent;
    additional_data.tasks_parallel_scheme =
      MatrixFree<3>::AdditionalData::partition_partition;

    additional_data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;

    std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;
    matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerDFTClass);
    matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerDFTClass);

    std::vector<const dealii::AffineConstraints<double> *>
      constraintsVectorAdjoint;

    constraintsVectorAdjoint.push_back(&d_constraintMatrixAdjoint);
    constraintsVectorAdjoint.push_back(d_constraintDFTClass);


    std::vector<Quadrature<1>> quadratureVector(0);

    unsigned int quadRhsVal = std::cbrt(d_gaussQuadAdjoint->size());
    pcout << " rhs quad adjoint val  = " << quadRhsVal << "\n";

    quadratureVector.push_back(QGauss<1>(quadRhsVal));

    d_matrixFreeDataAdjoint.reinit(dealii::MappingQ1<3, 3>(),
                                   matrixFreeDofHandlerVectorInput,
                                   constraintsVectorAdjoint,
                                   quadratureVector,
                                   additional_data);
    d_adjointMFAdjointConstraints = 0;
    d_adjointMFPsiConstraints     = 1;
    d_quadAdjointIndex            = 0;

    std::vector<unsigned int> constraintedDofsAdjointMF;
    constraintedDofsAdjointMF = d_matrixFreeDataAdjoint.get_constrained_dofs(
      d_adjointMFAdjointConstraints);
    unsigned int sizeLocalAdjointMF = constraintedDofsAdjointMF.size();

    unsigned int numConstraintsLocalMF = 0;
    for (unsigned int iNode = 0; iNode < sizeLocalAdjointMF; iNode++)
      {
        if (constraintedDofsAdjointMF[iNode] < locallyOwnedDofs)
          {
            numConstraintsLocalMF++;
          }
      }

    MPI_Allreduce(MPI_IN_PLACE,
                  &numConstraintsLocalMF,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numConstraintsLocalMF),
                  MPI_SUM,
                  d_mpiComm_domain);

    MPI_Allreduce(MPI_IN_PLACE,
                  &sizeLocalAdjointMF,
                  1,
                  dftfe::dataTypes::mpi_type_id(&sizeLocalAdjointMF),
                  MPI_SUM,
                  d_mpiComm_domain);

    pcout << " no of constrained adjoint from MF  = " << sizeLocalAdjointMF
          << "\n";
    pcout << " no of local constrained adjoint from MF  = "
          << numConstraintsLocalMF << "\n";
  }

  void
  inverseDFT::setTargetDensity(
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValues,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesSpinPolarized)
  {
    unsigned int totalOwnedCellsElectro =
      d_dftMatrixFreeDataElectro->n_physical_cells();

    const dealii::Quadrature<3> &quadratureRule =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroRhsQuadIndex);

    const unsigned int numQuadPointsElectroPerCell = quadratureRule.size();


    dealii::DoFHandler<3>::active_cell_iterator
      cellElectro = d_dofHandlerElectroDFTClass->begin_active(),
      endElectro  = d_dofHandlerElectroDFTClass->end();


    d_rhoTarget.resize(d_numSpins);
    if (d_numSpins == 1)
      {
        unsigned int spinIndex = 0;
        d_rhoTarget[spinIndex].resize(totalOwnedCellsElectro);
        unsigned int iElemElectro = 0;
        for (; cellElectro != endElectro; ++cellElectro)
          if (cellElectro->is_locally_owned())
            {
              d_rhoTarget[spinIndex][iElemElectro].resize(
                numQuadPointsElectroPerCell);
              const std::vector<double> &cellLevelRho =
                rhoOutValues.find(cellElectro->id())->second;
              for (unsigned int iQuad = 0; iQuad < numQuadPointsElectroPerCell;
                   iQuad++)
                {
                  d_rhoTarget[spinIndex][iElemElectro][iQuad] =
                    0.5 * cellLevelRho[iQuad];
                }
              iElemElectro++;
            }
      }
    else
      {
        unsigned int spinIndex1 = 0;
        d_rhoTarget[spinIndex1].resize(totalOwnedCellsElectro);

        unsigned int spinIndex2 = 0;
        d_rhoTarget[spinIndex2].resize(totalOwnedCellsElectro);

        unsigned int iElemElectro = 0;
        for (; cellElectro != endElectro; ++cellElectro)
          if (cellElectro->is_locally_owned())
            {
              d_rhoTarget[spinIndex1].resize(numQuadPointsElectroPerCell);
              d_rhoTarget[spinIndex2].resize(numQuadPointsElectroPerCell);
              const std::vector<double> &cellLevelRhoSpinPolarized =
                rhoOutValuesSpinPolarized.find(cellElectro->id())->second;
              for (unsigned int iQuad = 0; iQuad < numQuadPointsElectroPerCell;
                   iQuad++)
                {
                  d_rhoTarget[spinIndex1][iElemElectro][iQuad] =
                    cellLevelRhoSpinPolarized[d_numSpins * iQuad + spinIndex1];

                  d_rhoTarget[spinIndex2][iElemElectro][iQuad] =
                    cellLevelRhoSpinPolarized[d_numSpins * iQuad + spinIndex2];
                }
              iElemElectro++;
            }
      }
  }

  void
  inverseDFT::computeHartreePotOnParentQuad(
    std::map<dealii::CellId, std::vector<double>> &hartreeQuadData)
  {
    unsigned int numElectrons = d_dftBaseClass->getNumElectrons();
    // set up the constraints and the matrixFreeObj
    pcout << " numElectrons = " << numElectrons << "\n";

    // TODO does not assume periodic BCs.
    std::vector<std::vector<double>> atomLocations =
      d_dftBaseClass->getAtomLocationsCart();


    dealii::IndexSet locallyRelevantDofsElectro;


    dealii::DoFTools::extract_locally_relevant_dofs(
      *d_dofHandlerElectroDFTClass, locallyRelevantDofsElectro);

    dealii::AffineConstraints<double> d_constraintMatrixElectroHartree;
    // TODO periodic boundary conditions are not included
    d_constraintMatrixElectroHartree.clear();
    d_constraintMatrixElectroHartree.reinit(locallyRelevantDofsElectro);
    dealii::DoFTools::make_hanging_node_constraints(
      *d_dofHandlerElectroDFTClass, d_constraintMatrixElectroHartree);

    const unsigned int dofs_per_cell =
      d_dofHandlerElectroDFTClass->get_fe().dofs_per_cell;
    const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const unsigned int dofs_per_face =
      d_dofHandlerElectroDFTClass->get_fe().dofs_per_face;

    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      dofs_per_cell);
    std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
      dofs_per_face);

    std::vector<bool> dofs_touched(d_dofHandlerElectroDFTClass->n_dofs(),
                                   false);

    dealii::MappingQGeneric<3, 3> mapping(1);
    std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coords_electro;
    dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      mapping, *d_dofHandlerElectroDFTClass, dof_coords_electro);


    dealii::DoFHandler<3>::active_cell_iterator
      cellElectro = d_dofHandlerElectroDFTClass->begin_active(),
      endElectro  = d_dofHandlerElectroDFTClass->end();
    for (; cellElectro != endElectro; ++cellElectro)
      if (cellElectro->is_locally_owned() || cellElectro->is_ghost())
        {
          cellElectro->get_dof_indices(cellGlobalDofIndices);
          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              const unsigned int boundaryId =
                cellElectro->face(iFace)->boundary_id();
              if (boundaryId == 0)
                {
                  cellElectro->face(iFace)->get_dof_indices(
                    iFaceGlobalDofIndices);
                  for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                       ++iFaceDof)
                    {
                      const dealii::types::global_dof_index nodeId =
                        iFaceGlobalDofIndices[iFaceDof];
                      if (dofs_touched[nodeId])
                        continue;
                      dofs_touched[nodeId] = true;
                      if (!d_constraintMatrixElectroHartree.is_constrained(
                            nodeId))
                        {
                          pcout << " setting constraints for = " << nodeId
                                << "\n";
                          //                          double rad = 0.0;
                          //                          rad =
                          //                          dof_coords_electro[nodeId][0]*dof_coords_electro[nodeId][0];
                          //                          rad +=
                          //                          dof_coords_electro[nodeId][1]*dof_coords_electro[nodeId][1];
                          //                          rad +=
                          //                          dof_coords_electro[nodeId][2]*dof_coords_electro[nodeId][2];
                          //                          rad = std::sqrt(rad);
                          //                          if( rad < 1e-6)
                          //                            {
                          //                              pcout<<"Errorrrrrr in
                          //                              rad \n";
                          //                            }
                          double nodalConstraintVal = 0.0;
                          for (unsigned int iAtom = 0;
                               iAtom < atomLocations.size();
                               iAtom++)
                            {
                              double rad = 0.0;
                              rad += (atomLocations[iAtom][2] -
                                      dof_coords_electro[nodeId][0]) *
                                     (atomLocations[iAtom][2] -
                                      dof_coords_electro[nodeId][0]);
                              rad += (atomLocations[iAtom][3] -
                                      dof_coords_electro[nodeId][1]) *
                                     (atomLocations[iAtom][3] -
                                      dof_coords_electro[nodeId][1]);
                              rad += (atomLocations[iAtom][4] -
                                      dof_coords_electro[nodeId][2]) *
                                     (atomLocations[iAtom][4] -
                                      dof_coords_electro[nodeId][2]);
                              rad = std::sqrt(rad);
                              if (d_dftParams.isPseudopotential)
                                nodalConstraintVal +=
                                  atomLocations[iAtom][1] / rad;
                              else
                                nodalConstraintVal +=
                                  atomLocations[iAtom][0] / rad;
                            }
                          d_constraintMatrixElectroHartree.add_line(nodeId);
                          d_constraintMatrixElectroHartree.set_inhomogeneity(
                            nodeId, nodalConstraintVal);
                          //                          d_constraintMatrixElectroHartree.set_inhomogeneity(nodeId,
                          //                          0.0);
                        } // non-hanging node check
                    }     // Face dof loop
                }         // non-periodic boundary id
            }             // Face loop
        }                 // cell locally owned
    d_constraintMatrixElectroHartree.close();

    const dealii::Quadrature<3> &quadratureRuleElectroRhs =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroRhsQuadIndex);

    const dealii::Quadrature<3> &quadratureRuleElectroAx =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroAxQuadIndex);

    std::vector<Quadrature<1>> quadratureVector(0);


    unsigned int quadRhsVal = std::cbrt(quadratureRuleElectroRhs.size());
    pcout << " first quad val  = " << quadRhsVal << "\n";

    unsigned int quadAxVal = std::cbrt(quadratureRuleElectroAx.size());
    pcout << " second quad val  = " << quadAxVal << "\n";

    quadratureVector.push_back(QGauss<1>(quadRhsVal));
    quadratureVector.push_back(QGauss<1>(quadAxVal));

    typename MatrixFree<3>::AdditionalData additional_data;
    // comment this if using deal ii version 9
    // additional_data.mpi_communicator = d_mpiCommParent;
    additional_data.tasks_parallel_scheme =
      MatrixFree<3>::AdditionalData::partition_partition;
    additional_data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;

    std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;
    matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerElectroDFTClass);

    std::vector<const dealii::AffineConstraints<double> *>
      constraintsVectorElectro;

    constraintsVectorElectro.push_back(&d_constraintMatrixElectroHartree);

    // TODO check if passing the quadrature rules this way is correct
    dealii::MatrixFree<3, double> matrixFreeElectro;
    matrixFreeElectro.reinit(matrixFreeDofHandlerVectorInput,
                             constraintsVectorElectro,
                             quadratureVector,
                             additional_data);
    unsigned int dofHandlerElectroIndex = 0;
    unsigned int quadratureElectroRhsId = 0;
    unsigned int quadratureElectroAxId  = 1;

    std::map<dealii::types::global_dof_index, double> dummyAtomMap;
    std::map<dealii::CellId, std::vector<double>>     dummySmearedChargeValues;

    std::map<dealii::CellId, std::vector<double>> totalRhoValues;

    distributedCPUVec<double> vHartreeElectroNodal;

    vectorTools::createDealiiVector<double>(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex),
      1,
      vHartreeElectroNodal);

    vHartreeElectroNodal = 0.0;


    std::vector<unsigned int> constraintedDofsInverse;
    constraintedDofsInverse =
      matrixFreeElectro.get_constrained_dofs(dofHandlerElectroIndex);
    unsigned int sizeLocalInversePot = constraintedDofsInverse.size();

    MPI_Allreduce(MPI_IN_PLACE,
                  &sizeLocalInversePot,
                  1,
                  dftfe::dataTypes::mpi_type_id(&sizeLocalInversePot),
                  MPI_SUM,
                  d_mpiComm_domain);

    pcout << " no of constrained inverse  = " << sizeLocalInversePot << "\n";


    dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerDFTClass
                                                         ->begin_active(),
                                                endc =
                                                  d_dofHandlerDFTClass->end();
    unsigned int iElem = 0;

    const dealii::Quadrature<3> &quadratureRule =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
    const unsigned int numQuadPointsPerCell = quadratureRule.size();

    unsigned int spinIndex1 = 0;
    unsigned int spinIndex2 = (d_numSpins == 2) ? 1 : 0;

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          //          std::vector<double> cellLevelQuadInput;
          totalRhoValues[cell->id()].resize(numQuadPointsPerCell);
          std::fill(totalRhoValues[cell->id()].begin(),
                    totalRhoValues[cell->id()].end(),
                    0.0);
          for (unsigned int iQuad = 0; iQuad < numQuadPointsPerCell; iQuad++)
            {
              totalRhoValues[cell->id()][iQuad] =
                d_rhoTarget[spinIndex1][iElem][iQuad] +
                d_rhoTarget[spinIndex2][iElem][iQuad];
            }
          //          totalRhoValues[cell->id()] = cellLevelQuadInput;
          iElem++;
        }


    pcout << " solving possion in the pot base \n";
    d_dftBaseClass->solvePoissonProblem(
      matrixFreeElectro,
      vHartreeElectroNodal,
      d_constraintMatrixElectroHartree,
      dofHandlerElectroIndex,
      quadratureElectroRhsId,
      quadratureElectroAxId,
      dummyAtomMap,
      dummySmearedChargeValues,
      0, // smearedChargeQuadratureId
      totalRhoValues,
      true,  //          isComputeDiagonalA
      false, //         isComputeMeanValueConstraint
      false, //         smearedNuclearCharges
      true,  //         isRhoValues
      false, //           isGradSmearedChargeRhs
      0,     // smearedChargeGradientComponentId
      false, //          storeSmearedChargeRhs
      false, //          reuseSmearedChargeRhs
      true,  //        reinitializeFastConstraint
      d_mpiComm_parent,
      d_mpiComm_domain);

    dftUtils::constraintMatrixInfo constraintsMatrixDataInfoElectro;
    constraintsMatrixDataInfoElectro.initialize(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex),
      d_constraintMatrixElectroHartree);

    constraintsMatrixDataInfoElectro.precomputeMaps(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex),
      vHartreeElectroNodal.get_partitioner(),
      1); // blockSize


    constraintsMatrixDataInfoElectro.distribute(vHartreeElectroNodal, 1);
    //    vHartreeElectroNodal.update_ghost_values();
    /*
         pcout<<"writing hartree pot output\n";
            dealii::DataOut<3, dealii::DoFHandler<3>> data_out_hartree;

            data_out_hartree.attach_dof_handler(*d_dofHandlerElectroDFTClass);

            std::string outputVecName1 = "hartree pot";
            data_out_hartree.add_data_vector(vHartreeElectroNodal,outputVecName1);


            data_out_hartree.build_patches();
            data_out_hartree.write_vtu_with_pvtu_record("./", "hartreePot",
       0,d_mpiComm_domain,2, 4);
    */

    const unsigned int numQuadPointsElectroPerCell =
      quadratureRuleElectroRhs.size();

    const unsigned int nLocalCellsElectro =
      matrixFreeElectro.n_physical_cells();


    std::map<dealii::CellId, std::vector<double>> quadratureGradValueData;

    cellElectro = d_dofHandlerElectroDFTClass->begin_active();
    endElectro  = d_dofHandlerElectroDFTClass->end();
    for (; cellElectro != endElectro; ++cellElectro)
      if (cellElectro->is_locally_owned())
        {
          hartreeQuadData[cellElectro->id()].resize(numQuadPointsElectroPerCell,
                                                    0.0);
        }

    d_dftBaseClass->interpolateElectroNodalDataToQuadratureDataGeneral(
      matrixFreeElectro,
      dofHandlerElectroIndex,
      quadratureElectroRhsId,
      vHartreeElectroNodal,
      hartreeQuadData,
      quadratureGradValueData,
      false // isEvaluateGradData
    );
  }

  void
  inverseDFT::setPotBaseExactNuclear()
  {
    unsigned int numElectrons = d_dftBaseClass->getNumElectrons();
    // set up the constraints and the matrixFreeObj
    pcout << " numElectrons = " << numElectrons << "\n";

    // TODO does not assume periodic BCs.
    std::vector<std::vector<double>> atomLocations =
      d_dftBaseClass->getAtomLocationsCart();


    dealii::IndexSet locallyRelevantDofsElectro;


    dealii::DoFTools::extract_locally_relevant_dofs(
      *d_dofHandlerElectroDFTClass, locallyRelevantDofsElectro);

    dealii::AffineConstraints<double> d_constraintMatrixElectroHartree;
    // TODO periodic boundary conditions are not included
    d_constraintMatrixElectroHartree.clear();
    d_constraintMatrixElectroHartree.reinit(locallyRelevantDofsElectro);
    dealii::DoFTools::make_hanging_node_constraints(
      *d_dofHandlerElectroDFTClass, d_constraintMatrixElectroHartree);

    const unsigned int dofs_per_cell =
      d_dofHandlerElectroDFTClass->get_fe().dofs_per_cell;
    const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
    const unsigned int dofs_per_face =
      d_dofHandlerElectroDFTClass->get_fe().dofs_per_face;

    std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      dofs_per_cell);
    std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
      dofs_per_face);

    std::vector<bool> dofs_touched(d_dofHandlerElectroDFTClass->n_dofs(),
                                   false);

    dealii::MappingQGeneric<3, 3> mapping(1);
    std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coords_electro;
    dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      mapping, *d_dofHandlerElectroDFTClass, dof_coords_electro);


    dealii::DoFHandler<3>::active_cell_iterator
      cellElectro = d_dofHandlerElectroDFTClass->begin_active(),
      endElectro  = d_dofHandlerElectroDFTClass->end();
    for (; cellElectro != endElectro; ++cellElectro)
      if (cellElectro->is_locally_owned() || cellElectro->is_ghost())
        {
          cellElectro->get_dof_indices(cellGlobalDofIndices);
          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              const unsigned int boundaryId =
                cellElectro->face(iFace)->boundary_id();
              if (boundaryId == 0)
                {
                  cellElectro->face(iFace)->get_dof_indices(
                    iFaceGlobalDofIndices);
                  for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                       ++iFaceDof)
                    {
                      const dealii::types::global_dof_index nodeId =
                        iFaceGlobalDofIndices[iFaceDof];
                      if (dofs_touched[nodeId])
                        continue;
                      dofs_touched[nodeId] = true;
                      if (!d_constraintMatrixElectroHartree.is_constrained(
                            nodeId))
                        {
                          pcout << " setting constraints for = " << nodeId
                                << "\n";
                          //                          double rad = 0.0;
                          //                          rad =
                          //                          dof_coords_electro[nodeId][0]*dof_coords_electro[nodeId][0];
                          //                          rad +=
                          //                          dof_coords_electro[nodeId][1]*dof_coords_electro[nodeId][1];
                          //                          rad +=
                          //                          dof_coords_electro[nodeId][2]*dof_coords_electro[nodeId][2];
                          //                          rad = std::sqrt(rad);
                          //                          if( rad < 1e-6)
                          //                            {
                          //                              pcout<<"Errorrrrrr in
                          //                              rad \n";
                          //                            }
                          double nodalConstraintVal = 0.0;
                          for (unsigned int iAtom = 0;
                               iAtom < atomLocations.size();
                               iAtom++)
                            {
                              double rad = 0.0;
                              rad += (atomLocations[iAtom][2] -
                                      dof_coords_electro[nodeId][0]) *
                                     (atomLocations[iAtom][2] -
                                      dof_coords_electro[nodeId][0]);
                              rad += (atomLocations[iAtom][3] -
                                      dof_coords_electro[nodeId][1]) *
                                     (atomLocations[iAtom][3] -
                                      dof_coords_electro[nodeId][1]);
                              rad += (atomLocations[iAtom][4] -
                                      dof_coords_electro[nodeId][2]) *
                                     (atomLocations[iAtom][4] -
                                      dof_coords_electro[nodeId][2]);
                              rad = std::sqrt(rad);
                              if (d_dftParams.isPseudopotential)
                                nodalConstraintVal +=
                                  atomLocations[iAtom][1] / rad;
                              else
                                nodalConstraintVal +=
                                  atomLocations[iAtom][0] / rad;
                            }
                          d_constraintMatrixElectroHartree.add_line(nodeId);
                          d_constraintMatrixElectroHartree.set_inhomogeneity(
                            nodeId, nodalConstraintVal);
                          //                          d_constraintMatrixElectroHartree.set_inhomogeneity(nodeId,
                          //                          0.0);
                        } // non-hanging node check
                    }     // Face dof loop
                }         // non-periodic boundary id
            }             // Face loop
        }                 // cell locally owned
    d_constraintMatrixElectroHartree.close();

    const dealii::Quadrature<3> &quadratureRuleElectroRhs =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroRhsQuadIndex);

    const dealii::Quadrature<3> &quadratureRuleElectroAx =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroAxQuadIndex);

    std::vector<Quadrature<1>> quadratureVector(0);


    unsigned int quadRhsVal = std::cbrt(quadratureRuleElectroRhs.size());
    pcout << " first quad val  = " << quadRhsVal << "\n";

    unsigned int quadAxVal = std::cbrt(quadratureRuleElectroAx.size());
    pcout << " second quad val  = " << quadAxVal << "\n";

    quadratureVector.push_back(QGauss<1>(quadRhsVal));
    quadratureVector.push_back(QGauss<1>(quadAxVal));

    typename MatrixFree<3>::AdditionalData additional_data;
    // comment this if using deal ii version 9
    // additional_data.mpi_communicator = d_mpiCommParent;
    additional_data.tasks_parallel_scheme =
      MatrixFree<3>::AdditionalData::partition_partition;
    additional_data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;

    std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;
    matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerElectroDFTClass);

    std::vector<const dealii::AffineConstraints<double> *>
      constraintsVectorElectro;

    constraintsVectorElectro.push_back(&d_constraintMatrixElectroHartree);

    // TODO check if passing the quadrature rules this way is correct
    dealii::MatrixFree<3, double> matrixFreeElectro;
    matrixFreeElectro.reinit(matrixFreeDofHandlerVectorInput,
                             constraintsVectorElectro,
                             quadratureVector,
                             additional_data);
    unsigned int dofHandlerElectroIndex = 0;
    unsigned int quadratureElectroRhsId = 0;
    unsigned int quadratureElectroAxId  = 1;

    std::map<dealii::types::global_dof_index, double> dummyAtomMap;
    std::map<dealii::CellId, std::vector<double>>     dummySmearedChargeValues;

    std::map<dealii::CellId, std::vector<double>> totalRhoValues;

    distributedCPUVec<double> vHartreeElectroNodal;

    vectorTools::createDealiiVector<double>(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex),
      1,
      vHartreeElectroNodal);

    vHartreeElectroNodal = 0.0;


    std::vector<unsigned int> constraintedDofsInverse;
    constraintedDofsInverse =
      matrixFreeElectro.get_constrained_dofs(dofHandlerElectroIndex);
    unsigned int sizeLocalInversePot = constraintedDofsInverse.size();

    MPI_Allreduce(MPI_IN_PLACE,
                  &sizeLocalInversePot,
                  1,
                  dftfe::dataTypes::mpi_type_id(&sizeLocalInversePot),
                  MPI_SUM,
                  d_mpiComm_domain);

    pcout << " no of constrained inverse  = " << sizeLocalInversePot << "\n";


    dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerDFTClass
                                                         ->begin_active(),
                                                endc =
                                                  d_dofHandlerDFTClass->end();
    unsigned int iElem = 0;

    const dealii::Quadrature<3> &quadratureRule =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
    const unsigned int numQuadPointsPerCell = quadratureRule.size();

    unsigned int spinIndex1 = 0;
    unsigned int spinIndex2 = (d_numSpins == 2) ? 1 : 0;

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          //          std::vector<double> cellLevelQuadInput;
          totalRhoValues[cell->id()].resize(numQuadPointsPerCell);
          std::fill(totalRhoValues[cell->id()].begin(),
                    totalRhoValues[cell->id()].end(),
                    0.0);
          for (unsigned int iQuad = 0; iQuad < numQuadPointsPerCell; iQuad++)
            {
              totalRhoValues[cell->id()][iQuad] =
                d_rhoTarget[spinIndex1][iElem][iQuad] +
                d_rhoTarget[spinIndex2][iElem][iQuad];
            }
          //          totalRhoValues[cell->id()] = cellLevelQuadInput;
          iElem++;
        }


    pcout << " solving possion in the pot base \n";
    d_dftBaseClass->solvePoissonProblem(
      matrixFreeElectro,
      vHartreeElectroNodal,
      d_constraintMatrixElectroHartree,
      dofHandlerElectroIndex,
      quadratureElectroRhsId,
      quadratureElectroAxId,
      dummyAtomMap,
      dummySmearedChargeValues,
      0, // smearedChargeQuadratureId
      totalRhoValues,
      true,  //          isComputeDiagonalA
      false, //         isComputeMeanValueConstraint
      false, //         smearedNuclearCharges
      true,  //         isRhoValues
      false, //           isGradSmearedChargeRhs
      0,     // smearedChargeGradientComponentId
      false, //          storeSmearedChargeRhs
      false, //          reuseSmearedChargeRhs
      true,  //        reinitializeFastConstraint
      d_mpiComm_parent,
      d_mpiComm_domain);

    dftUtils::constraintMatrixInfo constraintsMatrixDataInfoElectro;
    constraintsMatrixDataInfoElectro.initialize(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex),
      d_constraintMatrixElectroHartree);

    constraintsMatrixDataInfoElectro.precomputeMaps(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex),
      vHartreeElectroNodal.get_partitioner(),
      1); // blockSize


    constraintsMatrixDataInfoElectro.distribute(vHartreeElectroNodal, 1);
    //    vHartreeElectroNodal.update_ghost_values();

    d_potBaseQuadData.resize(d_numSpins);

    const unsigned int numQuadPointsElectroPerCell =
      quadratureRuleElectroRhs.size();

    const unsigned int nLocalCellsElectro =
      matrixFreeElectro.n_physical_cells();


    d_potBaseQuadData[0].resize(nLocalCellsElectro);
    if (d_numSpins == 2)
      {
        d_potBaseQuadData[1].resize(nLocalCellsElectro);
      }


    std::map<dealii::CellId, std::vector<double>> quadratureValueData;
    std::map<dealii::CellId, std::vector<double>> quadratureGradValueData;

    cellElectro = d_dofHandlerElectroDFTClass->begin_active();
    endElectro  = d_dofHandlerElectroDFTClass->end();
    for (; cellElectro != endElectro; ++cellElectro)
      if (cellElectro->is_locally_owned())
        {
          quadratureValueData[cellElectro->id()].resize(
            numQuadPointsElectroPerCell, 0.0);
        }

    d_dftBaseClass->interpolateElectroNodalDataToQuadratureDataGeneral(
      matrixFreeElectro,
      dofHandlerElectroIndex,
      quadratureElectroRhsId,
      vHartreeElectroNodal,
      quadratureValueData,
      quadratureGradValueData,
      false // isEvaluateGradData
    );

    dealii::FEValues<3> fe_valuesElectro(d_dofHandlerElectroDFTClass->get_fe(),
                                         quadratureRuleElectroRhs,
                                         dealii::update_quadrature_points);


    unsigned int iElemElectro = 0;
    cellElectro               = d_dofHandlerElectroDFTClass->begin_active();
    endElectro                = d_dofHandlerElectroDFTClass->end();
    for (; cellElectro != endElectro; ++cellElectro)
      if (cellElectro->is_locally_owned())
        {
          fe_valuesElectro.reinit(cellElectro);
          d_potBaseQuadData[0][iElemElectro].resize(numQuadPointsElectroPerCell,
                                                    0.0);
          std::copy(quadratureValueData[cellElectro->id()].begin(),
                    quadratureValueData[cellElectro->id()].end(),
                    d_potBaseQuadData[0][iElemElectro].begin());

          for (unsigned int iQuad = 0; iQuad < numQuadPointsElectroPerCell;
               iQuad++)
            {
              dealii::Point<3, double> qPointVal =
                fe_valuesElectro.quadrature_point(iQuad);
              for (unsigned int iAtom = 0; iAtom < atomLocations.size();
                   iAtom++)
                {
                  double rad = 0.0;
                  rad += (atomLocations[iAtom][2] - qPointVal[0]) *
                         (atomLocations[iAtom][2] - qPointVal[0]);
                  rad += (atomLocations[iAtom][3] - qPointVal[1]) *
                         (atomLocations[iAtom][3] - qPointVal[1]);
                  rad += (atomLocations[iAtom][4] - qPointVal[2]) *
                         (atomLocations[iAtom][4] - qPointVal[2]);
                  rad = std::sqrt(rad);
                  if (d_dftParams.isPseudopotential)
                    d_potBaseQuadData[0][iElemElectro][iQuad] -=
                      atomLocations[iAtom][1] / rad;
                  else
                    d_potBaseQuadData[0][iElemElectro][iQuad] -=
                      atomLocations[iAtom][0] / rad;
                }
            }
          if (d_numSpins == 2)
            {
              d_potBaseQuadData[1][iElemElectro].resize(
                numQuadPointsElectroPerCell, 0.0);
              std::copy(d_potBaseQuadData[0][iElemElectro].begin(),
                        d_potBaseQuadData[0][iElemElectro].end(),
                        d_potBaseQuadData[1][iElemElectro].begin());
            }
          iElemElectro++;
        }
  }

  void
  inverseDFT::setPotBasePoissonNuclear()
  {
    unsigned int numElectrons = d_dftBaseClass->getNumElectrons();
    // set up the constraints and the matrixFreeObj
    pcout << " numElectrons = " << numElectrons << "\n";

    std::map<dealii::CellId, std::vector<double>> totalRhoValues;

    distributedCPUVec<double> vTotalElectroNodal;



    dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerDFTClass
                                                         ->begin_active(),
                                                endc =
                                                  d_dofHandlerDFTClass->end();
    unsigned int iElem = 0;

    const dealii::Quadrature<3> &quadratureRule =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
    const unsigned int numQuadPointsPerCell = quadratureRule.size();

    unsigned int spinIndex1 = 0;
    unsigned int spinIndex2 = (d_numSpins == 2) ? 1 : 0;

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          //          std::vector<double> cellLevelQuadInput;
          totalRhoValues[cell->id()].resize(numQuadPointsPerCell);
          std::fill(totalRhoValues[cell->id()].begin(),
                    totalRhoValues[cell->id()].end(),
                    0.0);
          for (unsigned int iQuad = 0; iQuad < numQuadPointsPerCell; iQuad++)
            {
              totalRhoValues[cell->id()][iQuad] =
                d_rhoTarget[spinIndex1][iElem][iQuad] +
                d_rhoTarget[spinIndex2][iElem][iQuad];
            }
          //          totalRhoValues[cell->id()] = cellLevelQuadInput;
          iElem++;
        }


    pcout << " solving poisson in the pot nuclear \n";

    d_dftBaseClass->solvePhiTotalAllElectronNonPeriodic(vTotalElectroNodal,
                                                        totalRhoValues,
                                                        d_mpiComm_parent,
                                                        d_mpiComm_domain);



    //    vTotalElectroNodal.update_ghost_values();

    d_potBaseQuadData.resize(d_numSpins);

    const dealii::Quadrature<3> &quadratureRuleElectroRhs =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroRhsQuadIndex);
    const unsigned int numQuadPointsElectroPerCell =
      quadratureRuleElectroRhs.size();

    const unsigned int nLocalCellsElectro =
      d_dftMatrixFreeDataElectro->n_physical_cells();


    d_potBaseQuadData[0].resize(nLocalCellsElectro);
    if (d_numSpins == 2)
      {
        d_potBaseQuadData[1].resize(nLocalCellsElectro);
      }


    std::map<dealii::CellId, std::vector<double>> quadratureValueData;
    std::map<dealii::CellId, std::vector<double>> quadratureGradValueData;

    dealii::DoFHandler<3>::active_cell_iterator
      cellElectro = d_dofHandlerElectroDFTClass->begin_active(),
      endElectro  = d_dofHandlerElectroDFTClass->end();

    for (; cellElectro != endElectro; ++cellElectro)
      if (cellElectro->is_locally_owned())
        {
          quadratureValueData[cellElectro->id()].resize(
            numQuadPointsElectroPerCell, 0.0);
        }

    d_dftBaseClass->interpolateElectroNodalDataToQuadratureDataGeneral(
      *d_dftMatrixFreeDataElectro,
      d_dftElectroDoFHandlerIndex,
      d_dftElectroRhsQuadIndex,
      vTotalElectroNodal,
      quadratureValueData,
      quadratureGradValueData,
      false // isEvaluateGradData
    );

    dealii::FEValues<3> fe_valuesElectro(d_dofHandlerElectroDFTClass->get_fe(),
                                         quadratureRuleElectroRhs,
                                         dealii::update_quadrature_points);


    unsigned int iElemElectro = 0;
    cellElectro               = d_dofHandlerElectroDFTClass->begin_active();
    endElectro                = d_dofHandlerElectroDFTClass->end();
    for (; cellElectro != endElectro; ++cellElectro)
      if (cellElectro->is_locally_owned())
        {
          fe_valuesElectro.reinit(cellElectro);
          d_potBaseQuadData[0][iElemElectro].resize(numQuadPointsElectroPerCell,
                                                    0.0);
          std::copy(quadratureValueData[cellElectro->id()].begin(),
                    quadratureValueData[cellElectro->id()].end(),
                    d_potBaseQuadData[0][iElemElectro].begin());

          if (d_numSpins == 2)
            {
              d_potBaseQuadData[1][iElemElectro].resize(
                numQuadPointsElectroPerCell, 0.0);
              std::copy(d_potBaseQuadData[0][iElemElectro].begin(),
                        d_potBaseQuadData[0][iElemElectro].end(),
                        d_potBaseQuadData[1][iElemElectro].begin());
            }
          iElemElectro++;
        }
  }

  void
  inverseDFT::setPotBase()
  {
    setPotBasePoissonNuclear();
  }

  void
  inverseDFT::readVxcDataFromFile(
    std::vector<distributedCPUVec<double>> &vxcChildNodes)
  {
    vxcChildNodes.resize(d_numSpins);
    vectorTools::createDealiiVector<double>(
      d_matrixFreeDataVxc.get_vector_partitioner(d_dofHandlerVxcIndex),
      1,
      vxcChildNodes[0]);
    vxcChildNodes[0] = 0.0;

    if (d_numSpins == 2)
      {
        vxcChildNodes[1].reinit(vxcChildNodes[0]);
      }

    std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coord_child;
    dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      dealii::MappingQ1<3, 3>(), d_dofHandlerTriaVxc, dof_coord_child);
    dealii::types::global_dof_index numberDofsChild =
      d_dofHandlerTriaVxc.n_dofs();

    std::vector<coordinateValues> inputDataFromFile;
    inputDataFromFile.resize(numberDofsChild);

    const std::string filename =
      d_dftParams.vxcDataFolder + "/" + d_dftParams.fileNameReadVxcPostFix;
    std::ifstream vxcInputFile(filename);

    double nodalValue  = 0.0;
    double xcoordValue = 0.0;
    double ycoordValue = 0.0;
    double zcoordValue = 0.0;
    double fieldValue0 = 0.0;
    double fieldValue1 = 0.0;


    for (dealii::types::global_dof_index iNode = 0; iNode < numberDofsChild;
         iNode++)
      {
        vxcInputFile >> nodalValue;
        vxcInputFile >> xcoordValue;
        vxcInputFile >> ycoordValue;
        vxcInputFile >> zcoordValue;
        vxcInputFile >> fieldValue0;
        if (d_numSpins == 2)
          {
            vxcInputFile >> fieldValue1;
          }
        if (vxcChildNodes[0].in_local_range(nodalValue))
          {
            double distBetweenNodes = 0.0;
            distBetweenNodes += (xcoordValue - dof_coord_child[iNode][0]) *
                                (xcoordValue - dof_coord_child[iNode][0]);
            distBetweenNodes += (ycoordValue - dof_coord_child[iNode][1]) *
                                (ycoordValue - dof_coord_child[iNode][1]);
            distBetweenNodes += (zcoordValue - dof_coord_child[iNode][2]) *
                                (zcoordValue - dof_coord_child[iNode][2]);
            distBetweenNodes = std::sqrt(distBetweenNodes);
            if (distBetweenNodes > 1e-6)
              {
                std::cout
                  << " Errorr while reading data global nodes do not match \n";
              }

            vxcChildNodes[0](iNode) = fieldValue0;
            if (d_numSpins == 2)
              {
                vxcChildNodes[1](iNode) = fieldValue1;
              }
          }
      }
    vxcInputFile.close();
  }
  template <typename T>
  void
  inverseDFT::readVxcInput()
  {
    unsigned int totalLocallyOwnedCellsVxc =
      d_matrixFreeDataVxc.n_physical_cells();

    const unsigned int numQuadPointsPerCellInVxc = d_gaussQuadVxc.size();

    double spinFactor = (d_dftParams.spinPolarized == 1) ? 1.0 : 2.0;



    unsigned int locallyOwnedDofs =
      d_dofHandlerDFTClass->n_locally_owned_dofs();

    d_vxcInitialChildNodes.resize(d_numSpins);
    readVxcDataFromFile(d_vxcInitialChildNodes);

    d_targetPotValuesParentQuadData.resize(d_numSpins);

    unsigned int totalOwnedCellsPsi = d_dftMatrixFreeData->n_physical_cells();

    const dealii::Quadrature<3> &quadratureRulePsi =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);

    unsigned int numQuadPointsPerPsiCell = quadratureRulePsi.size();


    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex)
      {
        d_targetPotValuesParentQuadData[spinIndex].resize(totalOwnedCellsPsi);
        for (unsigned int iCell = 0; iCell < totalOwnedCellsPsi; iCell++)
          {
            // TODO set the correct values. For now set to a dummy value
            d_targetPotValuesParentQuadData[spinIndex][iCell].resize(
              numQuadPointsPerPsiCell, 0.0);
          }
      }

    // TODO unComment this to set the adjoint constraints
    distributedCPUVec<double> rhoInputTotal;
    vectorTools::createDealiiVector<double>(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      1,
      rhoInputTotal);
    rhoInputTotal = 0.0;

    unsigned int totalLocallyOwnedCellsPsi =
      d_dftMatrixFreeData->n_physical_cells();

    unsigned int numLocallyOwnedDofsPsi =
      d_dofHandlerDFTClass->n_locally_owned_dofs();
    unsigned int numDofsPerCellPsi =
      d_dofHandlerDFTClass->get_fe().dofs_per_cell;

    std::map<dealii::CellId, std::vector<double>> rhoValues;

    typename DoFHandler<3>::active_cell_iterator cellPsiPtr =
      d_dofHandlerDFTClass->begin_active();
    typename DoFHandler<3>::active_cell_iterator endcellPsiPtr =
      d_dofHandlerDFTClass->end();

    unsigned int iElem      = 0;
    unsigned int spinIndex1 = 0;
    unsigned int spinIndex2 = 0;
    if (d_numSpins == 2)
      {
        spinIndex2 = 1;
      }
    for (; cellPsiPtr != endcellPsiPtr; ++cellPsiPtr)
      {
        if (cellPsiPtr->is_locally_owned())
          {
            const dealii::CellId cellId = cellPsiPtr->id();
            std::vector<double>  cellLevelRho;
            cellLevelRho.resize(numQuadPointsPerPsiCell);
            for (unsigned int iQuad = 0; iQuad < numQuadPointsPerPsiCell;
                 iQuad++)
              {
                cellLevelRho[iQuad] = d_rhoTarget[spinIndex1][iElem][iQuad] +
                                      d_rhoTarget[spinIndex2][iElem][iQuad];
              }
            rhoValues[cellId] = cellLevelRho;
            iElem++;
          }
      }

    d_dftBaseClass->l2ProjectionQuadToNodal(*d_dftMatrixFreeData,
                                            *d_constraintDFTClass,
                                            d_dftDensityDoFHandlerIndex,
                                            d_dftQuadIndex,
                                            rhoValues,
                                            rhoInputTotal);

    d_constraintDFTClass->distribute(rhoInputTotal);
    rhoInputTotal.update_ghost_values();

    setAdjointBoundaryCondition(rhoInputTotal);
  }

  void
  inverseDFT::run()
  {
    dftUtils::printCurrentMemoryUsage(d_mpiComm_domain,
                                      "Before parent cell manager");
    createParentChildDofManager();

    dftUtils::printCurrentMemoryUsage(d_mpiComm_domain,
                                      "after parent cell manager");

    const std::map<dealii::CellId, std::vector<double>> &rhoInValues =
      d_dftBaseClass->getRhoInValues();

    const std::map<dealii::CellId, std::vector<double>> &rhoInSpinPolarised =
      d_dftBaseClass->getRhoInValuesSpinPolarized();

    std::vector<std::vector<double>> rhoValuesFeSpin;
    rhoValuesFeSpin.resize(d_numSpins);
    if (d_numSpins == 1)
      {
        const dealii::Quadrature<3> &quadratureRuleParent =
          d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
        const unsigned int numQuadraturePointsPerCellParent =
          quadratureRuleParent.size();
        unsigned int totalLocallyOwnedCellsParent =
          d_dftMatrixFreeData->n_physical_cells();

        const unsigned int numTotalQuadraturePointsParent =
          totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;

        rhoValuesFeSpin[0].resize(numTotalQuadraturePointsParent, 0.0);
        typename dealii::DoFHandler<3>::active_cell_iterator
          cell             = d_dofHandlerDFTClass->begin_active(),
          endc             = d_dofHandlerDFTClass->end();
        unsigned int iElem = 0;
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              const std::vector<double> &cellLevelRho =
                rhoInValues.find(cell->id())->second;
              for (unsigned int iQuad = 0;
                   iQuad < numQuadraturePointsPerCellParent;
                   iQuad++)
                {
                  unsigned int index =
                    iElem * numQuadraturePointsPerCellParent + iQuad;
                  rhoValuesFeSpin[0][index] = 0.5 * cellLevelRho[iQuad];
                }
              iElem++;
            }
      }
    else
      {
        const dealii::Quadrature<3> &quadratureRuleParent =
          d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
        const unsigned int numQuadraturePointsPerCellParent =
          quadratureRuleParent.size();
        unsigned int totalLocallyOwnedCellsParent =
          d_dftMatrixFreeData->n_physical_cells();

        const unsigned int numTotalQuadraturePointsParent =
          totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;

        rhoValuesFeSpin[0].resize(numTotalQuadraturePointsParent, 0.0);
        rhoValuesFeSpin[1].resize(numTotalQuadraturePointsParent, 0.0);
        typename dealii::DoFHandler<3>::active_cell_iterator
          cell             = d_dofHandlerDFTClass->begin_active(),
          endc             = d_dofHandlerDFTClass->end();
        unsigned int iElem = 0;
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              const std::vector<double> &cellLevelRhoSpinPolarized =
                rhoInSpinPolarised.find(cell->id())->second;
              for (unsigned int iQuad = 0;
                   iQuad < numQuadraturePointsPerCellParent;
                   iQuad++)
                {
                  unsigned int index =
                    iElem * numQuadraturePointsPerCellParent + iQuad;
                  rhoValuesFeSpin[0][index] =
                    cellLevelRhoSpinPolarized[d_numSpins * iQuad + 0];
                  rhoValuesFeSpin[1][index] =
                    cellLevelRhoSpinPolarized[d_numSpins * iQuad + 1];
                }
              iElem++;
            }
      }

    if (d_dftParams.readGaussian)
      {
        setInitialDensityFromGaussian(rhoValuesFeSpin);
      }
    else
      {
        setTargetDensity(rhoInValues, rhoInSpinPolarised);
      }



    if (d_dftParams.readVxcData)
      {
        readVxcInput<double>();
      }
    else
      {
        setInitialPotL2Proj<double>();
      }


    setPotBase();

    inverseDFTSolverFunction<double> inverseDFTSolverFunctionObj(
      d_mpiComm_parent,
      d_mpiComm_domain,
      d_mpiComm_bandgroup,
      d_mpiComm_interpool);

    dftUtils::printCurrentMemoryUsage(d_mpiComm_domain,
                                      "Created inverse dft solver func");

    unsigned int spinFactor = (d_numSpins == 2) ? 1 : 2;

    std::vector<std::vector<std::vector<double>>> weightQuadData;
    weightQuadData.resize(d_numSpins);

    double tauWeight = d_dftParams.inverseTauForSmoothening;


    for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
      {
        unsigned int numCells = d_rhoTarget[iSpin].size();
        weightQuadData[iSpin].resize(numCells);
        for (unsigned int iCell = 0; iCell < numCells; iCell++)
          {
            unsigned int quadSize = d_rhoTarget[iSpin][iCell].size();
            weightQuadData[iSpin][iCell].resize(quadSize, 1.0);
            for (unsigned int iQuad = 0; iQuad < quadSize; iQuad++)
              {
                weightQuadData[iSpin][iCell][iQuad] = 1.0;
                // weightQuadData[iSpin][iCell][iQuad] = 1.0/(std::pow(
                //              spinFactor*d_rhoTarget[iSpin][iCell][iQuad],d_dftParams.inverseAlpha1ForWeights)
                //              + tauWeight);
                // weightQuadData[iSpin][iCell][iQuad] +=
                // std::pow(spinFactor*d_rhoTarget[iSpin][iCell][iQuad],d_dftParams.inverseAlpha2ForWeights);
              }
          }
      }



    operatorDFTClass *kohnShamClassPtr = d_dftBaseClass->getOperatorClass();
#ifdef DFTFE_WITH_DEVICE
    operatorDFTDeviceClass *kohnShamClassDevicePtr =
      d_dftBaseClass->getOperatorDeviceClass();
#endif
    inverseDFTSolverFunctionObj.reinit(
      d_rhoTarget,
      weightQuadData,
      d_potBaseQuadData,
      d_dftBaseClass,
      d_matrixFreeDataAdjoint,
      d_matrixFreeDataVxc,
      *d_constraintDFTClass,     // assumes that the constraint matrix has
                                 // homogenous BC
      d_constraintMatrixAdjoint, // assumes that the constraint matrix has
                                 // homogenous BC
      d_constraintMatrixVxc,
      *kohnShamClassPtr,
#ifdef DFTFE_WITH_DEVICE
      *kohnShamClassDevicePtr,
#endif
      d_inverseDftDoFManagerObjPtr,
      d_kpointWeights,
      d_numSpins,
      d_numEigenValues,
      d_adjointMFPsiConstraints,
      //                                       d_adjointMFPsiConstraints,
      d_adjointMFAdjointConstraints,
      d_dofHandlerVxcIndex,
      d_quadAdjointIndex,
      d_quadVxcIndex,
      true, //         isComputeDiagonalA
      true, //        isComputeShapeFunction
      d_dftParams);

    dftUtils::printCurrentMemoryUsage(d_mpiComm_domain,
                                      "after solver func reinit");


    inverseDFTSolverFunctionObj.setInitialGuess(
      d_vxcInitialChildNodes, d_targetPotValuesParentQuadData);

    std::cout << " vxc initial guess norm before constructor = "
              << d_vxcInitialChildNodes[0].l2_norm() << "\n";
    BFGSInverseDFTSolver BFGSInverseDFTSolverObj(
      d_numSpins,                           // numComponents
      d_dftParams.inverseBFGSTol,           // tol
      d_dftParams.inverseBFGSLineSearchTol, // lineSearchTol
      d_dftParams.inverseMaxBFGSIter,       // maxNumIter
      d_dftParams.inverseBFGSHistory,       // historySize
      d_dftParams.inverseBFGSLineSearch,    // numLineSearch
      d_mpiComm_parent);

    std::cout << " vxc initial guess norm before solve = "
              << d_vxcInitialChildNodes[0].l2_norm() << "\n";
    BFGSInverseDFTSolverObj.solve(inverseDFTSolverFunctionObj,
                                  BFGSInverseDFTSolver::LSType::CP);
  }
} // namespace dftfe

/*
void
  inverseDFT::readVxcInput()
{
  unsigned int totalLocallyOwnedCellsVxc =
    d_matrixFreeDataVxc.n_physical_cells();

const unsigned int numQuadPointsPerCellInVxc = d_gaussQuadVxc.size();

unsigned int locallyOwnedDofs =
  d_dofHandlerDFTClass->n_locally_owned_dofs();
std::vector<distributedCPUVec<double>> vxcReadFromFile;
vxcReadFromFile.resize(d_numSpins);
std::vector<distributedCPUVec<double> *> solutionVectors;
for( unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
  {
    solutionVectors.push_back(&vxcReadFromFile[iSpin]);
  }

d_dftTriaManager->loadTriangulationsSolutionVectors(
  d_dftParams.vxcDataFolder,
  d_dftBaseClass->getFEOrder(),
  1, //nComponents
  solutionVectors,
  d_dftParams.fileNameReadVxcPostFix,
  false // saveSupportTriangulationFlag
);

dftUtils::constraintMatrixInfo constraintsMatrixDataInfoPsi;
constraintsMatrixDataInfoPsi.initialize(
  d_dftMatrixFreeData->get_vector_partitioner(
    d_dftDensityDoFHandlerIndex),
  *d_constraintDFTClass);

std::vector<distributedCPUVec<double>> vxcInitialGuess;

vxcInitialGuess.resize(d_numSpins);

for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex)
  {
    vectorTools::createDealiiVector<double>(
      d_dftMatrixFreeData->get_vector_partitioner(
        d_dftDensityDoFHandlerIndex),
      1,
      vxcInitialGuess[spinIndex]);
    vxcInitialGuess[spinIndex] = 0.0;

    constraintsMatrixDataInfoPsi.precomputeMaps(
      d_dftMatrixFreeData->get_vector_partitioner(
        d_dftDensityDoFHandlerIndex),
      vxcInitialGuess[spinIndex].get_partitioner(),
      1); // blockSize

    for(unsigned int iNode = 0; iNode <locallyOwnedDofs;iNode++)
      {
        vxcInitialGuess[spinIndex].local_element(iNode) =
          solutionVectors[spinIndex]->local_element(iNode);
      }
    constraintsMatrixDataInfoPsi.distribute(vxcInitialGuess[spinIndex],1);

    std::vector<std::vector<double>> initialPotValuesChildQuad;
    initialPotValuesChildQuad.resize(d_numSpins);
    d_vxcInitialChildNodes.resize(d_numSpins);
    std::vector<std::map<dealii::CellId, std::vector<double>>>
      initialPotValuesChildQuadDealiiMap;
    initialPotValuesChildQuadDealiiMap.resize(d_numSpins);

    initialPotValuesChildQuad[spinIndex].resize(totalLocallyOwnedCellsVxc*numQuadPointsPerCellInVxc);
    std::fill(initialPotValuesChildQuad[spinIndex].begin(),initialPotValuesChildQuad[spinIndex].end(),0.0);

    std::vector<dealii::types::global_dof_index>
      fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent;
    vectorTools::computeCellLocalIndexSetMap(
      vxcInitialGuess[spinIndex].get_partitioner(),
      *d_dftMatrixFreeData,
      d_dftDensityDoFHandlerIndex,
      1,
      fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent);

    d_inverseDftDoFManagerObjPtr->interpolateMesh1DataToMesh2QuadPoints(vxcInitialGuess[spinIndex],
                                                                        1,
                                                                        //blockSize fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent,
                                                                        initialPotValuesChildQuad[spinIndex]);

    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  d_dofHandlerTriaVxc.begin_active(), endc = d_dofHandlerTriaVxc.end(); unsigned
      int iElem  = 0; for (; cell != endc; ++cell) if (cell->is_locally_owned() )
        {
          std::vector<double> cellLevelQuadInput;
          cellLevelQuadInput.resize(numQuadPointsPerCellInVxc);
          for(unsigned int iQuad = 0; iQuad < numQuadPointsPerCellInVxc;
               iQuad ++)
            {
              cellLevelQuadInput[iQuad] =
                initialPotValuesChildQuad[spinIndex][iElem*numQuadPointsPerCellInVxc+ iQuad];
            }
          initialPotValuesChildQuadDealiiMap[spinIndex][cell->id()] =
            cellLevelQuadInput; iElem ++;
        }


    d_dftBaseClass->l2ProjectionQuadToNodal(
      d_matrixFreeDataVxc,
      d_constraintMatrixVxc,
      d_dofHandlerVxcIndex,
      d_quadVxcIndex,
      initialPotValuesChildQuadDealiiMap[spinIndex],
      d_vxcInitialChildNodes[spinIndex]);

    d_constraintMatrixVxc.set_zero(d_vxcInitialChildNodes[spinIndex]);

  }
}

*/
//    std::sort(inputDataFromFile.begin(), inputDataFromFile.end(),
//    less_than_key());
//
//    for (dealii::types::global_dof_index iNode = 0; iNode <
//    numberDofsChild; iNode++)
//      {
//        if (vxcChildNodes[0].in_local_range(iNode))
//          {
//            coordinateValues targetCoord;
//            targetCoord.iNode = iNode;
//            targetCoord.xcoord = dof_coord_child[iNode][0];
//            targetCoord.ycoord = dof_coord_child[iNode][1];
//            targetCoord.zcoord = dof_coord_child[iNode][2];
//            targetCoord.value0 =  0.0;
//	    targetCoord.value1 =  0.0;
//            auto it = std::lower_bound(inputDataFromFile.begin(),
//            inputDataFromFile.end(),targetCoord,less_than_key());
////            AssertThrow(
////               it != inputDataFromFile.end(),
////              ExcMessage(
////                "DFT-FE error:  reading vxc error coord not found in
/// input file"));
//
//            double distBetweenNodes = 0.0;
//            distBetweenNodes += (it->xcoord - targetCoord.xcoord)*
//                                (it->xcoord - targetCoord.xcoord);
//            distBetweenNodes += (it->ycoord - targetCoord.ycoord)*
//                                (it->ycoord - targetCoord.ycoord);
//            distBetweenNodes += (it->zcoord - targetCoord.zcoord)*
//                                (it->zcoord - targetCoord.zcoord);
//            distBetweenNodes = std::sqrt(distBetweenNodes);
//            if(distBetweenNodes > 1e-6)
//              {
//                std::cout<<" Errorr while reading data global nodes do not
//                match \n";
//              }
//            vxcChildNodes[0](iNode) = it->value0;
//            if(d_numSpins == 2)
//              {
//                vxcChildNodes[1](iNode) = it->value1;
//              }
//          }
//      }
//  void
//    inverseDFT::setAdjointBoundaryCondition( distributedCPUVec<double>
//    &rhoTarget)
//    {
//
//
//      dealii::IndexSet localSet =
//      d_dftMatrixFreeData->get_locally_owned_set(d_dftDensityDoFHandlerIndex);
//      dealii::IndexSet ghostSet =
//      d_dftMatrixFreeData->get_ghost_set(d_dftDensityDoFHandlerIndex);
//
//
//
//      dealii::IndexSet locallyRelevantDofsVxc;
//
//
//      dealii::DoFTools::extract_locally_relevant_dofs(
//	  *d_dofHandlerDFTClass, locallyRelevantDofsVxc);
//
//
//      d_constraintMatrixAdjoint.clear();
//      d_constraintMatrixAdjoint.reinit(locallyRelevantDofsVxc);
//      dealii::DoFTools::make_hanging_node_constraints(*d_dofHandlerDFTClass,
//      d_constraintMatrixAdjoint);
//
//      const unsigned int dofs_per_cell  =
//      d_dofHandlerDFTClass->get_fe().dofs_per_cell; const unsigned int
//      faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell; const
//      unsigned int dofs_per_face  =
//      d_dofHandlerDFTClass->get_fe().dofs_per_face;
//
//      std::vector<dealii::types::global_dof_index>
//      cellGlobalDofIndices(dofs_per_cell);
//      std::vector<dealii::types::global_dof_index>
//      iFaceGlobalDofIndices(dofs_per_face);
//
//      std::vector<bool> dofs_touched(d_dofHandlerDFTClass->n_dofs(), false);
//
//      dealii::DoFHandler<3>::active_cell_iterator cell =
//      d_dofHandlerDFTClass->begin_active(),
//	    endc = d_dofHandlerDFTClass->end();
//      for (; cell != endc; ++cell)
//	      if (cell->is_locally_owned() || cell->is_ghost())
//          {
//            cell->get_dof_indices(cellGlobalDofIndices);
//            for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
//              {
//                const unsigned int boundaryId =
//                cell->face(iFace)->boundary_id(); if (boundaryId == 0)
//                  {
//                    cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
//                    for (unsigned int iFaceDof = 0; iFaceDof <
//                    dofs_per_face;
//                         ++iFaceDof)
//                      {
//                        const dealii::types::global_dof_index nodeId =
//                          iFaceGlobalDofIndices[iFaceDof];
//                        if (dofs_touched[nodeId])
//                          continue;
//                        dofs_touched[nodeId] = true;
//                        if
//                        (!d_constraintMatrixAdjoint.is_constrained(nodeId))
//                          {
//                            d_constraintMatrixAdjoint.add_line(nodeId);
//                            d_constraintMatrixAdjoint.set_inhomogeneity(nodeId,
//                            0.0);
//                          } // non-hanging node check
//                      }     // Face dof loop
//                  }         // non-periodic boundary id
//              }
//          }
//
//      d_constraintMatrixAdjoint.close();
//
//      typename MatrixFree<3>::AdditionalData additional_data;
//      // comment this if using deal ii version 9
//      // additional_data.mpi_communicator = d_mpiCommParent;
//      additional_data.tasks_parallel_scheme =
//        MatrixFree<3>::AdditionalData::partition_partition;
//
//      additional_data.mapping_update_flags =
//        update_values | update_gradients | update_JxW_values;
//
//      std::vector<const dealii::DoFHandler<3> *>
//      matrixFreeDofHandlerVectorInput;
//      matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerDFTClass);
//      matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerDFTClass);
//
//      std::vector<const dealii::AffineConstraints<double> *>
//        constraintsVectorAdjoint;
//
//      constraintsVectorAdjoint.push_back(&d_constraintMatrixAdjoint);
//      constraintsVectorAdjoint.push_back(d_constraintDFTClass);
//
//
//      std::vector<Quadrature<1>> quadratureVector(0);
//
//      unsigned int quadRhsVal = std::cbrt(d_gaussQuadAdjoint->size());
//      pcout<<" rhs quad adjoint val  = "<<quadRhsVal<<"\n";
//
//      quadratureVector.push_back(QGauss<1>(quadRhsVal));
//
//      d_matrixFreeDataAdjoint.reinit(dealii::MappingQ1<3,3>(),
//                                     matrixFreeDofHandlerVectorInput,
//                                     constraintsVectorAdjoint,
//                                     quadratureVector,
//                                     additional_data);
//      d_adjointMFAdjointConstraints = 0;
//      d_adjointMFPsiConstraints = 1;
//      d_quadAdjointIndex = 0;
//    }