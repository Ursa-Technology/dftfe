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

#include "hubbardClass.h"
#include "AtomCenteredSphericalFunctionProjectorSpline.h"
#include "dftParameters.h"
#include "DataTypeOverloads.h"
#include "constants.h"
#include "BLASWrapper.h"
#include "AtomCenteredPseudoWavefunctionSpline.h"

#if defined(DFTFE_WITH_DEVICE)
#include "deviceKernelsGeneric.h"
#endif

namespace dftfe
{

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  hubbard<ValueType, memorySpace>::hubbard(const MPI_Comm &mpi_comm_parent,
                   const MPI_Comm &mpi_comm_domain,
                   const MPI_Comm &mpi_comm_interPool):
  d_mpi_comm_parent(mpi_comm_parent),
    d_mpi_comm_domain(mpi_comm_domain),
    d_mpi_comm_interPool(mpi_comm_interPool),
    pcout(std::cout,
          (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0))
  {
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType,
            memorySpace>::createAtomCenteredSphericalFunctionsForProjectors()
  {
      for (auto const& [key, val] : d_hubbardSpeciesData)
      {
        pcout<<" key = "<<key<<"\n";
        unsigned int  Znum = val.atomicNumber;

        unsigned int  numberOfProjectors = val.numProj;

        unsigned int numProj;
        unsigned int alpha = 0;
        for (unsigned int i = 0 ; i < numberOfProjectors;  i++)
          {
            char         projRadialFunctionFileName[512];
            unsigned int nQuantumNo = val.nQuantumNum[i];
            unsigned int lQuantumNo = val.lQuantumNum[i];

            pcout<<" nQuantumNo = "<<nQuantumNo<<"\n";
            pcout<<" lQuantumNo = "<<lQuantumNo<<"\n";
            char         waveFunctionFileName[256];
            strcpy(waveFunctionFileName,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(Znum) +
                    "/psi" + std::to_string(nQuantumNo) +
                    std::to_string(lQuantumNo) + ".inp")
                     .c_str());

            d_atomicProjectorFnsMap[std::make_pair(Znum, alpha)] =
              std::make_shared<
                AtomCenteredPseudoWavefunctionSpline>(
                waveFunctionFileName,
                lQuantumNo,
                1E-12);
            alpha++;
          } // i loop

      } // for loop *it
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void hubbard<ValueType ,memorySpace>::init(std::shared_ptr<
                                          dftfe::basis::
                                            FEBasisOperations<ValueType, double, memorySpace>>
                                                                                basisOperationsMemPtr,
                                        std::shared_ptr<
                                          dftfe::basis::
                                            FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
                                          basisOperationsHostPtr,
                                        std::shared_ptr<
                                          dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                          BLASWrapperMemPtr,
                                        std::shared_ptr<
                                          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                        BLASWrapperHostPtr,
                                        const unsigned int matrixFreeVectorComponent,
                                    const unsigned int                       densityQuadratureId,
                                    const unsigned int sparsityPatternQuadratureId,
                                    const unsigned int numberWaveFunctions,
                                        const unsigned int numSpins,
                                    dftParameters *dftParam,
                                        const std::string &                         scratchFolderName,
                                    const bool                               singlePrecNonLocalOperator,
                                        const bool updateNonlocalSparsity,
                                        const std::vector<std::vector<double>> &atomLocations,
                                        const std::vector<std::vector<double>> &atomLocationsFrac,
                                        const std::vector<int>                 &imageIds,
                                        const std::vector<std::vector<double>> &imagePositions,
                                        std::vector<double>              &kPointCoordinates,
                                        const std::vector<double>  & kPointWeights,
                                        const std::vector <std::vector<double>> &domainBoundaries)
  {
    MPI_Barrier(d_mpi_comm_parent);
    d_BasisOperatorMemPtr = basisOperationsMemPtr;
    d_BLASWrapperMemPtr = BLASWrapperMemPtr;
    d_BasisOperatorHostPtr = basisOperationsHostPtr;

    d_BLASWrapperHostPtr = BLASWrapperHostPtr;
    d_densityQuadratureId = densityQuadratureId;
    d_dftfeScratchFolderName = scratchFolderName;
    d_kPointWeights = kPointWeights;

    d_numberWaveFunctions = numberWaveFunctions;
    d_dftParamsPtr = dftParam;


    d_kPointCoordinates = kPointCoordinates;
    d_numKPoints = kPointCoordinates.size()/3;
    d_domainBoundaries = domainBoundaries;


    d_numSpins = numSpins;
    readHubbardInput(atomLocations, imageIds, imagePositions);

    createAtomCenteredSphericalFunctionsForProjectors();

    d_atomicProjectorFnsContainer =
      std::make_shared<AtomCenteredSphericalFunctionContainer>();

    d_atomicProjectorFnsContainer->init(d_mapAtomToAtomicNumber, d_atomicProjectorFnsMap);

    d_nonLocalOperator = std::make_shared<
      AtomicCenteredNonLocalOperator<ValueType, memorySpace>>(
      d_BLASWrapperMemPtr,
      d_BasisOperatorMemPtr,
      d_atomicProjectorFnsContainer,
      d_mpi_comm_parent);


    d_atomicProjectorFnsContainer->initaliseCoordinates(d_atomicCoords,
                                                        d_periodicImagesCoords,
                                                        d_imageIds);

    d_atomicProjectorFnsContainer->computeSparseStructure(
      d_BasisOperatorHostPtr, sparsityPatternQuadratureId, 1E-8, 0);


    d_nonLocalOperator->intitialisePartitionerKPointsAndComputeCMatrixEntries(
      updateNonlocalSparsity,
      kPointWeights,
      kPointCoordinates,
      d_BasisOperatorHostPtr,
      densityQuadratureId); // TODO check if this is correct

    MPI_Barrier(d_mpi_comm_domain);
    double endRead = MPI_Wtime();


    d_spinPolarizedFactor =
      (d_dftParamsPtr->spinPolarized == 1) ? 1.0 : 2.0;

    d_noOfSpin =
      (d_dftParamsPtr->spinPolarized == 1) ? 2 : 1 ;

    unsigned int numLocalAtomsInProc = d_nonLocalOperator->getTotalAtomInCurrentProcessor();

    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    d_numTotalOccMatrixEntriesPerSpin = 0;
    d_OccMatrixEntryStartForAtom.resize(0);

    for (int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
      {
        const unsigned int atomId = atomIdsInProc[iAtom];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];

        d_OccMatrixEntryStartForAtom.push_back(d_numTotalOccMatrixEntriesPerSpin);
        d_numTotalOccMatrixEntriesPerSpin += d_hubbardSpeciesData[hubbardIds].numberSphericalFuncSq;
      }
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> occIn, occOut, occResidual;
    occIn.resize(d_numSpins*d_numTotalOccMatrixEntriesPerSpin);
    std::fill(occIn.begin(),occIn.end(),0.0);
    d_occupationMatrix[HubbardOccFieldType::In] = occIn;

    occOut.resize(d_numSpins*d_numTotalOccMatrixEntriesPerSpin);
    std::fill(occOut.begin(),occOut.end(),0.0);
    d_occupationMatrix[HubbardOccFieldType::Out] = occOut;

    occResidual.resize(d_numSpins*d_numTotalOccMatrixEntriesPerSpin);
    std::fill(occResidual.begin(),occResidual.end(),0.0);
    d_occupationMatrix[HubbardOccFieldType::Residual] = occResidual;

    setInitialOccMatrx();

// TODO commented for now. Uncomment if necessary
//    computeSymmetricTransforms(atomLocationsFrac,domainBoundaries);


    std::vector<unsigned int> atomProcessorMap;
    unsigned int numAtoms = atomLocations.size();
    atomProcessorMap.resize(numAtoms);

    int thisRank = dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_domain);
    const unsigned int nRanks =
      dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_domain);

    for( unsigned int iAtom = 0; iAtom  < numAtoms; iAtom++)
      {
        atomProcessorMap[iAtom] = nRanks;
      }

    for(int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
         {
           const unsigned int atomId = atomIdsInProc[iAtom];
           atomProcessorMap[atomId] = thisRank;
         }
    MPI_Allreduce(MPI_IN_PLACE,
              &atomProcessorMap[0],
              numAtoms,
              MPI_UNSIGNED,
              MPI_MIN,
              d_mpi_comm_domain);

        d_procLocalAtomId.resize(0);

        for(int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
          {
            const unsigned int atomId = atomIdsInProc[iAtom];
            if ( thisRank == atomProcessorMap[atomId])
              {
                d_procLocalAtomId.push_back(iAtom);
              }
          }

        std::cout<<" rank = "<<thisRank<<"numTotalAtom = "<<numLocalAtomsInProc<<" numAtoms = "<<d_procLocalAtomId.size()<<"\n";
        for (unsigned int iAtom  = 0; iAtom < d_procLocalAtomId.size(); iAtom++)
          {
            std::cout<<" iAtom = "<<iAtom<<" procLocal = "<<d_procLocalAtomId[iAtom]<<"\n";
          }
//    pcout<<" init var = "<<endVarInit-startInit<<" shape func = "<<endShape-endVarInit<<" read atom = "<<endRead-endShape<<" compute atom = "<<endAtom-endRead<<" disc atom = "<<endInit-endAtom<<"\n";
//    pcout<<" Total time for init = "<<endInit-startInit<<"\n";

  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
	  void hubbard<ValueType, memorySpace>::setInitialOccMatrx()
  {

	  unsigned int numLocalAtomsInProc = d_nonLocalOperator->getTotalAtomInCurrentProcessor();
	   const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    for( unsigned int iSpin = 0; iSpin < d_numSpins ; iSpin++)
      {
        for (int iAtom = 0; iAtom < numLocalAtomsInProc; iAtom++)
          {
            const unsigned int atomId = atomIdsInProc[iAtom];
            const unsigned int Znum   = atomicNumber[atomId];
            const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];
            double initOccValue = 0.0;
	    if ( d_numSpins ==1)
	    {
		    initOccValue = d_hubbardSpeciesData[hubbardIds].initialOccupation/(2.0* d_hubbardSpeciesData[hubbardIds].numberSphericalFunc);
	    }
	    else if (d_numSpins == 2)
	    {
		    if ( iSpin == 0)
		    {
			    if ( d_hubbardSpeciesData[hubbardIds].numberSphericalFunc < d_hubbardSpeciesData[hubbardIds].initialOccupation)
			    {
				    initOccValue = 1.0;
			    }
			    else
			    {
				    initOccValue = d_hubbardSpeciesData[hubbardIds].initialOccupation/(d_hubbardSpeciesData[hubbardIds].numberSphericalFunc);
			    }
		    }
		    else if (iSpin == 1)
		    {
			    if ( d_hubbardSpeciesData[hubbardIds].numberSphericalFunc < d_hubbardSpeciesData[hubbardIds].initialOccupation)
			    {
				    initOccValue = (d_hubbardSpeciesData[hubbardIds].initialOccupation - d_hubbardSpeciesData[hubbardIds].numberSphericalFunc) / (d_hubbardSpeciesData[hubbardIds].numberSphericalFunc) ; 
			    }
			    else
			    {
				    initOccValue = 0.0; 
			    }

		    }
	    }
	    for( unsigned int iOrb = 0; iOrb < d_hubbardSpeciesData[hubbardIds].numberSphericalFunc; iOrb++)
                {
                  d_occupationMatrix[HubbardOccFieldType::In][iSpin*d_numTotalOccMatrixEntriesPerSpin +
                                     d_OccMatrixEntryStartForAtom[iAtom] +
                                     iOrb* d_hubbardSpeciesData[hubbardIds].numberSphericalFunc + iOrb] = initOccValue;
                }
          }
      }


    computeCouplingMatrix();
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
  hubbard<ValueType, memorySpace>::getNonLocalOperator()
  {
    return d_nonLocalOperator;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
double hubbard<ValueType, memorySpace>::computeEnergyFromOccupationMatrix()
  {

    double hubbardEnergy = 0.0, hubbardEnergyCorrection = 0.0;

    d_spinPolarizedFactor =
      (d_dftParamsPtr->spinPolarized == 1) ? 1.0 : 2.0;
    unsigned int numOwnedAtomsInProc = d_procLocalAtomId.size();
    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    for ( unsigned int iAtom = 0; iAtom < numOwnedAtomsInProc; iAtom++)
      {
        const unsigned int atomId = atomIdsInProc[d_procLocalAtomId[iAtom]];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];

        const unsigned int numSphericalFunc =  d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

        for (unsigned int spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
          {
            for (unsigned int iOrb = 0; iOrb < numSphericalFunc ; iOrb++)
              {
                hubbardEnergy += 0.5*d_spinPolarizedFactor*d_hubbardSpeciesData[hubbardIds].hubbardValue*
                                 dftfe::utils::realPart(d_occupationMatrix[HubbardOccFieldType::Out][spinIndex*d_numTotalOccMatrixEntriesPerSpin +
                                                                           d_OccMatrixEntryStartForAtom[d_procLocalAtomId[iAtom]] +
                                                                           iOrb * numSphericalFunc+ iOrb]);


                double occMatrixSq = 0.0;

                for (unsigned int jOrb = 0; jOrb < numSphericalFunc ; jOrb++)
                  {
                    unsigned int index1 = iOrb*numSphericalFunc + jOrb;
                    unsigned int index2 = jOrb*numSphericalFunc + iOrb;

                    occMatrixSq += dftfe::utils::realPart(d_occupationMatrix[HubbardOccFieldType::Out][spinIndex*d_numTotalOccMatrixEntriesPerSpin +
                                                                             d_OccMatrixEntryStartForAtom[d_procLocalAtomId[iAtom]] +
                                                                             index1] *
                                   d_occupationMatrix[HubbardOccFieldType::Out][spinIndex*d_numTotalOccMatrixEntriesPerSpin +
                                                                             d_OccMatrixEntryStartForAtom[d_procLocalAtomId[iAtom]] +
                                                                             index2]);

                  }
                hubbardEnergy -= 0.5*d_spinPolarizedFactor*d_hubbardSpeciesData[hubbardIds].hubbardValue*dftfe::utils::realPart(occMatrixSq);
                hubbardEnergyCorrection -= 0.5*d_spinPolarizedFactor*d_hubbardSpeciesData[hubbardIds].hubbardValue*dftfe::utils::realPart(occMatrixSq);
              }
          }


      }

    MPI_Allreduce(MPI_IN_PLACE,
                  &hubbardEnergy,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);

    MPI_Allreduce(MPI_IN_PLACE,
                  &hubbardEnergyCorrection,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  d_mpi_comm_domain);
    pcout<<" Hubbard energy = "<<hubbardEnergy<<"\n";
    pcout<<" Hubbard energy correction = "<<hubbardEnergyCorrection<<"\n";
    return hubbardEnergy;
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void hubbard<ValueType, memorySpace>::computeOccupationMatrix(const dftfe::utils::MemoryStorage<ValueType, memorySpace> *X,
                                                                                        const std::vector<std::vector<double>> &      orbitalOccupancy,
                                                           const std::vector<double> &kPointWeights,
                                                           const std::vector<std::vector<double>> & eigenValues,
                                                           const double fermiEnergy,
                                                           const double fermiEnergyUp,
                                                           const double fermiEnergyDown)
  {
  
    unsigned int numLocalAtomsInProc = d_nonLocalOperator->getTotalAtomInCurrentProcessor();

    const std::vector<unsigned int> atomIdsInProc =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();

    std::fill(d_occupationMatrix[HubbardOccFieldType::Out].begin(),d_occupationMatrix[HubbardOccFieldType::Out].end(),0.0);



    const ValueType zero                    = 0;
    const unsigned int cellsBlockSize = d_BasisOperatorMemPtr->nCells();
     // memorySpace == dftfe::utils::MemorySpace::DEVICE ? 50 : 1;
    const unsigned int totalLocallyOwnedCells = d_BasisOperatorMemPtr->nCells();
    const unsigned int numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

    const unsigned int BVec = d_dftParamsPtr->chebyWfcBlockSize; // TODO extend to band parallelisation

    d_BasisOperatorMemPtr->reinit(BVec, cellsBlockSize, d_densityQuadratureId);
    const unsigned int numQuadPoints = d_BasisOperatorMemPtr->nQuadsPerCell();

    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
                                                         projectorKetTimesVector;

    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
                                                        *flattenedArrayBlock;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      partialOccupVecHost(BVec, 0.0);
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> partialOccupVec(
      partialOccupVecHost.size());
#else
    auto &partialOccupVec = partialOccupVecHost;
#endif

     unsigned int numLocalDofs = d_BasisOperatorHostPtr->nOwnedDofs();
     unsigned int numNodesPerElement = d_BasisOperatorHostPtr->nDofsPerCell();
    dftfe::utils::MemoryStorage<ValueType, memorySpace> tempCellNodalData;
 unsigned int previousSize = 0;
    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
	      //pcout<<" kPoint = "<<kPoint<<" weight = "<<kPointWeights[kPoint]<<"\n";
        for (unsigned int spinIndex = 0; spinIndex < d_noOfSpin;
             ++spinIndex)
          {
            d_nonLocalOperator->initialiseOperatorActionOnX(kPoint);
            for (unsigned int jvec = 0; jvec < d_numberWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, d_numberWaveFunctions - jvec);
                flattenedArrayBlock =
                  &(d_BasisOperatorMemPtr->getMultiVector(currentBlockSize, 0));
                d_nonLocalOperator->initialiseFlattenedDataStructure(
                  currentBlockSize, projectorKetTimesVector);

//                if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
//                  {
//                    d_nonLocalOperator->initialiseCellWaveFunctionPointers(
//                      d_cellWaveFunctionMatrixSrc);
//                  }

//                if ((jvec + currentBlockSize) <=
//                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
//                    (jvec + currentBlockSize) >
//                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                if ( true) /// TODO extend to band parallelisation
                  {
                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          {
                            partialOccupVecHost.data()[iEigenVec] =
				    orbitalOccupancy[kPoint][d_numberWaveFunctions *
                                                      spinIndex +
                                                    jvec + iEigenVec] * kPointWeights[kPoint];
                           //pcout<<" iWave = "<<iEigenVec<<" orb occ in hubb = "<<orbitalOccupancy[kPoint][d_numberWaveFunctions *
                             //                         spinIndex +
                               //                     jvec + iEigenVec]<<"\n";
			  }

                    if (memorySpace == dftfe::utils::MemorySpace::HOST)
                      for (unsigned int iNode = 0; iNode < numLocalDofs;
                           ++iNode)
                        std::memcpy(flattenedArrayBlock->data() +
                                      iNode * currentBlockSize,
                                    X->data() +
                                      numLocalDofs * d_numberWaveFunctions *
                                        (d_noOfSpin * kPoint +
                                         spinIndex) +
                                      iNode * d_numberWaveFunctions + jvec,
                                    currentBlockSize * sizeof(ValueType));
#if defined(DFTFE_WITH_DEVICE)
                    else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                      d_BLASWrapperMemPtr->stridedCopyToBlockConstantStride(
                          currentBlockSize,
                          d_numberWaveFunctions,
                          numLocalDofs,
                          jvec,
                          X->data() +
                            numLocalDofs * d_numberWaveFunctions *
                              (d_noOfSpin * kPoint + spinIndex),
                          flattenedArrayBlock->data());
#endif
                    d_BasisOperatorMemPtr->reinit(currentBlockSize,
                                               cellsBlockSize,
                                                  d_densityQuadratureId,
                                               false);
//                    for ( unsigned int iNode = 0 ; iNode < currentBlockSize*numLocalDofs ; iNode++)
//                      {
//                        flattenedArrayBlock->data()[iNode] = 1.0;
//                      }

                    flattenedArrayBlock->updateGhostValues();
                    d_BasisOperatorMemPtr->distribute(*(flattenedArrayBlock));


                    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                      {
                        const unsigned int currentCellsBlockSize =
                          (iblock == numCellBlocks) ? remCellBlockSize :
                                                      cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const unsigned int startingCellId =
                              iblock * cellsBlockSize;
                            if (currentCellsBlockSize * currentBlockSize !=
                                previousSize)
                              {
                                tempCellNodalData.resize(currentCellsBlockSize *
                                                         currentBlockSize *
                                                         numNodesPerElement);
                                previousSize =
                                  currentCellsBlockSize * currentBlockSize;
                              }
                            d_BasisOperatorMemPtr->extractToCellNodalDataKernel(
                              *(flattenedArrayBlock),
                              tempCellNodalData.data(),
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));

//                            for( unsigned int iWave = 0; iWave <
//                                                         currentCellsBlockSize *  currentBlockSize * numNodesPerElement; iWave++)
//                              {
//                                tempCellNodalData.data()[iWave] = 1.0;
//                              }

                            d_nonLocalOperator->applyCconjtransOnX(
                              tempCellNodalData.data(),
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));
                          }
                      }
                  }
                projectorKetTimesVector.setValue(0.0);
                d_nonLocalOperator
                  ->applyAllReduceOnCconjtransX(
                    projectorKetTimesVector);
		partialOccupVec.copyFrom(partialOccupVecHost);
                d_nonLocalOperator
                  ->copyBackFromDistributedVectorToLocalDataStructure(
                    projectorKetTimesVector, partialOccupVec);
                computeHubbardOccNumberFromCTransOnX(true,
                                                     currentBlockSize,
                                                     spinIndex,
                                                     kPoint);
              }

          }
      }

    unsigned int numOwnedAtomsInProc = d_procLocalAtomId.size();
    for ( unsigned int iAtom = 0; iAtom < numOwnedAtomsInProc; iAtom++)
      {
        const unsigned int atomId = atomIdsInProc[d_procLocalAtomId[iAtom]];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];

        const unsigned int numSphericalFunc =  d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

        for (unsigned int spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
          {
            std::cout<<" occ number for atom = "<<atomId<<" spin = "<<spinIndex<<"\n";
            for (unsigned int iOrb = 0; iOrb < numSphericalFunc; iOrb++)
              {
                for (unsigned int jOrb = 0; jOrb < numSphericalFunc; jOrb++)
                  {
                    std::cout<<" "<<d_occupationMatrix[HubbardOccFieldType::Out][spinIndex*d_numTotalOccMatrixEntriesPerSpin +
                                                           d_OccMatrixEntryStartForAtom[d_procLocalAtomId[iAtom]] +
                                                           iOrb * numSphericalFunc+ jOrb];
                  }
                std::cout<<"\n";
              }
          }
      }

    if (dealii::Utilities::MPI::n_mpi_processes(d_mpi_comm_interPool) > 1)
      {
            MPI_Allreduce(MPI_IN_PLACE,
                          d_occupationMatrix[HubbardOccFieldType::Out].data(),
                      d_numSpins*d_numTotalOccMatrixEntriesPerSpin,
                          MPI_DOUBLE,
                          MPI_SUM,
                      d_mpi_comm_interPool);

      }

    computeResidualOccMat();

  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void
  hubbard<ValueType, memorySpace>::compuseteHubbardOccNumberFromCTransOnX(
    const bool         isOccOut,
    const unsigned int vectorBlockSize,
    const unsigned int spinIndex,
    const unsigned int kpointIndex)
  {
    const std::vector<unsigned int> atomIdsInProcessor =
      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
    std::vector<unsigned int> atomicNumber =
      d_atomicProjectorFnsContainer->getAtomicNumbers();
    char transB = 'N';
#ifdef USE_COMPLEX
    char transA = 'C';
#else
    char transA = 'T';
#endif
    const ValueType beta  = 0.0;
    const ValueType alpha = 1.0;
    for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
      {

        const unsigned int atomId = atomIdsInProcessor[iAtom];
        const unsigned int Znum   = atomicNumber[atomId];
        const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];
        const unsigned int numberSphericalFunctionsSq =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFuncSq;

        const unsigned int numberSphericalFunctions =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;

        const unsigned int numberSphericalFunc =
          d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;
//        if (startVectorIndex == 0)
//          {
//            D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In][atomId] =
//              std::vector<double>(numberSphericalFunctions *
//                                    numberSphericalFunctions,
//                                  0.0);
//          }
        std::vector<ValueType> tempOccMat(numberSphericalFunctionsSq, 0.0);

//        if (d_verbosity >= 5)
//          {
//            std::cout << "U Matrix Entries" << std::endl;
//            for (int i = 0; i < numberSphericalFunctions * vectorBlockSize; i++)
//              std::cout << *(d_nonLocalOperator
//                               ->getCconjtansXLocalDataStructure(atomId) +
//                             i)
//                        << std::endl;
//          }
        d_BLASWrapperHostPtr->xgemm(
          transA,
          transB,
          numberSphericalFunc,
          numberSphericalFunc,
          vectorBlockSize,
          &alpha,
          d_nonLocalOperator->getCconjtansXLocalDataStructure(iAtom),
          vectorBlockSize,
          d_nonLocalOperator->getCconjtansXLocalDataStructure(iAtom),
          vectorBlockSize,
          &beta,
          &tempOccMat[0],
          numberSphericalFunc);
/*
        std::cout<<" tempOccMat\n";
        for( unsigned int iWave = 0; iWave < numberSphericalFunc*numberSphericalFunc; iWave++)
          {
            std::cout<<" iWave = "<<iWave<<" tempOccMat = "<<tempOccMat[iWave] <<"\n";
          }
*/
        std::transform(
          d_occupationMatrix[HubbardOccFieldType::Out].data() + spinIndex*d_numTotalOccMatrixEntriesPerSpin
          + d_OccMatrixEntryStartForAtom[iAtom],
          d_occupationMatrix[HubbardOccFieldType::Out].data() + spinIndex*d_numTotalOccMatrixEntriesPerSpin
            + d_OccMatrixEntryStartForAtom[iAtom] +
            numberSphericalFunctions * numberSphericalFunctions,
          tempOccMat.data(),
          d_occupationMatrix[HubbardOccFieldType::Out].data() + spinIndex*d_numTotalOccMatrixEntriesPerSpin
            + d_OccMatrixEntryStartForAtom[iAtom] ,
          [](auto &p, auto &q) { return p + dftfe::utils::realPart(q); });
        // pcout << "DEBUG: PAW Dij size: "
        //       << D_ij[isDijOut ? TypeOfField::Out : TypeOfField::In].size()
        //       << std::endl;
      }
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void hubbard<ValueType, memorySpace>::computeResidualOccMat()
  {
    for( unsigned int iElem = 0; iElem < d_numSpins*d_numTotalOccMatrixEntriesPerSpin; iElem++)
      {
        d_occupationMatrix[HubbardOccFieldType::Residual][iElem] = d_occupationMatrix[HubbardOccFieldType::Out][iElem] -
                                                                   d_occupationMatrix[HubbardOccFieldType::In][iElem];
      }

  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
  hubbard<ValueType, memorySpace>::getOccMatIn()
  {
    return d_occupationMatrix[HubbardOccFieldType::In];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
  hubbard<ValueType, memorySpace>::getOccMatRes()
  {
    return d_occupationMatrix[HubbardOccFieldType::Residual];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
  hubbard<ValueType, memorySpace>::getOccMatOut()
  {
    return d_occupationMatrix[HubbardOccFieldType::Out];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void hubbard<ValueType, memorySpace>::
    setInOccMatrix(const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> & inputOccMatrix)
  {
    for( unsigned int iElem = 0; iElem < d_numSpins*d_numTotalOccMatrixEntriesPerSpin; iElem++)
      {
        d_occupationMatrix[HubbardOccFieldType::In][iElem] = inputOccMatrix[iElem];
      }

    computeCouplingMatrix();
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
  hubbard<ValueType, memorySpace>::getCouplingMatrix(unsigned int spinIndex)
  {
    return d_couplingMatrixEntries[spinIndex];
  }

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
void
  hubbard<ValueType, memorySpace>::computeCouplingMatrix()
  {
    d_couplingMatrixEntries.resize(d_numSpins);
    for (unsigned int spinIndex = 0; spinIndex < d_numSpins; spinIndex++)
      {
        std::vector<ValueType>          Entries;
        const std::vector<unsigned int> atomIdsInProcessor =
          d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
        std::vector<unsigned int> atomicNumber =
          d_atomicProjectorFnsContainer->getAtomicNumbers();
        d_couplingMatrixEntries[spinIndex].clear();

        for (int iAtom = 0; iAtom < atomIdsInProcessor.size(); iAtom++)
          {
            const unsigned int atomId     = atomIdsInProcessor[iAtom];
            const unsigned int Znum       = atomicNumber[atomId];
            const unsigned int hubbardIds = d_mapAtomToHubbardIds[atomId];
            const unsigned int numberSphericalFunctions =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFunc;
            const unsigned int numberSphericalFunctionsSq =
              d_hubbardSpeciesData[hubbardIds].numberSphericalFuncSq;

            std::vector<ValueType> V(numberSphericalFunctions *
                                     numberSphericalFunctions);
            std::fill(V.begin(), V.end(), 0.0);

            for (unsigned int iOrb = 0; iOrb < numberSphericalFunctions; iOrb++)
              {
                V[iOrb * numberSphericalFunctions + iOrb] =
                  0.5 * d_hubbardSpeciesData[hubbardIds].hubbardValue;

                for (unsigned int jOrb = 0; jOrb < numberSphericalFunctions;
                     jOrb++)
                  {
                    unsigned int index1 =
                      iOrb * numberSphericalFunctions + jOrb;
                    unsigned int index2 =
                      jOrb * numberSphericalFunctions + iOrb;
                    V[iOrb * numberSphericalFunctions + jOrb] -=
                      0.5 * (d_hubbardSpeciesData[hubbardIds].hubbardValue) *
                      (d_occupationMatrix[HubbardOccFieldType::In]
                                         [spinIndex *
                                            d_numTotalOccMatrixEntriesPerSpin +
                                          d_OccMatrixEntryStartForAtom[iAtom] +
                                          index1] +
                       d_occupationMatrix[HubbardOccFieldType::In]
                                         [spinIndex *
                                            d_numTotalOccMatrixEntriesPerSpin +
                                          d_OccMatrixEntryStartForAtom[iAtom] +
                                          index2]);
                  }
              }

            for (unsigned int iOrb = 0;
                 iOrb < numberSphericalFunctions * numberSphericalFunctions;
                 iOrb++)
              {
                Entries.push_back(V[iOrb]);
              }
          }

        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
          {
            d_couplingMatrixEntries.resize(Entries.size());
            d_couplingMatrixEntries.copyFrom(Entries);
          }
#if defined(DFTFE_WITH_DEVICE)
        else
          {
            std::vector<ValueType> EntriesPadded;
            d_nonLocalOperator->paddingCouplingMatrix(Entries,
                                                      EntriesPadded,
                                                      CouplingStructure::dense);
            d_couplingMatrixEntries[spinIndex].resize(EntriesPadded.size());
            d_couplingMatrixEntries[spinIndex].copyFrom(EntriesPadded);
          }
#endif
      }
  }


  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void hubbard<ValueType, memorySpace>::readHubbardInput( const std::vector<std::vector<double>> &atomLocations,
                                                    const std::vector<int>                 &imageIds,
                                                    const std::vector<std::vector<double>> &imagePositions)
  {

    std::ifstream hubbardInputFile(d_dftParamsPtr->hubbardFileName);

    unsigned int numberOfSpecies;
    hubbardInputFile >> numberOfSpecies;
    d_noSpecies = numberOfSpecies - 1; // 0 is default species corresponding to no hubbard correction

    unsigned int id, numberOfProjectors, atomicNumber;
    double hubbardValue;
    unsigned int numOfOrbitals;
    hubbardInputFile >> id >> numOfOrbitals; // reading for 0
    int n,l;
    double initialOccupation;

    for (unsigned int i = 1 ; i< numberOfSpecies; i++)
      {
        hubbardInputFile >> id >>atomicNumber >> hubbardValue>> numberOfProjectors >> initialOccupation;

        pcout<<" id = "<<id<<"\n";

        pcout<<" atomicNumber = "<<atomicNumber<<"\n";
        pcout<<" hubbardValue = "<<hubbardValue<<"\n";
        pcout<<" numberOfProjectors = "<<numberOfProjectors<<"\n";
	pcout<<" initialOccupation = "<<initialOccupation<<"\n";
        hubbardSpecies hubbardSpeciesObj;

        hubbardSpeciesObj.hubbardValue = hubbardValue;
        hubbardSpeciesObj.numProj = numberOfProjectors;
        hubbardSpeciesObj.atomicNumber = atomicNumber;
        hubbardSpeciesObj.nQuantumNum.resize(numberOfProjectors);
        hubbardSpeciesObj.lQuantumNum.resize(numberOfProjectors);
	hubbardSpeciesObj.initialOccupation = initialOccupation;
        hubbardSpeciesObj.numberSphericalFunc = 0;
        for( unsigned int orbitalId = 0 ; orbitalId<numberOfProjectors;orbitalId++ )
          {
            hubbardInputFile >> n >> l;
            hubbardSpeciesObj.nQuantumNum[orbitalId] = n;
            hubbardSpeciesObj.lQuantumNum[orbitalId] = l;

            pcout<<" n = "<<n<<"\n";
            pcout<<" l = "<<l<<"\n";

            hubbardSpeciesObj.numberSphericalFunc += 2*l+1;
          }

        hubbardSpeciesObj.numberSphericalFuncSq = hubbardSpeciesObj.numberSphericalFunc*
                                                  hubbardSpeciesObj.numberSphericalFunc;
        d_hubbardSpeciesData[id-1] = hubbardSpeciesObj;
      }

    std::vector<std::vector<unsigned int>> mapAtomToImageAtom;
    mapAtomToImageAtom.resize(atomLocations.size());

    for ( unsigned int iAtom = 0; iAtom < atomLocations.size();iAtom++)
      {
        mapAtomToImageAtom[iAtom].resize(0,0);
      }
    for(unsigned int imageIdIter = 0; imageIdIter< imageIds.size();imageIdIter++)
      {
        mapAtomToImageAtom[imageIds[imageIdIter]].push_back(imageIdIter);
      }

    std::vector<double> atomCoord;
    atomCoord.resize(3,0.0);

    d_atomicCoords.resize(0);
    d_periodicImagesCoords.resize(0);
    d_imageIds.resize(0);
    d_mapAtomToHubbardIds.resize(0);
    d_mapAtomToAtomicNumber.resize(0);
    unsigned int hubbardAtomId = 0 ;
    unsigned int atomicNum;
    for ( unsigned int iAtom = 0; iAtom < atomLocations.size();iAtom++)
      {
        hubbardInputFile >> atomicNum >> id;
        if( id != 0)
        {
          d_atomicCoords.push_back(atomLocations[iAtom][2]);
          d_atomicCoords.push_back(atomLocations[iAtom][3]);
          d_atomicCoords.push_back(atomLocations[iAtom][4]);
          d_mapAtomToHubbardIds.push_back(id-1);
          d_mapAtomToAtomicNumber.push_back(atomicNum);
          for(unsigned int jImageAtom = 0; jImageAtom < mapAtomToImageAtom[iAtom].size();jImageAtom++)
            {
              atomCoord[0] = imagePositions[mapAtomToImageAtom[iAtom][jImageAtom]][0];
              atomCoord[1] = imagePositions[mapAtomToImageAtom[iAtom][jImageAtom]][1];
              atomCoord[2] = imagePositions[mapAtomToImageAtom[iAtom][jImageAtom]][2];

              d_periodicImagesCoords.push_back(atomCoord);
              d_imageIds.push_back(hubbardAtomId);
            }
          hubbardAtomId++;
        }
      }
    hubbardInputFile.close();
  }


  template class hubbard<dataTypes::number,dftfe::utils::MemorySpace::HOST >;
#if defined(DFTFE_WITH_DEVICE)
  template class hubbard<dataTypes::number,dftfe::utils::MemorySpace::DEVICE >;
#endif

  /*
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  void hubbard<ValueType, memorySpace>::readHubbardInput( const std::vector<std::vector<double>> &atomLocations,
                                                const std::vector<int>                 &imageIds,
                                                const std::vector<std::vector<double>> &imagePositions)
  {

    std::ifstream hubbardInputFile(d_dftParamsPtr->hubbardFileName);

    unsigned int numberOfSpecies;
    hubbardInputFile >> numberOfSpecies;
    d_noSpecies = numberOfSpecies - 1; // 0 is default species corresponding to no hubbard correction

    //TODO set up these properly
    d_atomicCoordinates.resize(d_noSpecies) ;
    d_atomicImageCoordinates.resize(d_noSpecies) ;
    d_orbitalIds.resize(d_noSpecies);
    d_atomicNumber.resize(d_noSpecies);
    d_atomicWavefunctionPerSpecies.resize(d_noSpecies) ;
    d_atomicWavefunctionPerSpeciesSq.resize(d_noSpecies);
    d_hubbardParameter.resize(d_noSpecies) ;
    d_noAtoms.resize(d_noSpecies) ;
    d_speciesImageAtomList.resize(d_noSpecies) ;
    std::fill(d_noAtoms.begin(),d_noAtoms.end(),0);
    startIndexAtomicWavefunction.resize(d_noSpecies) ;

    d_orbital_n_l_Id.resize(d_noSpecies);
    unsigned int id, numOfOrbitals, atomicNumber;
    double hubbardValue;
    hubbardInputFile >> id >> numOfOrbitals; // reading for 0
    int n,l;
    for (unsigned int i = 1 ; i< numberOfSpecies; i++)
      {
        hubbardInputFile >> id >>atomicNumber >> hubbardValue>> numOfOrbitals;
        d_atomicNumber[id-1] = atomicNumber;
        d_hubbardParameter[id-1] = hubbardValue;
        d_orbital_n_l_Id[id-1].resize(numOfOrbitals*2);
        pcout<<" id = "<<id<<"numOfOrbitals = "<<numOfOrbitals<<"\n";
        for( unsigned int orbitalId = 0 ; orbitalId<numOfOrbitals;orbitalId++ )
          {
            hubbardInputFile >> n >> l;
            d_orbital_n_l_Id[id-1][2*orbitalId] = n;
            d_orbital_n_l_Id[id-1][2*orbitalId+1] = l;
            for( int m = -l; m <=l;m++)
              {
                pcout<<" n = "<<n<<" l = "<<l<<" m = "<<m<<"\n";
                OrbitalQuantumNumbers orbitalNumObj;
                orbitalNumObj.nOrbitalNumber = n;
                orbitalNumObj.lOrbitalNumber = l;
                orbitalNumObj.mOrbitalNumber = m;
                d_orbitalIds[id-1].push_back(orbitalNumObj); // I hope this is a copy operation
              }

          }
      }

    // the coordinates list and the hubbard parameters must be same
    unsigned int atomicNum;
    std::vector<double> atomCoord;
    atomCoord.resize(3,0.0);

    //TODO see if you can get this somewhere
    std::vector<std::vector<unsigned int>> mapAtomToImageAtom;
    mapAtomToImageAtom.resize(atomLocations.size());

    for ( unsigned int iAtom = 0; iAtom < atomLocations.size();iAtom++)
      {
        mapAtomToImageAtom[iAtom].resize(0,0);
      }
    for(unsigned int imageIdIter = 0; imageIdIter< imageIds.size();imageIdIter++)
      {
        mapAtomToImageAtom[imageIds[imageIdIter]].push_back(imageIdIter);
      }

    d_mapImageAtomToSpeciesAtomId.resize(imageIds.size()*2,0);

    // TODO account for the fractional coordinates
    d_phaseFactorImageAtom.resize(d_numKPoints);
    for(unsigned int iKPoint = 0; iKPoint  <d_numKPoints; iKPoint++)
      {
        d_phaseFactorImageAtom[iKPoint].resize(imageIds.size());
      }
    for ( unsigned int iAtom = 0; iAtom < atomLocations.size();iAtom++)
      {
        hubbardInputFile >> atomicNum >> id;
        if( id != 0)
          {
            d_noAtoms[id-1]++;
            atomCoord[0] = atomLocations[iAtom][2];
            atomCoord[1] = atomLocations[iAtom][3];
            atomCoord[2] = atomLocations[iAtom][4];

            d_atomicCoordinates[id-1].push_back(atomCoord);

            for(unsigned int jImageAtom = 0; jImageAtom < mapAtomToImageAtom[iAtom].size();jImageAtom++)
              {
                d_speciesImageAtomList[id-1].push_back(mapAtomToImageAtom[iAtom][jImageAtom]);
                unsigned int speciesAtomId = mapAtomToImageAtom[iAtom][jImageAtom];
                atomCoord[0] = imagePositions[speciesAtomId][0];
                atomCoord[1] = imagePositions[speciesAtomId][1];
                atomCoord[2] = imagePositions[speciesAtomId][2];
                d_atomicImageCoordinates[id-1].push_back(atomCoord);
                d_mapImageAtomToSpeciesAtomId[2*mapAtomToImageAtom[iAtom][jImageAtom]] = id-1; //species index
                d_mapImageAtomToSpeciesAtomId[2*mapAtomToImageAtom[iAtom][jImageAtom] + 1] = d_noAtoms[id-1] -1; // atom index

                for(unsigned int iKPoint = 0; iKPoint  <d_numKPoints; iKPoint++)
                  {
                    double angle = (atomLocations[iAtom][2] - imagePositions[speciesAtomId][0] )*d_kPointCoordinates[3*iKPoint+0] ;
                    angle       += (atomLocations[iAtom][3] - imagePositions[speciesAtomId][1] )*d_kPointCoordinates[3*iKPoint+1] ;
                    angle       += (atomLocations[iAtom][4] - imagePositions[speciesAtomId][2] )*d_kPointCoordinates[3*iKPoint+2] ;

                    copyComlexValueCompatibly(std::exp(std::complex<double>(0,-angle)), d_phaseFactorImageAtom[iKPoint][speciesAtomId]);
                  }
              }
          }

      }

    hubbardInputFile.close();

    d_totalAtomicWavefunction = 0;

    for ( unsigned int iSpecies = 0 ; iSpecies < d_noSpecies ; iSpecies++)
      {
        // TODO have to read this from the psp file
        d_atomicWavefunctionPerSpecies[iSpecies] = d_orbitalIds[iSpecies].size();
        startIndexAtomicWavefunction[iSpecies] = d_totalAtomicWavefunction;
        //          d_noAtoms[iSpecies] = d_atomicCoordinates[iSpecies].size();
        d_totalAtomicWavefunction += (d_noAtoms[iSpecies]*d_atomicWavefunctionPerSpecies[iSpecies]);
        d_atomicWavefunctionPerSpeciesSq[iSpecies] =  d_atomicWavefunctionPerSpecies[iSpecies]*d_atomicWavefunctionPerSpecies[iSpecies];

      }

  }
  */
//
//  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
//  void hubbard<ValueType, memorySpace>::computeAtomicData(const std::vector<std::vector<double>> &imagePositions)
//  {
//    std::vector<distributedCPUMultiVec<ValueType>> atomicWaveNodalData;
//    atomicWaveNodalData.resize(d_noSpecies);
//
//    //std::map< dealii::types::global_dof_index, dealii::Point<3,double>>dof_coord ;
//    //dealii::DoFTools::map_dofs_to_support_points<3,3>(dealii::MappingQ1<3,3>() ,*d_dofHandler,dof_coord);
//    //dealii::types::global_dof_index numberDofs = d_dofHandler->n_dofs();
//    atomicWaveQuadDataJxW.resize(d_numKPoints);
//    for(unsigned int iKPoint = 0; iKPoint  <d_numKPoints; iKPoint++)
//      {
//        atomicWaveQuadDataJxW[iKPoint].resize(d_noSpecies);
//      }
//    //atomicWaveQuadDataJxWOpt.resize(d_noSpecies);
//    //atomicWaveCellWiseQuadDataJxWOpt.resize(d_noSpecies);
//
//
//
//    d_fullFlattenedArrayMacroCellLocalProcIndexIdMapAtomic.resize(d_noSpecies);
//    d_normalCellIdToMacroCellIdMapAtomic.resize(d_noSpecies);
//    d_macroCellIdToNormalCellIdMapAtomic.resize(d_noSpecies);
//    d_fullFlattenedArrayCellLocalProcIndexIdMapAtomic.resize(d_noSpecies);
//    cumulativeAtomicData.resize(d_noSpecies);
//    cumulativeAtomicData[0] = 0;
//    for(unsigned int iSpecies = 1 ; iSpecies <  d_noSpecies; ++iSpecies )
//      {
//        cumulativeAtomicData[iSpecies] = cumulativeAtomicData[iSpecies-1] + d_noAtoms[iSpecies-1]*d_atomicWavefunctionPerSpecies[iSpecies-1];
//      }
//
//    std::vector<double> orbitalOverLap;
//
//    initialOccupation.resize(d_noSpecies);
//    //d_atomicOrbitalOverlapWithCell.resize(d_totalLocallyOwnedCells);
//    //d_cellToAtomicOrbitalOverlap.resize(d_noSpecies);
//    //d_cellToNumAtomicOrbitalOverlap.resize(d_noSpecies);
//    //std::fill(d_atomicOrbitalOverlapWithCell.begin(),d_atomicOrbitalOverlapWithCell.end(),false); //TODO check if this is right
//
//    d_atomicWaveFuncStart.resize(d_noSpecies);
//    d_atomicWaveFuncEnd.resize(d_noSpecies);
//    for(unsigned int iSpecies = 0 ; iSpecies <  d_noSpecies; ++iSpecies )
//      {
//        d_atomicWaveFuncStart[iSpecies].resize(n_mpi_processes);
//        d_atomicWaveFuncEnd[iSpecies].resize(n_mpi_processes);
//
//        unsigned int totalAtomicWavePerSpecies = d_atomicWavefunctionPerSpecies[iSpecies]*
//                                                 d_noAtoms[iSpecies];
//        d_atomicWaveFuncStart[iSpecies][0] = 0;
//        d_atomicWaveFuncEnd[iSpecies][0] = d_atomicWaveFuncStart[iSpecies][0]
//                                           + totalAtomicWavePerSpecies/n_mpi_processes;
//        if(totalAtomicWavePerSpecies%n_mpi_processes >  0)
//          {
//            d_atomicWaveFuncEnd[iSpecies][0] = d_atomicWaveFuncEnd[iSpecies][0] +1;
//          }
//        for(unsigned int iProc = 1; iProc <n_mpi_processes; iProc++)
//          {
//            d_atomicWaveFuncStart[iSpecies][iProc] =  d_atomicWaveFuncEnd[iSpecies][iProc-1];
//            d_atomicWaveFuncEnd[iSpecies][iProc] = d_atomicWaveFuncStart[iSpecies][iProc]
//                                                   + totalAtomicWavePerSpecies/n_mpi_processes ;
//
//            if(totalAtomicWavePerSpecies%n_mpi_processes >  iProc)
//              {
//                d_atomicWaveFuncEnd[iSpecies][iProc] = d_atomicWaveFuncEnd[iSpecies][iProc] +1;
//              }
//          }
//
//        initialOccupation[iSpecies].resize(d_atomicWavefunctionPerSpecies[iSpecies]);
//        //d_cellToAtomicOrbitalOverlap[iSpecies].resize(d_totalLocallyOwnedCells);
//        //d_cellToNumAtomicOrbitalOverlap[iSpecies].resize(d_totalLocallyOwnedCells);
//        for(unsigned int iKPoint = 0; iKPoint  <d_numKPoints; iKPoint++)
//          {
//            atomicWaveQuadDataJxW[iKPoint][iSpecies].resize(d_totalLocallyOwnedCells*
//                                                              d_numberQuadraturePoints*
//                                                              d_atomicWavefunctionPerSpecies[iSpecies]*
//                                                              d_noAtoms[iSpecies], 0.0 );
//
//            std::fill(atomicWaveQuadDataJxW[iKPoint][iSpecies].begin(),atomicWaveQuadDataJxW[iKPoint][iSpecies].end(),0.0);
//          }
//
//
//        //atomicWaveQuadDataJxWOpt[iSpecies].resize(d_totalLocallyOwnedCells*
//        //                                            d_numberQuadraturePoints*
//        //                                            d_atomicWavefunctionPerSpecies[iSpecies]*
//        //                                            d_noAtoms[iSpecies], 0.0 );
//        //atomicWaveCellWiseQuadDataJxWOpt[iSpecies].resize(d_totalLocallyOwnedCells);
//
//        //std::fill(atomicWaveQuadDataJxWOpt[iSpecies].begin(),atomicWaveQuadDataJxWOpt[iSpecies].end(),0.0);
//
//        dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
//          d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
//          d_noAtoms[iSpecies]*d_atomicWavefunctionPerSpecies[iSpecies],
//          atomicWaveNodalData[iSpecies]);
//
//        vectorTools::computeCellLocalIndexSetMap(
//          atomicWaveNodalData[iSpecies].getMPIPatternP2P(),
//          *d_matrixFreeDataPtr,
//          d_matrixFreeVectorComponent,
//          d_atomicWavefunctionPerSpecies[iSpecies]*d_noAtoms[iSpecies],
//          d_fullFlattenedArrayCellLocalProcIndexIdMapAtomic[iSpecies]);
//
//        AtomicOrbitalBasisManager atomicOrbitalObj( d_atomicNumber[iSpecies], // atomic number how do you get this ?
//                                                   false, // normalised ?
//                                                   d_dftParamsPtr,
//                                                   d_orbital_n_l_Id[iSpecies],
//                                                   d_mpi_comm_parent);
//
//
//        dealii::IndexSet  localIndex = d_dofHandler->locally_owned_dofs();
//
//        if( localIndex.is_contiguous() == false)
//          {
//            std::cerr << " Local range is not contiguous " << localIndex.is_contiguous()
//                      << std::endl;
//          }
//
//        // TODO extend it to k points
//        for( unsigned int orbitalId = 0 ; orbitalId < d_atomicWavefunctionPerSpecies[iSpecies]; orbitalId++)
//          {
//            initialOccupation[iSpecies][orbitalId] = 0.5*atomicOrbitalObj.getOccupationForOrbital( d_orbitalIds[iSpecies][orbitalId]);
//          }
//
//        //	for (dealii::types::global_dof_index iNode = 0; iNode < numberDofs; iNode++)
//        //          {
//        //            if (localIndex.is_element(iNode))
//        //              {
//        //                unsigned int localNode = iNode - (*(localIndex.begin()));
//        //                for( unsigned int iWave = 0 ; iWave < d_atomicWavefunctionPerSpecies[iSpecies]*d_noAtoms[iSpecies]; iWave++)
//        //                  {
//        //                    unsigned int atomId = iWave/d_atomicWavefunctionPerSpecies[iSpecies];
//        //                    unsigned int orbitalId = iWave%d_atomicWavefunctionPerSpecies[iSpecies];
//        //                    atomicWaveNodalData[iSpecies].data()[localNode*d_atomicWavefunctionPerSpecies[iSpecies]*d_noAtoms[iSpecies] + iWave] =
//        //                      atomicOrbitalObj.PseudoAtomicOrbitalvalue(
//        //                        d_orbitalIds[iSpecies][orbitalId],
//        //                        dof_coord[iNode],
//        //                        d_atomicCoordinates[iSpecies][atomId]);
//        //                  }
//        //              }
//        //          }
//        //        atomicWaveNodalData[iSpecies].updateGhostValues();
//
//        unsigned int iElem = 0;
//
//        //TODO this is changed to FE +1
//        //    const dealii::Quadrature<3> &quadratureRhs =
//        //      d_matrixFreeDataPtr->get_quadrature(
//        //        d_matrixFreeQuadratureComponent);
//
//        dealii::QGauss<3> quadratureRhs(d_dftParamsPtr->finiteElementPolynomialOrder + 1);
//
//        dealii::FEValues<3> fe_values(d_dofHandler->get_fe(),
//                                      quadratureRhs,
//                                      dealii::update_quadrature_points |
//                                        dealii::update_JxW_values);
//
//        typename dealii::DoFHandler<3>::active_cell_iterator
//          cell    = d_dofHandler->begin_active(),
//          endc = d_dofHandler->end();
//
//        orbitalOverLap.resize(d_atomicWavefunctionPerSpecies[iSpecies]*d_atomicWavefunctionPerSpecies[iSpecies]);
//        std::fill(orbitalOverLap.begin(),orbitalOverLap.end(),0.0);
//        unsigned int elemQuadIndex = 0, elemQPointIndex = 0, quadIndex = 0;
//        unsigned int elemWaveQPointContiguousIndex = 0;
//        for (cell = d_dofHandler->begin_active() ; cell != endc; ++cell)
//          if (cell->is_locally_owned())
//            {
//              //d_cellToAtomicOrbitalOverlap[iSpecies][iElem].resize(0);
//              // get the shortest distance between the cell center and atom
//
//
//              dealii::Point<3,double> center =  cell->center() ;
//
//              // TODO account for fractional coordinates
//              std::vector<unsigned int> atomListInCell;
//              atomListInCell.resize(0);
//              double minDistAtom = 10000;
//              for(unsigned int iAtom = 0; iAtom <d_noAtoms[iSpecies]; iAtom++)
//                {
//                  double atomicDist =
//                    (center[0] - d_atomicCoordinates[iSpecies][iAtom][0]) *
//                      (center[0] - d_atomicCoordinates[iSpecies][iAtom][0]) +
//                    (center[1] - d_atomicCoordinates[iSpecies][iAtom][1]) *
//                      (center[1] - d_atomicCoordinates[iSpecies][iAtom][1]) +
//                    (center[2] - d_atomicCoordinates[iSpecies][iAtom][2]) *
//                      (center[2] - d_atomicCoordinates[iSpecies][iAtom][2]);
//
//                  atomicDist = std::sqrt(atomicDist);
//                  if (atomicDist <
//                      d_atomOrbitalMaxLength)
//                    {
//                      atomListInCell.push_back(iAtom);
//                    }
//                }
//
//              //TODO extend this to multiple species
//              std::vector<unsigned int> imageAtomListInCell;
//              imageAtomListInCell.resize(0);
//              minDistAtom = 10000;
//              for(unsigned int iAtom = 0; iAtom <d_speciesImageAtomList[iSpecies].size(); iAtom++)
//                {
//                  double atomicDist = (center[0] - d_atomicImageCoordinates[iSpecies][iAtom][0])*(center[0] - d_atomicImageCoordinates[iSpecies][iAtom][0]) +
//                                      (center[1] - d_atomicImageCoordinates[iSpecies][iAtom][1])*(center[1] - d_atomicImageCoordinates[iSpecies][iAtom][1]) +
//                                      (center[2] - d_atomicImageCoordinates[iSpecies][iAtom][2])*(center[2] - d_atomicImageCoordinates[iSpecies][iAtom][2]);
//
//                  atomicDist = std::sqrt(atomicDist);
//                  //if( minDistAtom > atomicDist)
//                  //  minDistAtom = atomicDist;
//
//                  if(atomicDist<d_atomOrbitalMaxLength)
//                    {
//                      imageAtomListInCell.push_back(iAtom);
//                    }
//
//                }
//
//              elemQuadIndex = iElem*
//                              d_numberQuadraturePoints*
//                              d_atomicWavefunctionPerSpecies[iSpecies]*
//                              d_noAtoms[iSpecies];
//              fe_values.reinit(cell);
//              /*
//                            for( unsigned int waveId = 0 ; waveId < d_cellToNumAtomicOrbitalOverlap[iSpecies][iElem]; waveId++)
//                              {
//                                unsigned int iWave = d_cellToAtomicOrbitalOverlap[iSpecies][iElem][waveId];
//                                unsigned int atomId = iWave/d_atomicWavefunctionPerSpecies[iSpecies];
//                                unsigned int orbitalId = iWave%d_atomicWavefunctionPerSpecies[iSpecies];
//                                for (unsigned int qPoint = 0; qPoint < d_numberQuadraturePoints;
//                                     qPoint++)
//                                  {
//                                    dealii::Point<3, double> qPointVal =
//                                      fe_values.quadrature_point(qPoint);
//
//                                    atomicWaveCellWiseQuadDataJxWOpt[iSpecies][iElem][waveId*d_numberQuadraturePoints + qPoint]
//                                      =  atomicOrbitalObj.PseudoAtomicOrbitalvalue(
//                                          d_orbitalIds[iSpecies][orbitalId],
//                                          qPointVal,
//                                          d_atomicCoordinates[iSpecies][atomId])
//                                        * fe_values.JxW(qPoint);
//                                  }
//                              }
//              */
//              /*
//                            for( unsigned int iWave = 0 ; iWave < d_atomicWavefunctionPerSpecies[iSpecies]*d_noAtoms[iSpecies]; iWave++)
//                              {
//                                unsigned int atomId = iWave/d_atomicWavefunctionPerSpecies[iSpecies];
//                                unsigned int orbitalId = iWave%d_atomicWavefunctionPerSpecies[iSpecies];
//                                elemWaveQPointContiguousIndex = elemQuadIndex + iWave*d_numberQuadraturePoints;
//                                for (unsigned int qPoint = 0; qPoint < d_numberQuadraturePoints;
//                                     qPoint++)
//                                  {
//                                    dealii::Point<3, double> qPointVal =
//                                      fe_values.quadrature_point(qPoint);
//                                    unsigned int atomicWaveQuadDataJxWOptIndex = elemWaveQPointContiguousIndex + qPoint;
//
//                                    atomicWaveQuadDataJxWOpt[iSpecies][atomicWaveQuadDataJxWOptIndex]
//                                      =  atomicOrbitalObj.PseudoAtomicOrbitalvalue(
//                                                                             d_orbitalIds[iSpecies][orbitalId],
//                                                                             qPointVal,
//                                                                             d_atomicCoordinates[iSpecies][atomId])
//                                        * fe_values.JxW(qPoint);
//                                  }
//                              }
//
//              */
//              for (unsigned int qPoint = 0; qPoint < d_numberQuadraturePoints;
//                   qPoint++)
//                {
//                  elemQPointIndex = elemQuadIndex +
//                                    qPoint*
//                                      d_atomicWavefunctionPerSpecies[iSpecies]*
//                                      d_noAtoms[iSpecies];
//                  dealii::Point<3, double> qPointVal =
//                    fe_values.quadrature_point(qPoint);
//
//                  double JxWValue = fe_values.JxW(qPoint);
//                  //for( unsigned int iWave = 0 ; iWave < d_atomicWavefunctionPerSpecies[iSpecies]*d_noAtoms[iSpecies]; iWave++)
//                  //  {
//                  //    unsigned int atomId = iWave/d_atomicWavefunctionPerSpecies[iSpecies];
//                  //    unsigned int orbitalId = iWave%d_atomicWavefunctionPerSpecies[iSpecies];
//                  //    quadIndex = elemQPointIndex + iWave;
//                  //    atomicWaveQuadDataJxW[iSpecies][quadIndex] =
//                  //      atomicOrbitalObj.PseudoAtomicOrbitalvalue(
//                  //        d_orbitalIds[iSpecies][orbitalId],
//                  //        qPointVal,
//                  //        d_atomicCoordinates[iSpecies][atomId]) * JxWValue;
//                  //
//                  //  }
//
//                  std::vector<ValueType> phaseFactorQuadPoint;
//                  phaseFactorQuadPoint.resize(d_numKPoints);
//                  for(unsigned int iKPoint = 0; iKPoint  <d_numKPoints; iKPoint++)
//                    {
//                      double angle = d_kPointCoordinates[3*iKPoint +0]*qPointVal[0];
//                      angle       += d_kPointCoordinates[3*iKPoint +1]*qPointVal[1];
//                      angle       += d_kPointCoordinates[3*iKPoint +2]*qPointVal[2];
//                      copyComlexValueCompatibly(std::exp(std::complex<double>(0,-angle)),phaseFactorQuadPoint[iKPoint]);
//                    }
//
//                  for(unsigned int atomIndex =0; atomIndex <  atomListInCell.size();atomIndex++)
//                    {
//                      unsigned int atomId = atomListInCell[atomIndex];
//
//                      int n = d_orbitalIds[iSpecies][0].nOrbitalNumber;
//                      int l = d_orbitalIds[iSpecies][0].lOrbitalNumber;
//                      int m = d_orbitalIds[iSpecies][0].mOrbitalNumber;
//
//                      double r{}, theta{}, phi{};
//
//                      auto relativeEvalPoint = relativeVector3d(qPointVal, d_atomicCoordinates[iSpecies][atomId]);
//
//                      convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);
//
//                      double radialPart = atomicOrbitalObj.RadialPseudoAtomicOrbital(n, l, r);
//
//                      for(unsigned int orbitalId = 0; orbitalId<d_atomicWavefunctionPerSpecies[iSpecies]; orbitalId++)
//                        {
//                          m = d_orbitalIds[iSpecies][orbitalId].mOrbitalNumber;
//                          quadIndex = elemQPointIndex + atomId*d_atomicWavefunctionPerSpecies[iSpecies]+orbitalId;
//                          double harmonicPart = atomicOrbitalObj.realSphericalHarmonics(l, m, theta, phi);
//
//                          double atomicWaveFunctionVal = radialPart*harmonicPart;
//                          for(unsigned int iKPoint = 0; iKPoint  <d_numKPoints; iKPoint++)
//                            {
//                              atomicWaveQuadDataJxW[iKPoint][iSpecies][quadIndex] =
//                                atomicWaveFunctionVal *
//                                phaseFactorQuadPoint[iKPoint] *
//                                JxWValue;
//                            }
//                        }
//                    }
//
//                  for(unsigned int atomIndex =0; atomIndex <  imageAtomListInCell.size();atomIndex++)
//                    {
//                      unsigned int atomId = d_mapImageAtomToSpeciesAtomId[2*imageAtomListInCell[d_speciesImageAtomList[iSpecies][atomIndex]] + 1]; // atom id
//                      unsigned int speciesAtomId = imageAtomListInCell[d_speciesImageAtomList[iSpecies][atomIndex]];
//
//                      int n = d_orbitalIds[iSpecies][0].nOrbitalNumber;
//                      int l = d_orbitalIds[iSpecies][0].lOrbitalNumber;
//                      int m = d_orbitalIds[iSpecies][0].mOrbitalNumber;
//
//                      double r{}, theta{}, phi{};
//
//                      auto relativeEvalPoint = relativeVector3d(qPointVal, imagePositions[speciesAtomId]);
//
//                      convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);
//
//                      double radialPart = atomicOrbitalObj.RadialPseudoAtomicOrbital(n, l, r);
//
//                      for(unsigned int orbitalId = 0; orbitalId<d_atomicWavefunctionPerSpecies[iSpecies]; orbitalId++)
//                        {
//                          m = d_orbitalIds[iSpecies][orbitalId].mOrbitalNumber;
//                          double harmonicPart = atomicOrbitalObj.realSphericalHarmonics(l, m, theta, phi);
//
//                          double atomicWaveFunctionVal = radialPart*harmonicPart;
//
//                          quadIndex = elemQPointIndex + atomId*d_atomicWavefunctionPerSpecies[iSpecies]+orbitalId;
//                          for(unsigned int iKPoint = 0; iKPoint  <d_numKPoints; iKPoint++)
//                            {
//                              atomicWaveQuadDataJxW[iKPoint][iSpecies][quadIndex] +=
//                                atomicWaveFunctionVal *
//                                phaseFactorQuadPoint[iKPoint] *
//                                d_phaseFactorImageAtom[iKPoint][speciesAtomId] *
//                                JxWValue;
//                            }
//                        }
//                    }
//                }
//              iElem++;
//            }
//      }
//  }
//*/
//  template < dftfe::utils::MemorySpace memorySpace>
//  void hubbard<ValueType, memorySpace>::computeSymmetricTransforms(const std::vector<std::vector<double>> &atomLocationsFractional,
//                                                          const std::vector <std::vector<double>> &domainBoundaries)
//  {
//    const int                                     max_size = 500;
//    int                                           rotation[max_size][3][3];
//    double                                        translation[max_size][3];
//    //
//
//    if (d_dftParamsPtr->useSymm || !d_dftParamsPtr->useSymm)
//      {
//        const int num_atom = atomLocationsFractional.size();
//        double    lattice[3][3], position[num_atom][3];
//        int       types[num_atom];
//
//        for (unsigned int i = 0; i < 3; ++i)
//          {
//            for (unsigned int j = 0; j < 3; ++j)
//              lattice[i][j] = domainBoundaries[i][j];
//          }
//        for (unsigned int i = 0; i < num_atom; ++i)
//          {
//            types[i] = atomLocationsFractional[i][0];
//            for (unsigned int j = 0; j < 3; ++j)
//              position[i][j] = atomLocationsFractional[i][j + 2];
//          }
//        //
//        if (!d_dftParamsPtr->reproducible_output)
//          pcout << " getting space group symmetries from spg " << std::endl;
//        d_numSymm = spg_get_symmetry(rotation,
//                                     translation,
//                                     max_size,
//                                     lattice,
//                                     position,
//                                     types,
//                                     num_atom,
//                                     1e-5);
//        if (!d_dftParamsPtr->reproducible_output &&
//            d_dftParamsPtr->verbosity > 3)
//          {
//            pcout << " number of symmetries allowed for the lattice "
//                  << d_numSymm << std::endl;
//            for (unsigned int iSymm = 0; iSymm < d_numSymm;
//                 ++iSymm)
//              {
//                pcout << " Symmetry " << iSymm + 1 << std::endl;
//                pcout << " Rotation " << std::endl;
//                for (unsigned int ipol = 0; ipol < 3; ++ipol)
//                  pcout << rotation[iSymm][ipol][0] << "  "
//                        << rotation[iSymm][ipol][1] << "  "
//                        << rotation[iSymm][ipol][2] << std::endl;
//                pcout << " translation " << std::endl;
//                pcout << translation[iSymm][0] << "  "
//                      << translation[iSymm][1] << "  "
//                      << translation[iSymm][2] << std::endl;
//                pcout << "	" << std::endl;
//              }
//          }
//      }
//    if (d_dftParamsPtr->timeReversal)
//      {
//        for (unsigned int iSymm = d_numSymm;
//             iSymm < 2 * d_numSymm;
//             ++iSymm)
//          {
//            for (unsigned int j = 0; j < 3; ++j)
//              {
//                for (unsigned int k = 0; k < 3; ++k)
//                  rotation[iSymm][j][k] =
//                    -1 * rotation[iSymm - d_numSymm][j][k];
//                translation[iSymm][j] =
//                  translation[iSymm - d_numSymm][j];
//              }
//          }
//        d_numSymm = 2 * d_numSymm;
//      }
//
//    // generate 5 random points
//
//    std::vector<double> sphericalHarmonicsInitial(25,0.0);
//    std::vector<double> sphericalHarmonicsInitialInv(25,0.0);
//    std::vector<double> coordinatesInitial(15,0.0);
//    for(unsigned int iPoint = 0 ; iPoint < 5; iPoint++)
//      {
//        double r{}, theta{}, phi{};
//        coordinatesInitial[3*iPoint + 0 ] = rand() - 0.5;
//        coordinatesInitial[3*iPoint + 1 ] = rand() - 0.5;
//        coordinatesInitial[3*iPoint + 2 ] = rand() - 0.5;
//
//        std::array<double, 3> relativeEvalPoint;
//        relativeEvalPoint[0] = coordinatesInitial[3*iPoint + 0 ];
//        relativeEvalPoint[1] = coordinatesInitial[3*iPoint + 1 ];
//        relativeEvalPoint[2] = coordinatesInitial[3*iPoint + 2 ];
//        convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);
//
//        for(int mOrb = -2; mOrb <= 2; mOrb++)
//          {
//            sphericalHarmonicsInitial[iPoint*5 + mOrb + 2]=
//              realSphericalHarmonics(2, mOrb, theta, phi);
//
//            sphericalHarmonicsInitialInv[iPoint*5 + mOrb+ 2 ]=
//              sphericalHarmonicsInitial[iPoint*5 + mOrb +2 ];
//          }
//      }
//
//    int dimIn = 5;
//    linearAlgebraOperations::inverse(&sphericalHarmonicsInitialInv[0], dimIn);
//
//    pcout<<"printing sphericalHarmonicsInitial \n";
//    for(unsigned int i = 0 ; i  < 5; i++)
//      {
//        for(unsigned int j = 0 ; j<5; j++)
//          {
//            pcout<< sphericalHarmonicsInitial[i*5 + j]<< " ";
//          }
//        pcout<<"\n";
//      }
//
//    pcout<<"printing sphericalHarmonicsInitialInv \n";
//    for(unsigned int i = 0 ; i  < 5; i++)
//      {
//        for(unsigned int j = 0 ; j<5; j++)
//          {
//            pcout<< sphericalHarmonicsInitialInv[i*5 + j]<< " ";
//          }
//        pcout<<"\n";
//      }
//
//    dMatrix.resize(d_numSymm*25,0.0);
//    std::fill(dMatrix.begin(),dMatrix.end(),0.0);
//
//    for( unsigned int iSymm = 0 ; iSymm < d_numSymm; iSymm++ )
//      {
//        std::vector<double> sphericalHarmonicsNew(25,0.0);
//
//        for(unsigned int iPoint = 0 ; iPoint < 5; iPoint++)
//          {
//            double r{}, theta{}, phi{};
//            std::array<double, 3> relativeEvalPoint;
//            relativeEvalPoint[0] = 0;
//            relativeEvalPoint[1] = 0;
//            relativeEvalPoint[2] = 0;
//
//
//            for (unsigned int j = 0; j < 3; ++j)
//              {
//                for (unsigned int k = 0; k < 3; ++k)
//                  {
//                    relativeEvalPoint[j] +=
//                      rotation[iSymm][j][k]* coordinatesInitial[3*iPoint + k ] ;
//                  }
//              }
//            convertCartesianToSpherical(relativeEvalPoint, r, theta, phi);
//            for(int mOrb = -2; mOrb <= 2; mOrb++)
//              {
//                sphericalHarmonicsNew[iPoint*5 + mOrb+ 2]=
//                  realSphericalHarmonics(2, mOrb, theta, phi);
//              }
//          }
//
//        pcout<<"printing sphericalHarmonicsNew for Symm = "<<iSymm<<"\n";
//        for(unsigned int i = 0 ; i  < 5; i++)
//          {
//            for(unsigned int j = 0 ; j<5; j++)
//              {
//                pcout<<sphericalHarmonicsNew[i*5 + j]<< " ";
//              }
//            pcout<<"\n";
//          }
//
//        for(unsigned int i = 0 ; i<5;i++)
//          {
//            for(unsigned int j = 0 ; j < 5; j++)
//              {
//                for(unsigned int k = 0; k<5; k++)
//                  {
//                    dMatrix[iSymm*25 + i*5 + j ] +=
//                      sphericalHarmonicsInitialInv[i*5 + k] *
//                      sphericalHarmonicsNew[k*5 + j];
//                  }
//              }
//          }
//
//        pcout<<"Printing d matrix for iSymm = "<<iSymm<<"\n";
//        for(unsigned int i = 0 ; i<5;i++)
//          {
//            for(unsigned int j = 0 ; j < 5; j++)
//              {
//                pcout<<dMatrix[iSymm*25 + i*5 + j ]<<" ";
//              }
//            pcout<<" \n";
//          }
//      }
//  }

//  // The function does not call compress or constraints.slave_to_master().
//  // It is expected that it will be called in the kohnShamOperator.cc after this
//  // function is called
//  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
//  void hubbard<ValueType, memorySpace>::preComputeDiscreteAtomicWavefunctions()
//  {
//    const ValueType scalarCoeffAlpha = ValueType(1.0);
//    const ValueType scalarCoeffBeta  = ValueType(0.0);
//    const char transA = 'N', transB = 'N';
//    const unsigned int inc = 1;
//    unsigned int iElem = 0 ;
//
//    typename dealii::DoFHandler<3>::active_cell_iterator
//      cell    = d_dofHandler->begin_active(),
//      endc    = d_dofHandler->end();
//
//    std::vector<ValueType> cellLevelDiscreteAtomicWavefunc;
//    atomicDiscreteData.resize(d_numKPoints);
//#ifdef DFTFE_WITH_DEVICE
//    if(d_dftParamsPtr->useDevice)
//      {
//        d_atomicDiscreteDataDevice.resize(d_numKPoints);
//      }
//#endif
//    for(unsigned int iKPoint = 0; iKPoint  <d_numKPoints; iKPoint++)
//      {
//        atomicDiscreteData[iKPoint].resize(d_noSpecies);
//        for (unsigned int iSpecies = 0 ; iSpecies < d_noSpecies ; iSpecies++)
//          {
//            iElem = 0 ;
//            unsigned int totalAtomicWaveFunctions = d_atomicWavefunctionPerSpecies[iSpecies]*d_noAtoms[iSpecies];
//
//
//            dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
//              d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
//              totalAtomicWaveFunctions,
//              atomicDiscreteData[iKPoint][iSpecies]);
//
//            dftUtils::constraintMatrixInfo constraintsMatrixSpecies;
//
//            constraintsMatrixSpecies.initialize(
//              d_matrixFreeDataPtr->get_vector_partitioner(
//                d_matrixFreeVectorComponent),
//              *d_constraintMatrixPtr);
//
//            constraintsMatrixSpecies.precomputeMaps(
//              atomicDiscreteData[iKPoint][iSpecies].getMPIPatternP2P(),
//              totalAtomicWaveFunctions);
//            std::vector<ValueType> shapeFunctionValues;
//            shapeFunctionValues.resize
//              (d_numberDofsPerElement*d_numberQuadraturePoints, ValueType(0.0));
//
//            for (unsigned int iquad = 0; iquad < d_numberQuadraturePoints; ++iquad)
//              for (unsigned int iNode = 0; iNode < d_numberDofsPerElement; ++iNode)
//                shapeFunctionValues[iquad + iNode*d_numberQuadraturePoints] =
//                  ValueType(d_shapeFunctionValues[iquad * d_numberDofsPerElement + iNode]);
//
//            //TODO see if this is necessary
//            atomicDiscreteData[iKPoint][iSpecies].setValue(0.0);
//            cell    = d_dofHandler->begin_active();
//            std::vector<ValueType>cellLevelAtomicWave;
//            cellLevelAtomicWave.resize(d_numberDofsPerElement*totalAtomicWaveFunctions);
//            cellLevelDiscreteAtomicWavefunc.resize(d_numberDofsPerElement*totalAtomicWaveFunctions);
//            for(cell = d_dofHandler->begin_active() ; cell!= endc; ++cell)
//              {
//                if (cell->is_locally_owned())
//                  {
//                    //                for (unsigned int iNode = 0; iNode < d_numberDofsPerElement; iNode++)
//                    //                  {
//                    //                    dcopy_(&totalAtomicWaveFunctions,
//                    //                           atomicWaveNodalData[iSpecies].begin() +
//                    //                             d_fullFlattenedArrayCellLocalProcIndexIdMapAtomic[iSpecies]
//                    //                                                                              [iElem * d_numberDofsPerElement + iNode],
//                    //                           &inc,
//                    //                           &cellLevelAtomicWave[totalAtomicWaveFunctions * iNode],
//                    //                           &inc);
//                    //                  }
//                    //
//                    //                dgemm(&transA,
//                    //                      &transB,
//                    //                      &totalAtomicWaveFunctions,
//                    //                      &d_numberQuadraturePoints,
//                    //                      &d_numberDofsPerElement,
//                    //                      &scalarCoeffAlpha,
//                    //                      &cellLevelAtomicWave[0],
//                    //                      &totalAtomicWaveFunctions,
//                    //                      &d_shapeFunctionValues[0],
//                    //                      &d_numberDofsPerElement,
//                    //                      &scalarCoeffBeta,
//                    //                      &cellLevelAtomicWaveQuad[0],
//                    //                      &totalAtomicWaveFunctions);
//
//                    xgemm(&transA,
//                          &transB,
//                          &totalAtomicWaveFunctions,
//                          &d_numberDofsPerElement,
//                          &d_numberQuadraturePoints,
//                          &scalarCoeffAlpha,
//                          &atomicWaveQuadDataJxW[iKPoint][iSpecies][(iElem*d_numberQuadraturePoints)*totalAtomicWaveFunctions],
//                          &totalAtomicWaveFunctions,
//                          &shapeFunctionValues[0],
//                          &d_numberQuadraturePoints,
//                          &scalarCoeffBeta,
//                          &cellLevelDiscreteAtomicWavefunc[0],
//                          &totalAtomicWaveFunctions);
//
//                    //std::fill(cellLevelDiscreteAtomicWavefunc.begin(),cellLevelDiscreteAtomicWavefunc.end(),0.0);
//
//                    //for (unsigned int dofId = 0; dofId < d_numberDofsPerElement; dofId++)
//                    //  {
//                    //    for (unsigned int q_point = 0; q_point < d_numberQuadraturePoints ; q_point++)
//                    //      {
//                    //        for (unsigned int inputId = 0; inputId < totalAtomicWaveFunctions; inputId++)
//                    //          {
//                    //            cellLevelDiscreteAtomicWavefunc[dofId * totalAtomicWaveFunctions + inputId ] +=
//                    //              atomicWaveQuadDataJxW[iSpecies][(iElem*d_numberQuadraturePoints)*totalAtomicWaveFunctions +
//                    //                                           q_point*totalAtomicWaveFunctions +  inputId ]*
//                    //              d_shapeFunctionValues[dofId + q_point*d_numberDofsPerElement];
//                    //
//                    //                          }
//                    //                      }
//                    //
//                    //                  }
//                    for (unsigned int iNode = 0; iNode < d_numberDofsPerElement; ++iNode)
//                      {
//                        dealii::types::global_dof_index localNodeId =
//                          d_fullFlattenedArrayCellLocalProcIndexIdMapAtomic[iSpecies]
//                                                                           [iElem * d_numberDofsPerElement + iNode];
//                        axpy(&totalAtomicWaveFunctions,
//                             &scalarCoeffAlpha,
//                             &cellLevelDiscreteAtomicWavefunc[totalAtomicWaveFunctions * iNode],
//                             &inc,
//                             atomicDiscreteData[iKPoint][iSpecies].data() + localNodeId,
//                             &inc);
//                      }
//
//                    iElem++;
//
//                  }
//              }
//
//            //constraintsMatrixSpecies.distribute_slave_to_master(atomicDiscreteData[iSpecies], totalAtomicWaveFunctions);
//            atomicDiscreteData[iKPoint][iSpecies].accumulateAddLocallyOwned();
//            //        constraintsMatrixSpecies.distribute(atomicDiscreteData[iSpecies], totalAtomicWaveFunctions);
//
//            /*
//  std::vector<double> vecNorm;
//    vecNorm.resize(totalAtomicWaveFunctions);
//    std::fill(vecNorm.begin(),vecNorm.end(),0.0);
//    atomicDiscreteData[iKPoint][iSpecies].l2Norm(&vecNorm[0]);
//    pcout<<" Norm of discrete data \n";
//    for(unsigned int iWave =0 ; iWave <totalAtomicWaveFunctions; iWave++ )
//      {
//        pcout<<" iWave = "<<vecNorm[iWave]<<"\n";
//      }
//      */
//
//#ifdef DFTFE_WITH_DEVICE
//            if(d_dftParamsPtr->useDevice)
//              {
//                const unsigned int BVec = d_dftParamsPtr->chebyWfcBlockSize ;
//                d_dotProductAtomicWaveInputWaveApplyOptDevice.resize(d_noSpecies);
//                d_atomicDiscreteDataDevice[iKPoint].resize(d_noSpecies);
//                coeffApplyDevice.resize(d_noSpecies);
//                d_dotProductAtomicWaveInputWaveOccMatOptDevice.resize(d_noSpecies);
//                d_dotProdVecOccMat.resize(d_noSpecies);
//                for (unsigned int iSpecies = 0; iSpecies < d_noSpecies; iSpecies++)
//                  {
//                    unsigned int totalAtomicWaveFunctions = d_atomicWavefunctionPerSpecies[iSpecies]*d_noAtoms[iSpecies];
//                    d_dotProductAtomicWaveInputWaveApplyOptDevice[iSpecies].resize(d_noAtoms[iSpecies]*
//                                                                                     d_atomicWavefunctionPerSpecies[iSpecies]*
//                                                                                     BVec, ValueType(0.0));
//
//                    d_dotProductAtomicWaveInputWaveOccMatOptDevice[iSpecies].resize(d_noAtoms[iSpecies]*
//                                                                                      d_atomicWavefunctionPerSpecies[iSpecies]*
//                                                                                      d_noOfSpin* d_numKPoints*// get number of k points.
//                                                                                      d_numberWaveFunctions,ValueType(0.0));
//                    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
//                      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
//                      totalAtomicWaveFunctions,
//                      d_atomicDiscreteDataDevice[iKPoint][iSpecies]);
//
//                    dftfe::utils::deviceMemcpyH2D(d_atomicDiscreteDataDevice[iKPoint][iSpecies].data(),
//                                                  atomicDiscreteData[iKPoint][iSpecies].data(),
//                                                  totalAtomicWaveFunctions*d_localVectorSize
//                                                    * sizeof(ValueType));
//                    coeffApplyDevice[iSpecies].resize(totalAtomicWaveFunctions*BVec,ValueType(0.0));
//
//                    d_constraintsInputSrcDevice.initialize(
//                      d_matrixFreeDataPtr->get_vector_partitioner(
//                        d_matrixFreeVectorComponent),
//                      *d_constraintMatrixPtr);
//
//                    d_dotProdVecOccMat[iSpecies].resize(BVec*d_noAtoms[iSpecies]*d_atomicWavefunctionPerSpecies[iSpecies],ValueType(0.0));
//                  }
//              }
//#endif
//          }
//
//      }
//
//  }


//  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
//  unsigned int
//  hubbard<ValueType, memorySpace>::getTotalNumberOfAtomsInCurrentProcessor()
//  {
//    return d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess().size();
//  }

//  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
//  unsigned int
//  hubbard<ValueType, memorySpace>::getAtomIdInCurrentProcessor(
//    unsigned int iAtom)
//  {
//    std::vector<unsigned int> atomIdList =
//      d_atomicProjectorFnsContainer->getAtomIdsInCurrentProcess();
//    return (atomIdList[iAtom]);
//  }

//  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
//  unsigned int
//  hubbard< ValueType, memorySpace>::
//    getTotalNumberOfSphericalFunctionsForAtomId(unsigned int atomId)
//  {
//    std::vector<unsigned int> atomicNumbers =
//      d_atomicProjectorFnsContainer->getAtomicNumbers();
//    return (
//      d_atomicProjectorFnsContainer->getTotalNumberOfSphericalFunctionsPerAtom(
//        atomicNumbers[atomId]));
//  }



}
