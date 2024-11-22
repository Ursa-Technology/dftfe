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
// @author Sambit Das
//

#include "constants.h"
#include "dftUtils.h"
#include "nonlocalPspEnergyDensityWfcContractions.h"
#include "vectorUtilities.h"
#include <MemoryStorage.h>
#include <MemoryTransfer.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceAPICalls.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <nonlocalPspEnergyDensityWfcContractionsDeviceKernels.h>
#endif


namespace dftfe
{
    namespace
    {
      template <dftfe::utils::MemorySpace memorySpace>
      void
      interpolatePsiGradPsiNlpQuads(
        std::shared_ptr<dftfe::basis::FEBasisOperations<dataTypes::number,
                                                        double,
                                                        memorySpace>>
          &                basisOperationsPtr,
        const unsigned int nlpspQuadratureId,
        const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          &BLASWrapperPtr,
        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &Xb,
        const unsigned int                                                 BVec,
        const unsigned int numCells,
        const unsigned int cellsBlockSize,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &psiQuadsNLP,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &gradPsiQuadsNLP)
      {
        const int blockSize    = cellsBlockSize;
        const int numberBlocks = numCells / blockSize;
        const int remBlockSize = numCells - numberBlocks * blockSize;


        basisOperationsPtr->reinit(BVec, cellsBlockSize, nlpspQuadratureId);


        for (int iblock = 0; iblock < (numberBlocks + 1); iblock++)
          {
            const int currentBlockSize =
              (iblock == numberBlocks) ? remBlockSize : blockSize;
            const int startingId = iblock * blockSize;

            if (currentBlockSize > 0)
              {
                basisOperationsPtr->interpolateKernel(
                  Xb,
                  psiQuadsNLP.data() +
                    startingId * basisOperationsPtr->nQuadsPerCell() * BVec,
                  gradPsiQuadsNLP.data() +
                    startingId * 3 * basisOperationsPtr->nQuadsPerCell() * BVec,
                  std::pair<unsigned int, unsigned int>(startingId,
                                                        startingId +
                                                          currentBlockSize));
              }
          }
      }


      template <dftfe::utils::MemorySpace memorySpace>
      void
      nlpPsiContraction(
        const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          &BLASWrapperPtr,
        const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &psiQuadsNLP,
        const dftfe::utils::MemoryStorage<double, memorySpace>
          &partialOccupancies,
        const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &                      onesVecNLP,
        const dataTypes::number *projectorKetTimesVectorParFlattened,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &nonTrivialIdToElemIdMap,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &                projecterKetTimesFlattenedVectorLocalIds,
        const unsigned int numCells,
        const unsigned int numQuadsNLP,
        const unsigned int numPsi,
        const unsigned int totalNonTrivialPseudoWfcs,
        const unsigned int innerBlockSizeEnlp,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &nlpContractionContribution,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
        dataTypes::number
          *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHPinnedTemp)
      {
        const int blockSizeNlp    = innerBlockSizeEnlp;
        const int numberBlocksNlp = totalNonTrivialPseudoWfcs / blockSizeNlp;
        const int remBlockSizeNlp =
          totalNonTrivialPseudoWfcs - numberBlocksNlp * blockSizeNlp;


        dataTypes::number scalarCoeffAlphaNlp = dataTypes::number(1.0);
        dataTypes::number scalarCoeffBetaNlp  = dataTypes::number(0.0);

        for (int iblocknlp = 0; iblocknlp < (numberBlocksNlp + 1); iblocknlp++)
          {
            const int currentBlockSizeNlp =
              (iblocknlp == numberBlocksNlp) ? remBlockSizeNlp : blockSizeNlp;
            const int startingIdNlp = iblocknlp * blockSizeNlp;
            if (currentBlockSizeNlp > 0)
              {
                if (memorySpace == dftfe::utils::MemorySpace::HOST)
                  {
                    for (unsigned int ipseudowfc = 0;
                         ipseudowfc < currentBlockSizeNlp;
                         ipseudowfc++)
                      for (unsigned int iquad = 0; iquad < numQuadsNLP; iquad++)
                        for (unsigned int iwfc = 0; iwfc < numPsi; iwfc++)
                          nlpContractionContribution[ipseudowfc * numQuadsNLP *
                                                       numPsi +
                                                     iquad * numPsi + iwfc] =
                            partialOccupancies.data()[iwfc] *
                            dftfe::utils::complexConj(
                              psiQuadsNLP.data()[nonTrivialIdToElemIdMap
                                                     .data()[startingIdNlp +
                                                             ipseudowfc] *
                                                   numQuadsNLP * numPsi +
                                                 iquad * numPsi + iwfc]) *
                            projectorKetTimesVectorParFlattened
                              [projecterKetTimesFlattenedVectorLocalIds
                                   .data()[startingIdNlp + ipseudowfc] *
                                 numPsi +
                               iwfc];
                  }
#  if defined(DFTFE_WITH_DEVICE)
                else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                  nonlocalPspEnergyDensityDeviceKernels::nlpContractionContributionPsiIndex(
                    numPsi,
                    currentBlockSizeNlp,
                    numQuadsNLP,
                    startingIdNlp,
                    projectorKetTimesVectorParFlattened,
                    psiQuadsNLP.data(),
                    partialOccupancies.data(),
                    nonTrivialIdToElemIdMap.data(),
                    projecterKetTimesFlattenedVectorLocalIds.data(),
                    nlpContractionContribution.data());
#  endif


                BLASWrapperPtr->xgemm(
                  'N',
                  'N',
                  1,
                  currentBlockSizeNlp * numQuadsNLP,
                  numPsi,
                  &scalarCoeffAlphaNlp,
                  onesVecNLP.data(),
                  1,
                  nlpContractionContribution.data(),
                  numPsi,
                  &scalarCoeffBetaNlp,
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock
                    .data(),
                  1);


                dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::HOST,
                                             memorySpace>::
                  copy(
                    currentBlockSizeNlp * numQuadsNLP,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHPinnedTemp,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock
                      .data());


                for (unsigned int i = 0; i < currentBlockSizeNlp * numQuadsNLP;
                     i++)
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH
                    [startingIdNlp * numQuadsNLP + i] +=
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHPinnedTemp
                      [i];
              }
          }
      }


      template <dftfe::utils::MemorySpace memorySpace>
      void
      nonlocalPspEnergyDensityWfcContractionsKernelsAll(
        std::shared_ptr<dftfe::basis::FEBasisOperations<dataTypes::number,
                                                        double,
                                                        memorySpace>>
          &                basisOperationsPtr,
        const unsigned int nlpspQuadratureId,
        const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          &BLASWrapperPtr,
        std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
          oncvClassPtr,
        const unsigned int kPointIndex,
        const unsigned int spinIndex,
        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
          &flattenedArrayBlock,
        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
          &projectorKetTimesVector,
        const dataTypes::number *X,
        const dftfe::utils::MemoryStorage<double, memorySpace> &eigenValues,
        const dftfe::utils::MemoryStorage<double, memorySpace>
          &          partialOccupancies,
        const double kcoordx,
        const double kcoordy,
        const double kcoordz,
        const dftfe::utils::MemoryStorage<double, memorySpace> &onesVec,
        const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &onesVecNLP,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &nonTrivialIdToElemIdMap,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &projecterKetTimesFlattenedVectorLocalIds,
        const unsigned int startingVecId,
        const unsigned int N,
        const unsigned int numPsi,
        const unsigned int numCells,
        const unsigned int numQuadsNLP,
        const unsigned int totalNonTrivialPseudoWfcs,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &psiQuadsNLP,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &gradPsiQuadsNLP,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &nlpContractionContribution,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
        dataTypes::number
          *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHPinnedTemp,
        const unsigned int cellsBlockSize,
        const unsigned int innerBlockSizeEnlp)
      {
        if (memorySpace == dftfe::utils::MemorySpace::HOST)
          for (unsigned int iNode = 0; iNode < basisOperationsPtr->nOwnedDofs();
               ++iNode)
            std::memcpy(flattenedArrayBlock.data() + iNode * numPsi,
                        X + iNode * N + startingVecId,
                        numPsi * sizeof(dataTypes::number));
#if defined(DFTFE_WITH_DEVICE)
        else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          BLASWrapperPtr->stridedCopyToBlockConstantStride(
            numPsi,
            N,
            basisOperationsPtr->nOwnedDofs(),
            startingVecId,
            X,
            flattenedArrayBlock.data());
#endif


        flattenedArrayBlock.updateGhostValues();
        basisOperationsPtr->distribute(flattenedArrayBlock);




        oncvClassPtr->getNonLocalOperator()->applyVCconjtransOnX(
          flattenedArrayBlock,
          kPointIndex,
          CouplingStructure::diagonal,
          oncvClassPtr->getCouplingMatrix(),
          projectorKetTimesVector);



        interpolatePsiGradPsiNlpQuads(basisOperationsPtr,
                                      nlpspQuadratureId,
                                      BLASWrapperPtr,
                                      flattenedArrayBlock,
                                      numPsi,
                                      numCells,
                                      cellsBlockSize,
                                      psiQuadsNLP,
                                      gradPsiQuadsNLP);

        if (totalNonTrivialPseudoWfcs > 0)
          {
            nlpPsiContraction(
              BLASWrapperPtr,
              psiQuadsNLP,
              partialOccupancies,
              onesVecNLP,
              projectorKetTimesVector.data(),
              nonTrivialIdToElemIdMap,
              projecterKetTimesFlattenedVectorLocalIds,
              numCells,
              numQuadsNLP,
              numPsi,
              totalNonTrivialPseudoWfcs,
              innerBlockSizeEnlp,
              nlpContractionContribution,
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHPinnedTemp);
          }
      }

    } // namespace

    template <dftfe::utils::MemorySpace memorySpace>
    void
    nonlocalPspEnergyDensityWfcContractionsAllH(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
        &                basisOperationsPtr,
      const unsigned int nlpspQuadratureId,
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        &BLASWrapperPtr,
      std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
                                                               oncvClassPtr,
      const dataTypes::number *                                X,
      const unsigned int                      spinPolarizedFlag,
      const unsigned int                      spinIndex,
      const std::vector<std::vector<double>> &eigenValuesH,
      const std::vector<std::vector<double>> &partialOccupanciesH,
      const std::vector<double> &             kPointCoordinates,
      const unsigned int                      MLoc,
      const unsigned int                      N,
      const unsigned int                      numCells,
      const unsigned int                      numQuadsNLP,
      dataTypes::number
        *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
      const MPI_Comm &     mpiCommParent,
      const MPI_Comm &     interBandGroupComm,
      const dftParameters &dftParams)
    {
      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int blockSize =
        std::min(dftParams.chebyWfcBlockSize,
                 bandGroupLowHighPlusOneIndices[1]);



      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        *flattenedArrayBlockPtr;

      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        projectorKetTimesVector;


      dftfe::utils::MemoryStorage<double, memorySpace> eigenValues(blockSize,
                                                                   0.0);
      dftfe::utils::MemoryStorage<double, memorySpace> partialOccupancies(
        blockSize, 0.0);

      dftfe::utils::MemoryStorage<double, memorySpace> onesVec(blockSize, 1.0);
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> onesVecNLP(
        blockSize, dataTypes::number(1.0));

      const unsigned int cellsBlockSize = std::min((unsigned int)10, numCells);


      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> psiQuadsNLP(
        numCells * numQuadsNLP * blockSize, dataTypes::number(0.0));

      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        gradPsiQuadsNLPFlat(numCells * numQuadsNLP * 3 * blockSize,
                            dataTypes::number(0.0));


      const unsigned int totalNonTrivialPseudoWfcs =oncvClassPtr->getNonLocalOperator()
                  ->getTotalNonTrivialSphericalFnsOverAllCells();

      const unsigned int innerBlockSizeEnlp =
        std::min((unsigned int)2, totalNonTrivialPseudoWfcs);
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        nlpContractionContribution(innerBlockSizeEnlp * numQuadsNLP *
                                     blockSize,
                                   dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock;
      dftfe::utils::MemoryStorage<unsigned int, memorySpace>
        projecterKetTimesFlattenedVectorLocalIds;
      dftfe::utils::MemoryStorage<unsigned int, memorySpace>
        nonTrivialIdToElemIdMap;
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHPinnedTemp;
      if (totalNonTrivialPseudoWfcs > 0)
        {
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP, dataTypes::number(0.0));
          projecterKetTimesFlattenedVectorLocalIds.resize(
            totalNonTrivialPseudoWfcs, 0.0);
          nonTrivialIdToElemIdMap.resize(totalNonTrivialPseudoWfcs, 0);



          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHPinnedTemp
            .resize(innerBlockSizeEnlp * numQuadsNLP, 0);

          dftfe::utils::
            MemoryTransfer<memorySpace, dftfe::utils::MemorySpace::HOST>::copy(
              totalNonTrivialPseudoWfcs,
              nonTrivialIdToElemIdMap.data(),
              &(oncvClassPtr->getNonLocalOperator()
                  ->getNonTrivialAllCellsSphericalFnAlphaToElemIdMap()[0]));

          dftfe::utils::
            MemoryTransfer<memorySpace, dftfe::utils::MemorySpace::HOST>::copy(
              totalNonTrivialPseudoWfcs,
              projecterKetTimesFlattenedVectorLocalIds.data(),
              &(oncvClassPtr->getNonLocalOperator()
                  ->getSphericalFnTimesVectorFlattenedVectorLocalIds()[0]));

        }

      const unsigned numKPoints = kPointCoordinates.size() / 3;
      for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
        {
          // spin index update is not required

          const double kcoordx = kPointCoordinates[kPoint * 3 + 0];
          const double kcoordy = kPointCoordinates[kPoint * 3 + 1];
          const double kcoordz = kPointCoordinates[kPoint * 3 + 2];

          if (totalNonTrivialPseudoWfcs > 0)
            {
              std::fill(
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH +
                  kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP,
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH +
                  (kPoint + 1) * totalNonTrivialPseudoWfcs * numQuadsNLP,
                dataTypes::number(0.0));
            }


          for (unsigned int ivec = 0; ivec < N; ivec += blockSize)
            {
              const unsigned int currentBlockSize =
                std::min(blockSize, N - ivec);

              flattenedArrayBlockPtr =
                &(basisOperationsPtr->getMultiVector(currentBlockSize, 0));

              oncvClassPtr->getNonLocalOperator()
                ->initialiseFlattenedDataStructure(currentBlockSize,
                                                   projectorKetTimesVector);


              if ((ivec + currentBlockSize) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (ivec + currentBlockSize) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  std::vector<double> blockedEigenValues(currentBlockSize, 0.0);
                  std::vector<double> blockedPartialOccupancies(
                    currentBlockSize, 0.0);
                  for (unsigned int iWave = 0; iWave < currentBlockSize;
                       ++iWave)
                    {
                      blockedEigenValues[iWave] =
                        eigenValuesH[kPoint][spinIndex * N + ivec + iWave];
                      blockedPartialOccupancies[iWave] =
                        partialOccupanciesH[kPoint]
                                           [spinIndex * N + ivec + iWave];
                    }


                  dftfe::utils::MemoryTransfer<
                    memorySpace,
                    dftfe::utils::MemorySpace::HOST>::
                    copy(currentBlockSize,
                         eigenValues.data(),
                         &blockedEigenValues[0]);

                  dftfe::utils::MemoryTransfer<
                    memorySpace,
                    dftfe::utils::MemorySpace::HOST>::
                    copy(currentBlockSize,
                         partialOccupancies.data(),
                         &blockedPartialOccupancies[0]);

                  nonlocalPspEnergyDensityWfcContractionsKernelsAll(
                    basisOperationsPtr,
                    nlpspQuadratureId,
                    BLASWrapperPtr,
                    oncvClassPtr,
                    kPoint,
                    spinIndex,
                    *flattenedArrayBlockPtr,
                    projectorKetTimesVector,
                    X +
                      ((1 + spinPolarizedFlag) * kPoint + spinIndex) * MLoc * N,
                    eigenValues,
                    partialOccupancies,
                    kcoordx,
                    kcoordy,
                    kcoordz,
                    onesVec,
                    onesVecNLP,
                    nonTrivialIdToElemIdMap,
                    projecterKetTimesFlattenedVectorLocalIds,
                    ivec,
                    N,
                    currentBlockSize,
                    numCells,
                    numQuadsNLP,
                    totalNonTrivialPseudoWfcs,
                    psiQuadsNLP,
                    gradPsiQuadsNLPFlat,
                    nlpContractionContribution,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH +
                      kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHPinnedTemp
                      .data(),
                    cellsBlockSize,
                    innerBlockSizeEnlp);
                } // band parallelization
            }     // ivec loop
        } // k point loop
    }


#if defined(DFTFE_WITH_DEVICE)
    template void
    nonlocalPspEnergyDensityWfcContractions(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
        &                basisOperationsPtr,
      const unsigned int nlpspQuadratureId,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
      std::shared_ptr<
        dftfe::oncvClass<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>>
        oncvClassPtr,
      const dataTypes::number *               X,
      const unsigned int                      spinPolarizedFlag,
      const unsigned int                      spinIndex,
      const std::vector<std::vector<double>> &eigenValuesH,
      const std::vector<std::vector<double>> &partialOccupanciesH,
      const std::vector<double> &             kPointCoordinates,
      const unsigned int                      MLoc,
      const unsigned int                      N,
      const unsigned int                      numCells,
      const unsigned int                      numQuadsNLP,
      dataTypes::number
        *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
      const MPI_Comm &     mpiCommParent,
      const MPI_Comm &     interBandGroupComm,
      const dftParameters &dftParams);
#endif

    template void
    nonlocalPspEnergyDensityWfcContractionsAllH(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int nlpspQuadratureId,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperPtr,
      std::shared_ptr<
        dftfe::oncvClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>>
                                                                oncvClassPtr,
      const dataTypes::number *                                 X,
      const unsigned int                      spinPolarizedFlag,
      const unsigned int                      spinIndex,
      const std::vector<std::vector<double>> &eigenValuesH,
      const std::vector<std::vector<double>> &partialOccupanciesH,
      const std::vector<double> &             kPointCoordinates,
      const unsigned int                      MLoc,
      const unsigned int                      N,
      const unsigned int                      numCells,
      const unsigned int                      numQuadsNLP,
      dataTypes::number
        *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
      const MPI_Comm &     mpiCommParent,
      const MPI_Comm &     interBandGroupComm,
      const dftParameters &dftParams);

} // namespace dftfe
