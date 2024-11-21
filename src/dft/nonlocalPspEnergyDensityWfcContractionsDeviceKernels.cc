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

#if defined(DFTFE_WITH_DEVICE)
#  include "dftfeDataTypes.h"
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceAPICalls.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <nonlocalPspEnergyDensityWfcContractionsDeviceKernels.h>

namespace dftfe
{
  namespace nonlocalPspEnergyDensityDeviceKernels
  {
    namespace
    {
      __global__ void
      nlpContractionContributionPsiIndexDeviceKernel(
        const unsigned int  numPsi,
        const unsigned int  numQuadsNLP,
        const unsigned int  totalNonTrivialPseudoWfcs,
        const unsigned int  startingId,
        const double *      projectorKetTimesVectorPar,
        const double *      gradPsiOrPsiQuadValuesNLP,
        const double *      partialOccupancies,
        const unsigned int *nonTrivialIdToElemIdMap,
        const unsigned int *projecterKetTimesFlattenedVectorLocalIds,
        double *            nlpContractionContribution)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex  = index / numPsi;
            const unsigned int wfcId       = index - blockIndex * numPsi;
            unsigned int       pseudoWfcId = blockIndex / numQuadsNLP;
            const unsigned int quadId = blockIndex - pseudoWfcId * numQuadsNLP;
            pseudoWfcId += startingId;
            nlpContractionContribution[index] =
              partialOccupancies[wfcId] *
              gradPsiOrPsiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId] *
                                          numQuadsNLP * numPsi +
                                        quadId * numPsi + wfcId] *
              projectorKetTimesVectorPar
                [projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId] *
                   numPsi +
                 wfcId];
          }
      }

      __global__ void
      nlpContractionContributionPsiIndexDeviceKernel(
        const unsigned int                       numPsi,
        const unsigned int                       numQuadsNLP,
        const unsigned int                       totalNonTrivialPseudoWfcs,
        const unsigned int                       startingId,
        const dftfe::utils::deviceDoubleComplex *projectorKetTimesVectorPar,
        const dftfe::utils::deviceDoubleComplex *gradPsiOrPsiQuadValuesNLP,
        const double *                           partialOccupancies,
        const unsigned int *                     nonTrivialIdToElemIdMap,
        const unsigned int *projecterKetTimesFlattenedVectorLocalIds,
        dftfe::utils::deviceDoubleComplex *nlpContractionContribution)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex  = index / numPsi;
            const unsigned int wfcId       = index - blockIndex * numPsi;
            unsigned int       pseudoWfcId = blockIndex / numQuadsNLP;
            const unsigned int quadId = blockIndex - pseudoWfcId * numQuadsNLP;
            pseudoWfcId += startingId;

            const dftfe::utils::deviceDoubleComplex temp = dftfe::utils::mult(
              dftfe::utils::conj(
                gradPsiOrPsiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId] *
                                            numQuadsNLP * numPsi +
                                          quadId * numPsi + wfcId]),
              projectorKetTimesVectorPar
                [projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId] *
                   numPsi +
                 wfcId]);
            nlpContractionContribution[index] =
              dftfe::utils::makeComplex(partialOccupancies[wfcId] * temp.x,
                                        partialOccupancies[wfcId] * temp.y);
          }
      }

    } // namespace

    template <typename ValueType>
    void
    nlpContractionContributionPsiIndex(
      const unsigned int  wfcBlockSize,
      const unsigned int  blockSizeNlp,
      const unsigned int  numQuadsNLP,
      const unsigned int  startingIdNlp,
      const ValueType *   projectorKetTimesVectorPar,
      const ValueType *   gradPsiOrPsiQuadValuesNLP,
      const double *      partialOccupancies,
      const unsigned int *nonTrivialIdToElemIdMap,
      const unsigned int *projecterKetTimesFlattenedVectorLocalIds,
      ValueType *         nlpContractionContribution)
    {
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      nlpContractionContributionPsiIndexDeviceKernel<<<
        (wfcBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * numQuadsNLP * blockSizeNlp,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        wfcBlockSize,
        numQuadsNLP,
        blockSizeNlp,
        startingIdNlp,
        dftfe::utils::makeDataTypeDeviceCompatible(projectorKetTimesVectorPar),
        dftfe::utils::makeDataTypeDeviceCompatible(gradPsiOrPsiQuadValuesNLP),
        partialOccupancies,
        nonTrivialIdToElemIdMap,
        projecterKetTimesFlattenedVectorLocalIds,
        dftfe::utils::makeDataTypeDeviceCompatible(nlpContractionContribution));
#  elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        nlpContractionContributionPsiIndexDeviceKernel,
        (wfcBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * numQuadsNLP * blockSizeNlp,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        wfcBlockSize,
        numQuadsNLP,
        blockSizeNlp,
        startingIdNlp,
        dftfe::utils::makeDataTypeDeviceCompatible(projectorKetTimesVectorPar),
        dftfe::utils::makeDataTypeDeviceCompatible(gradPsiOrPsiQuadValuesNLP),
        partialOccupancies,
        nonTrivialIdToElemIdMap,
        projecterKetTimesFlattenedVectorLocalIds,
        dftfe::utils::makeDataTypeDeviceCompatible(nlpContractionContribution));
#  endif
    }


    template void
    nlpContractionContributionPsiIndex(
      const unsigned int       wfcBlockSize,
      const unsigned int       blockSizeNlp,
      const unsigned int       numQuadsNLP,
      const unsigned int       startingIdNlp,
      const dataTypes::number *projectorKetTimesVectorPar,
      const dataTypes::number *gradPsiOrPsiQuadValuesNLP,
      const double *           partialOccupancies,
      const unsigned int *     nonTrivialIdToElemIdMap,
      const unsigned int *     projecterKetTimesFlattenedVectorLocalIds,
      dataTypes::number *      nlpContractionContribution);
  } // namespace nonlocalPspEnergyDensityDeviceKernels
} // namespace dftfe
#endif
