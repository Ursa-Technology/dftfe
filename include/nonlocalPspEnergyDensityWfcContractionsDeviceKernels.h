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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef nonlocalPspEnergyDensityWfcContractionsDeviceKernels_H_
#    define nonlocalPspEnergyDensityWfcContractionsDeviceKernels_H_

namespace dftfe
{
  namespace nonlocalPspEnergyDensityDeviceKernels
  {
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
      ValueType *         nlpContractionContribution);
  }
} // namespace dftfe
#  endif
#endif
