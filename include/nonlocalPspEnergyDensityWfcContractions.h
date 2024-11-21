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

#ifndef nonlocalPspEnergyDensityWfcContractions_H_
#define nonlocalPspEnergyDensityWfcContractions_H_

#include "headers.h"
#include "dftParameters.h"
#include "FEBasisOperations.h"
#include "oncvClass.h"
#include <memory>
#include <BLASWrapper.h>

namespace dftfe
{
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
      const dftParameters &dftParams);
} // namespace dftfe
#endif
