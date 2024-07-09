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

#ifndef DFTFE_TESTTRANFERBETWEENMESHES_H
#define DFTFE_TESTTRANFERBETWEENMESHES_H

#include "TransferDataBetweenMeshesIncompatiblePartitioning.h"
#include "headers.h"
#include "triangulationManager.h"
#include "dftUtils.h"
#include "constraintMatrixInfo.h"
#include "vectorUtilities.h"

#include "dftParameters.h"

namespace dftfe
{
  /**
   * @brief Unit test that tests the transfer between meshes
   * @param mpi_comm_parent
   * @param mpi_comm_domain
   * @param interpoolcomm
   * @param interbandgroup_comm
   * @param FEOrder
   * @param dftParams
   * @param atomLocations
   * @param imageAtomLocations
   * @param imageIds
   * @param nearestAtomDistances
   * @param domainBoundingVectors
   * @param generateSerialTria
   * @param generateElectrostaticsTria
   */
  void
  testTransferFromDftToVxcMesh(
    const MPI_Comm &                        mpi_comm_parent,
    const MPI_Comm &                        mpi_comm_domain,
    const MPI_Comm &                        interpoolcomm,
    const MPI_Comm &                        interbandgroup_comm,
    const unsigned int                      FEOrder,
    const dftParameters &                   dftParams,
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &imageAtomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<double> &             nearestAtomDistances,
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const bool                              generateSerialTria,
    const bool                              generateElectrostaticsTria);

#ifdef DFTFE_WITH_DEVICE
  void
  testTransferFromDftToVxcMeshDevice(
    const MPI_Comm &                        mpi_comm_parent,
    const MPI_Comm &                        mpi_comm_domain,
    const MPI_Comm &                        interpoolcomm,
    const MPI_Comm &                        interbandgroup_comm,
    const unsigned int                      FEOrder,
    const dftParameters &                   dftParams,
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &imageAtomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<double> &             nearestAtomDistances,
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const bool                              generateSerialTria,
    const bool                              generateElectrostaticsTria);

#endif
} // namespace dftfe


#endif // DFTFE_TESTTRANFERBETWEENMESHES_H
