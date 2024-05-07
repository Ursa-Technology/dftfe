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

#ifndef DFTFE_UNITTESTS_H
#define DFTFE_UNITTESTS_H

namespace unitTest
{
  void testTransferFromParentToChildIncompatiblePartitioning( const MPI_Comm &                  mpi_comm_parent,
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

  void
  testMultiVectorPoissonSolver(
    const dealii::MatrixFree<3, double> &          matrixFreeData,
    const dealii::AffineConstraints<double> &      constraintMatrix,
    std::map<dealii::CellId, std::vector<double>> &inputVec,
    const unsigned int                             matrixFreeVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhsDensity,
    const unsigned int matrixFreeQuadratureComponentAX,
    const MPI_Comm &   mpi_comm_parent,
    const MPI_Comm &   mpi_comm_domain);

} // end of namespace unitTest

#endif // DFTFE_UNITTESTS_H
