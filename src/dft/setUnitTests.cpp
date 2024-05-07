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


namespace dftfe
{
  dftClass::runUnitTest()
  {
    pcout<<" Testing transfer between incompatible meshes in HOST\n";

    unitTest::testTransferFromParentToChildIncompatiblePartitioning(d_mpiCommParent,
                                  mpi_communicator,
                                  interpoolcomm,
                                  interBandGroupComm,
                                  d_dftParamsPtr->finiteElementPolynomialOrder,
                                  d_dftParamsPtr->meshSizeOuterBall,
                                  *d_dftParamsPtr,
                                  atomLocations,
                                  d_imagePositionsAutoMesh,
                                  d_imageIds,
                                  d_nearestAtomDistances,
                                  d_domainBoundingVectors,
                                  false,
                                  false);

    pcout<<" Testing MultiVector CG for Poisson Poisson problem in HOST\n";

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

  }
}