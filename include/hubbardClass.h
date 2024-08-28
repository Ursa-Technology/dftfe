// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTFE_EXE_HUBBARDCLASS_H
#define DFTFE_EXE_HUBBARDCLASS_H

#include "headers.h"

namespace dftfe
{

  struct atomProjectors
  {
  public :
    unsigned int atomicNumber;
    std::vector<
  };

  struct hubbardSpecies
  {
  public :
    unsigned int atomicNumber;
    double hubbardValue;
    unsigned int numProj;
    unsigned int numberSphericalFunc;
    unsigned int numberSphericalFuncSq;
    std::vector<unsigned int> nQuantumNum;
    std::vector<unsigned int> lQuantumNum;
  };

  template <dftfe::utils::MemorySpace memorySpace>
  class hubbard
  {

  public:
    hubbard(const MPI_Comm &mpi_comm_parent,
            const MPI_Comm &mpi_comm_domain,
            const MPI_Comm &mpi_comm_interPool);

    void init(std::shared_ptr<
           dftfe::basis::
             FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
           basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
         std::shared_ptr<
           dftfe::basis::FEBasisOperations<ValueType,
                                           double,
                                           dftfe::utils::MemorySpace::DEVICE>>
           basisOperationsDevicePtr,
#endif
         std::shared_ptr<
           dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
           BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
         std::shared_ptr<
           dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
           BLASWrapperPtrDevice,
#endif
         unsigned int                             densityQuadratureId,
         unsigned int                             localContributionQuadratureId,
         unsigned int                             sparsityPatternQuadratureId,
         unsigned int                             nlpspQuadratureId,
         unsigned int                             densityQuadratureIdElectro,
         std::shared_ptr<excManager<memorySpace>> excFunctionalPtr,
         const std::vector<std::vector<double>> & atomLocations,
         unsigned int                             numEigenValues,
         const bool                               singlePrecNonLocalOperator);

    void createAtomCenteredSphericalFunctionsForProjectors();
  private:



    std::map<unsigned int, hubbardSpecies> d_hubbardSpeciesData;
    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsMap;

    bool d_occupationMatrixHasBeenComputed;

    double d_atomOrbitalMaxLength;

    const MPI_Comm d_mpi_comm_parent;
    const MPI_Comm d_mpi_comm_domain;
    const MPI_Comm d_mpi_comm_interPool;

    unsigned int n_mpi_processes,this_mpi_process;

    unsigned int d_numSpins;
    std::vector<unsigned int> d_procLocalAtomId;

    dealii::ConditionalOStream   pcout;

    std::vector<double> d_atomicCoords;
    std::vector<std::vector<double>> d_periodicImagesCoords;
    std::vector<int> d_imageIds;
    std::vector<unsigned int> d_mapAtomToHubbardIds;

    dftfe::utils::MemoryStorage<ValueType, memorySpace> d_couplingMatrixEntries;

    std::vector<std::vector<std::vector<dataTypes::number>>> d_occupationMatrix;

  };
}

#endif // DFTFE_EXE_HUBBARDCLASS_H
