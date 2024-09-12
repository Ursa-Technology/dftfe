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
#include "FEBasisOperations.h"
#include "AtomCenteredSphericalFunctionBase.h"
#include "AtomicCenteredNonLocalOperator.h"
#include "AuxDensityMatrix.h"
#include "AtomPseudoWavefunctions.h"

namespace dftfe
{

  struct hubbardSpecies
  {
  public :
    unsigned int atomicNumber;
    double hubbardValue;
    unsigned int numProj;
    unsigned int numberSphericalFunc;
    unsigned int numberSphericalFuncSq;
    double initialOccupation;
    std::vector<unsigned int> nQuantumNum;
    std::vector<unsigned int> lQuantumNum;
  };

  enum class HubbardOccFieldType
  {
    In,
    Out,
    Residual
  };

  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class hubbard
  {

  public:
    hubbard(const MPI_Comm &mpi_comm_parent,
            const MPI_Comm &mpi_comm_domain,
            const MPI_Comm &mpi_comm_interPool);

    void init(std::shared_ptr<
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
         const dftParameters &dftParam,
         const std::string &                         scratchFolderName,
         const bool                               singlePrecNonLocalOperator,
         const bool updateNonlocalSparsity,
         const std::vector<std::vector<double>> &atomLocations,
         const std::vector<std::vector<double>> &atomLocationsFrac,
         const std::vector<int>                 &imageIds,
         const std::vector<std::vector<double>> &imagePositions,
         const std::vector<double>              &kPointCoordinates,
         const std::vector<double>  & kPointWeights,
         const std::vector <std::vector<double>> &domainBoundaries);

    void createAtomCenteredSphericalFunctionsForProjectors();

    void computeEnergyFromOccupationMatrix(double & hubbardEnergy,
                                           double & hubbardEnergyCorrection);
    void computeOccupationMatrix(const dftfe::utils::MemoryStorage<ValueType, memorySpace> *X,
                            const std::vector<std::vector<double>> &      orbitalOccupancy);

    void applyPotentialDueToHubbardCorrection(const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
                                         dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &dst,
                                         const unsigned int inputVecSize,
                                         const double factor,
                                         const unsigned int                                         kPointIndex,
                                         const unsigned int                                         spinIndex);

    void initialiseOperatorActionOnX(unsigned int kPointIndex);
    void setInitialOccMatrix();


    void initialiseFlattenedDataStructure(unsigned int numVectors);

    void initialiseCellWaveFunctionPointers(unsigned int numVectors);
    void
    computeHubbardOccNumberFromCTransOnX(
      const bool         isOccOut,
      const unsigned int vectorBlockSize,
      const unsigned int spinIndex,
      const unsigned int kpointIndex);

    void
    computeCouplingMatrix();

    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
      getCouplingMatrix(unsigned int spinIndex);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
    getOccMatIn();

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
    getOccMatRes();

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
    getOccMatOut();

    void setInOccMatrix(const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> & inputOccMatrix);



    const std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
    getNonLocalOperator();

  private:

    void computeResidualOccMat();
    void readHubbardInput( const std::vector<std::vector<double>> &atomLocations,
                                                      const std::vector<int>                 &imageIds,
                                                      const std::vector<std::vector<double>> &imagePositions);


      std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, memorySpace>>
      d_BasisOperatorMemPtr;
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        d_BasisOperatorHostPtr;

      std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
        d_nonLocalOperator;

      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        d_BLASWrapperMemPtr;

      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperHostPtr;
    std::map<unsigned int, hubbardSpecies> d_hubbardSpeciesData;
    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsMap;

    std::shared_ptr<AtomCenteredSphericalFunctionContainer> d_atomicProjectorFnsContainer;

    std::vector<double>  d_kPointWeights;
    std::vector <std::vector<double>> d_domainBoundaries;
    const dftParameters *d_dftParamsPtr;
    std::vector<double>  d_kPointCoordinates;

    unsigned int d_numKPoints;

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
    std::vector<unsigned int> d_mapAtomToAtomicNumber;

    double     d_spinPolarizedFactor ;
    unsigned int d_noOfSpin;
    std::string                                       d_dftfeScratchFolderName;

    std::vector<dftfe::utils::MemoryStorage<ValueType, memorySpace>> d_couplingMatrixEntries;

    std::map<HubbardOccFieldType, dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> d_occupationMatrix;
    unsigned int d_noSpecies;

    unsigned int d_densityQuadratureId, d_numberWaveFunctions;

    unsigned int d_numTotalOccMatrixEntriesPerSpin;
    std::vector<unsigned int> d_OccMatrixEntryStartForAtom;

    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> d_hubbNonLocalProjectorTimesVectorBlock;
    dftfe::utils::MemoryStorage<ValueType, memorySpace>
        d_cellWaveFunctionMatrixSrc, d_cellWaveFunctionMatrixDst;

    unsigned int d_cellsBlockSizeApply;
    unsigned int d_verbosity;
  };
}

#endif // DFTFE_EXE_HUBBARDCLASS_H
