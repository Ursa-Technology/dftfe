// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
//
// @authors Bikash Kanungo, Vishal Subramanian
//

#ifndef DFTFE_INVERSEDFTSOLVERFUNCTION_H
#define DFTFE_INVERSEDFTSOLVERFUNCTION_H

#include <transferDataBetweenMeshesBase.h>
#include <headers.h>
#include <multiVectorAdjointProblem.h>
#include <multiVectorLinearMINRESSolver.h>
#include <multiVectorAdjointProblemDevice.h>
#include <multiVectorLinearMINRESSolverDevice.h>
#include <nonlinearSolverFunction.h>
#include <constraintMatrixInfo.h>
#include <vectorUtilities.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <dft.h>
namespace dftfe
{
  /**
   * @brief Class implementing the inverse DFT problem
   *
   */
  template <typename T, dftfe::utils::MemorySpace memorySpace>
  class InverseDFTSolverFunction
  {
  public:
    /**
     * @brief Constructor
     */
    InverseDFTSolverFunction(const MPI_Comm &mpi_comm_parent,
                             const MPI_Comm &mpi_comm_domain,
                             const MPI_Comm &mpi_comm_interpool,
                             const MPI_Comm &mpi_comm_interband);

    //
    // reinit
    //
    void
    reinit(
      const dftfe::utils::MemoryStorage<double , mmeorySpace> &rhoTargetQuadData,
      const dftfe::utils::MemoryStorage<double , mmeorySpace> &weightQuadData,
      const dftfe::utils::MemoryStorage<double , mmeorySpace> &potBaseQuadData,
      dftBase *                                            dft,
      const dealii::MatrixFree<3, double> &                matrixFreeDataParent,
      const dealii::MatrixFree<3, double> &                matrixFreeDataChild,
      const dealii::AffineConstraints<double>
        &constraintMatrixHomogeneousPsi, // assumes that the constraint matrix
                                         // has homogenous BC
      const dealii::AffineConstraints<double>
        &constraintMatrixHomogeneousAdjoint, // assumes that the constraint
                                             // matrix has homogenous BC
      const dealii::AffineConstraints<double> &constraintMatrixPot,
      KohnShamHamiltonianOperator<
        memorySpace> & ksHamiltonianObj,
      std::shared_ptr<TransferDataBetweenMeshesBase> &  inverseDFTDoFManagerObjPtr,
      const std::vector<double> &kpointWeights,
      const unsigned int         numSpins,
      const unsigned int         numEigenValues,
      const unsigned int         matrixFreePsiVectorComponent,
      const unsigned int         matrixFreeAdjointVectorComponent,
      const unsigned int         matrixFreePotVectorComponent,
      const unsigned int         matrixFreeQuadratureComponentAdjointRhs,
      const unsigned int         matrixFreeQuadratureComponentPot,
      const bool                 isComputeDiagonalA,
      const bool                 isComputeShapeFunction,
      const dftParameters &      dftParams);


    void
    writeVxcDataToFile(std::vector<distributedCPUVec<double>> &pot,
                       unsigned int                            counter);

    void
    solveEigen(const std::vector<distributedCPUVec<double>> &pot);

    void
    dotProduct(const distributedCPUVec<double> &vec1,
               const distributedCPUVec<double> &vec2,
               unsigned int                     blockSize,
               std::vector<double> &            outputDot);

    void
    setInitialGuess(const std::vector<distributedCPUVec<double>> &pot,
                    const std::vector<std::vector<std::vector<double>>>
                      &targetPotValuesParentQuadData);

    std::vector<distributedCPUVec<double>>
    getInitialGuess() const;

    void
    getForceVector(std::vector<dftfe::linearAlgebra::MultiVector<double,
                                                                 dftfe::utils::MemorySpace::HOST>> &pot,
                   std::vector<dftfe::linearAlgebra::MultiVector<double,
                                                                 dftfe::utils::MemorySpace::HOST>> &force,
                   std::vector<double> &                   loss);



    void
    setSolution(const std::vector<dftfe::linearAlgebra::MultiVector<double,
                                                                    dftfe::utils::MemorySpace::HOST>> &pot);

  private:
    std::vector<dftfe::linearAlgebra::MultiVector<double,
                                                  dftfe::utils::MemorySpace::HOST>> d_pot;
    std::vector<dftfe::linearAlgebra::MultiVector<double,
                                                  dftfe::utils::MemorySpace::HOST>>
      d_solutionPotVecForWritingInParentNodes;
    std::vector<dftfe::linearAlgebra::MultiVector<double,
                                                  dftfe::utils::MemorySpace::HOST>>
                                                  d_solutionPotVecForWritingInParentNodesMFVec;
    std::vector<std::vector<std::vector<double>>> d_rhoTargetQuadData;
    std::vector<std::vector<std::vector<double>>> d_rhoKSQuadData;
    std::vector<std::vector<std::vector<double>>> d_weightQuadData;
    std::vector<std::vector<std::vector<double>>> d_potBaseQuadData;
    dftfe::linearAlgebra::MultiVector<double,
      dftfe::utils::MemorySpace::HOST>>                     d_adjointBlock;
    MultiVectorAdjointLinearSolverProblem<memorySpace> d_multiVectorAdjointProblem;
    MultiVectorMinResSolver d_multiVectorLinearMINRESSolver;

    std::vector<dealii::types::global_dof_index>
      fullFlattenedArrayCellLocalProcIndexIdMapPsi,
      fullFlattenedArrayCellLocalProcIndexIdMapAdjoint;

    // TODO remove this from gerForceVectorCPU
    distributedCPUMultiVec<double> psiBlockVec,
      adjointInhomogenousDirichletValues,
      multiVectorAdjointOutputWithPsiConstraints,
      multiVectorAdjointOutputWithAdjointConstraints;
    dftUtils::constraintMatrixInfo constraintsMatrixPsiDataInfo,
      constraintsMatrixAdjointDataInfo;


    const dealii::MatrixFree<3, double> *    d_matrixFreeDataParent;
    const dealii::MatrixFree<3, double> *    d_matrixFreeDataChild;
    const dealii::AffineConstraints<double> *d_constraintMatrixHomogeneousPsi;
    const dealii::AffineConstraints<double>
                                            *d_constraintMatrixHomogeneousAdjoint;
    const dealii::AffineConstraints<double> *d_constraintMatrixPot;
    dftUtils::constraintMatrixInfo           d_constraintsMatrixDataInfoPot;
    KohnShamHamiltonianOperator<
      memorySpace> *                       d_kohnShamClass;

    std::shared_ptr<TransferDataBetweenMeshesBase> d_transferDataPtr;
    std::vector<double>   d_kpointWeights;

    std::vector<double> d_childCellJxW, d_childCellShapeFunctionValue;
    std::vector<double> d_parentCellJxW, d_shapeFunctionValueParent;
    unsigned int        d_numSpins;
    unsigned int        d_numKPoints;
    unsigned int        d_matrixFreePsiVectorComponent;
    unsigned int        d_matrixFreeAdjointVectorComponent;
    unsigned int        d_matrixFreePotVectorComponent;
    unsigned int        d_matrixFreeQuadratureComponentAdjointRhs;
    unsigned int        d_matrixFreeQuadratureComponentPot;
    bool                d_isComputeDiagonalA;
    bool                d_isComputeShapeFunction;
    double              d_degeneracyTol;
    double              d_adjointTol;
    int                 d_adjointMaxIterations;
    MPI_Comm            d_mpi_comm_parent;
    MPI_Comm            d_mpi_comm_domain;
    MPI_Comm            d_mpi_comm_interband;
    MPI_Comm            d_mpi_comm_interpool;

    unsigned int                 d_numLocallyOwnedCellsParent;
    unsigned int                 d_numLocallyOwnedCellsChild;
    const dealii::DoFHandler<3> *d_dofHandlerParent;
    const dealii::DoFHandler<3> *d_dofHandlerChild;
    const dftParameters *        d_dftParams;

    distributedCPUVec<double>        d_MInvSqrt;
    distributedCPUVec<double>        d_MSqrt;
    unsigned int                     d_numEigenValues;
    std::vector<std::vector<double>> d_fractionalOccupancy;

    std::vector<double> d_wantedLower;
    std::vector<double> d_unwantedUpper;
    std::vector<double> d_unwantedLower;
    unsigned int        d_getForceCounter;
    double              d_fractionalOccupancyTol;
    dftBase *           d_dft;

    unsigned int                                    d_maxChebyPasses;
    double                                          d_lossPreviousIteration;
    double                                          d_tolForChebFiltering;
    elpaScalaManager *                              d_elpaScala;
    chebyshevOrthogonalizedSubspaceIterationSolver *d_subspaceIterationSolver;

    std::vector<std::vector<double>> d_residualNormWaveFunctions;
    std::vector<std::vector<double>> d_eigenValues;
    unsigned int                     d_numElectrons;

    dealii::ConditionalOStream pcout;

    // TODO implemented for debugging purpose
    std::vector<std::vector<std::vector<double>>>
      d_targetPotValuesParentQuadData;

    bool d_resizeGPUVecDuringInterpolation, d_resizeCPUVecDuringInterpolation;
  };
} // end of namespace dftfe
#endif // DFTFE_INVERSEDFTSOLVERFUNCTION_H
