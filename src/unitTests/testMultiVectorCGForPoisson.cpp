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

#include "unitTest.h"



namespace unitTest
{
  void
  testMultiVectorPoissonSolver(
    const dealii::MatrixFree<3, double> &          matrixFreeData,
    const dealii::AffineConstraints<double> &      constraintMatrix,
    std::map<dealii::CellId, std::vector<double>> &inputVec,
    const unsigned int                             matrixFreeVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhsDensity,
    const unsigned int matrixFreeQuadratureComponentAX,
    const MPI_Comm &   mpi_comm_parent,
    const MPI_Comm &   mpi_comm_domain)
  {
    //        int this_process;
    //        MPI_Comm_rank(mpi_comm, &this_process);
    //        if(this_process == 0)
    //        {

    //        }

    //    MultiVectorLinearCGSolver
    //    linearSolver(mpi_comm_parent,mpi_comm_domain);
    MultiVectorLinearMINRESSolver   linearSolver(mpi_comm_parent,
                                               mpi_comm_domain);
    MultiVectorPoissonSolverProblem multiPoissonSolver(mpi_comm_parent,
                                                       mpi_comm_domain);
    const dealii::DoFHandler<3> *   d_dofHandler;
    d_dofHandler = &matrixFreeData.get_dof_handler(matrixFreeVectorComponent);

    unsigned int blockSizeInput = 5;
    std::cout << " Changing block Size to " << blockSizeInput << "\n";

    std::vector<std::vector<double>> rhoQuadInputValues;

    unsigned int totalLocallyOwnedCells = matrixFreeData.n_physical_cells();
    rhoQuadInputValues.resize(totalLocallyOwnedCells);

    // set up solver functions for Poisson
    poissonSolverProblem<6, 8> phiTotalSolverProblem(mpi_comm_domain);
    std::cout
      << " testing multiVector Poisson Solve with FeOrder = 6, FeOrderElectro = 8\n";
    // Reinit poisson solver problem to not include atoms
    dealii::AffineConstraints<double> d_constraintMatrixSingleHangingPeriodic,
      d_constraintMatrixSingleHangingPeriodicHomogeneous,
      d_constraintMatrixSingleHangingPeriodicInhomogeneous;
    d_constraintMatrixSingleHangingPeriodic.clear();
    dealii::DoFTools::make_hanging_node_constraints(
      *d_dofHandler, d_constraintMatrixSingleHangingPeriodic);
    d_constraintMatrixSingleHangingPeriodic.close();
    d_constraintMatrixSingleHangingPeriodicHomogeneous.clear();
    dealii::DoFTools::make_hanging_node_constraints(
      *d_dofHandler, d_constraintMatrixSingleHangingPeriodicHomogeneous);


    dealii::VectorTools::interpolate_boundary_values(
      *d_dofHandler,
      0,
      dealii::Functions::ZeroFunction<3>(),
      d_constraintMatrixSingleHangingPeriodicHomogeneous);
    d_constraintMatrixSingleHangingPeriodicHomogeneous.close();

    d_constraintMatrixSingleHangingPeriodic.clear();
    dealii::DoFTools::make_hanging_node_constraints(
      *d_dofHandler, d_constraintMatrixSingleHangingPeriodic);
    d_constraintMatrixSingleHangingPeriodic.close();

    distributedCPUVec<double>      expectedOutput;
    distributedCPUMultiVec<double> multiExpectedOutput;
    // set up linear solver
    dealiiLinearSolver dealiiCGSolver(mpi_comm_parent,
                                      mpi_comm_domain,
                                      dealiiLinearSolver::CG);
    std::map<dealii::types::global_dof_index, double> atoms;
    std::map<dealii::CellId, std::vector<double>>     smearedChargeValues;
    vectorTools::createDealiiVector<double>(
      matrixFreeData.get_vector_partitioner(matrixFreeVectorComponent),
      1,
      expectedOutput);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      matrixFreeData.get_vector_partitioner(matrixFreeVectorComponent),
      blockSizeInput,
      multiExpectedOutput);

    distributedCPUMultiVec<double> multiVectorOutput;
    distributedCPUMultiVec<double> boundaryValues;
    multiVectorOutput.reinit(multiExpectedOutput);
    boundaryValues.reinit(multiExpectedOutput);
    //    boundaryValues = 0.0;
    boundaryValues.setValue(0.0);

    for (unsigned int iBlockId = 0; iBlockId < blockSizeInput; iBlockId++)
      {
        distributedCPUVec<double> singleBoundaryCond;
        singleBoundaryCond.reinit(expectedOutput);
        singleBoundaryCond = 0.0;
        d_constraintMatrixSingleHangingPeriodicInhomogeneous.clear();
        dealii::DoFTools::make_hanging_node_constraints(
          *d_dofHandler, d_constraintMatrixSingleHangingPeriodicInhomogeneous);

        const unsigned int vertices_per_cell =
          dealii::GeometryInfo<3>::vertices_per_cell;
        const unsigned int dofs_per_cell = d_dofHandler->get_fe().dofs_per_cell;
        const unsigned int faces_per_cell =
          dealii::GeometryInfo<3>::faces_per_cell;
        const unsigned int dofs_per_face = d_dofHandler->get_fe().dofs_per_face;

        std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
          dofs_per_cell);
        std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
          dofs_per_face);

        std::vector<bool> dofs_touched(d_dofHandler->n_dofs(), false);
        dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandler
                                                             ->begin_active(),
                                                    endc = d_dofHandler->end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned() || cell->is_ghost())
            {
              cell->get_dof_indices(cellGlobalDofIndices);
              for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
                {
                  const unsigned int boundaryId =
                    cell->face(iFace)->boundary_id();
                  if (boundaryId == 0)
                    {
                      cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                      for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                           ++iFaceDof)
                        {
                          const dealii::types::global_dof_index nodeId =
                            iFaceGlobalDofIndices[iFaceDof];
                          if (dofs_touched[nodeId])
                            continue;
                          dofs_touched[nodeId] = true;
                          if (
                            !d_constraintMatrixSingleHangingPeriodicInhomogeneous
                               .is_constrained(nodeId))
                            {
                              d_constraintMatrixSingleHangingPeriodicInhomogeneous
                                .add_line(nodeId);
                              d_constraintMatrixSingleHangingPeriodicInhomogeneous
                                .set_inhomogeneity(nodeId, iBlockId + 2);
                              //                                d_constraintMatrixSingleHangingPeriodicInhomogeneous.set_inhomogeneity(nodeId,
                              //                                0);
                            } // non-hanging node check

                        } // Face dof loop
                    }     // non-periodic boundary id
                }         // Face loop
            }             // cell locally owned
        d_constraintMatrixSingleHangingPeriodicInhomogeneous.close();
        d_constraintMatrixSingleHangingPeriodicInhomogeneous.distribute(
          singleBoundaryCond);
        for (unsigned int iNodeId = 0;
             iNodeId < singleBoundaryCond.local_size();
             iNodeId++)
          {
            boundaryValues.data()[iNodeId * blockSizeInput + iBlockId] =
              singleBoundaryCond.local_element(iNodeId);
          }
        boundaryValues.updateGhostValues();
      }


    unsigned int                                iElem = 0;
    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  d_dofHandler->begin_active(),
                                                endc = d_dofHandler->end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          std::vector<double> &cellLevelQuadInput =
            inputVec.find(cell->id())->second;
          rhoQuadInputValues[iElem].resize(cellLevelQuadInput.size() *
                                           blockSizeInput);
          for (unsigned int i = 0; i < cellLevelQuadInput.size(); i++)
            {
              for (unsigned int k = 0; k < blockSizeInput; k++)
                {
                  rhoQuadInputValues[iElem][i * blockSizeInput + k] =
                    cellLevelQuadInput[i] * (k + 1);
                }
            }
          iElem++;
        }

    for (unsigned int k = 0; k < blockSizeInput; k++)
      {
        expectedOutput = 0;
        d_constraintMatrixSingleHangingPeriodicInhomogeneous.clear();
        dealii::DoFTools::make_hanging_node_constraints(
          *d_dofHandler, d_constraintMatrixSingleHangingPeriodicInhomogeneous);

        const unsigned int vertices_per_cell =
          dealii::GeometryInfo<3>::vertices_per_cell;
        const unsigned int dofs_per_cell = d_dofHandler->get_fe().dofs_per_cell;
        const unsigned int faces_per_cell =
          dealii::GeometryInfo<3>::faces_per_cell;
        const unsigned int dofs_per_face = d_dofHandler->get_fe().dofs_per_face;

        std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
          dofs_per_cell);
        std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
          dofs_per_face);

        std::vector<bool> dofs_touched(d_dofHandler->n_dofs(), false);
        dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandler
                                                             ->begin_active(),
                                                    endc = d_dofHandler->end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned() || cell->is_ghost())
            {
              cell->get_dof_indices(cellGlobalDofIndices);
              for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
                {
                  const unsigned int boundaryId =
                    cell->face(iFace)->boundary_id();
                  if (boundaryId == 0)
                    {
                      cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
                      for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                           ++iFaceDof)
                        {
                          const dealii::types::global_dof_index nodeId =
                            iFaceGlobalDofIndices[iFaceDof];
                          if (dofs_touched[nodeId])
                            continue;
                          dofs_touched[nodeId] = true;
                          if (
                            !d_constraintMatrixSingleHangingPeriodicInhomogeneous
                               .is_constrained(nodeId))
                            {
                              d_constraintMatrixSingleHangingPeriodicInhomogeneous
                                .add_line(nodeId);
                              d_constraintMatrixSingleHangingPeriodicInhomogeneous
                                .set_inhomogeneity(nodeId, k + 2);
                              //                                  d_constraintMatrixSingleHangingPeriodicInhomogeneous.set_inhomogeneity(nodeId,
                              //                                  0);
                            } // non-hanging node check

                        } // Face dof loop
                    }     // non-periodic boundary id
                }         // Face loop
            }             // cell locally owned
        d_constraintMatrixSingleHangingPeriodicInhomogeneous.close();
        for (std::map<dealii::CellId, std::vector<double>>::iterator it =
               inputVec.begin();
             it != inputVec.end();
             it++)
          {
            std::vector<double> &cellLevelQuadInput = it->second;

            for (unsigned int i = 0; i < cellLevelQuadInput.size(); i++)
              {
                //                cellLevelQuadInput[i]  = 0.0;
                if ((k != 0) && (k != 1))
                  cellLevelQuadInput[i] = cellLevelQuadInput[i] * (k + 1) / (k);
                else
                  cellLevelQuadInput[i] = cellLevelQuadInput[i] * (k + 1);
              }
          }
        phiTotalSolverProblem.reinit(
          matrixFreeData,
          expectedOutput,
          d_constraintMatrixSingleHangingPeriodicInhomogeneous,
          matrixFreeVectorComponent,
          matrixFreeQuadratureComponentRhsDensity,
          matrixFreeQuadratureComponentAX,
          atoms,
          smearedChargeValues,
          matrixFreeQuadratureComponentAX,
          inputVec,
          true,  // isComputeDiagonalA
          false, // isComputeMeanValueConstraint
          false, // smearedNuclearCharges
          true,  // isRhoValues
          false, // isGradSmearedChargeRhs
          0,     // smearedChargeGradientComponentId
          false, // storeSmearedChargeRhs
          false, // reuseSmearedChargeRhs
          true); // reinitializeFastConstraints

        dealiiCGSolver.solve(phiTotalSolverProblem, 1e-10, 10000, 4);

        //        std::cout<<" norm expected =
        //        "<<expectedOutput.l2_norm()<<"\n";
        //        dealii::types::global_dof_index indexVec;
        //        for (dealii::types::global_dof_index i = 0; i <
        //        expectedOutput.size();
        //             i++)
        //          {
        //            indexVec = i * blockSizeInput + k;
        //            if (expectedOutput.in_local_range(i))
        //              multiExpectedOutput(indexVec) = expectedOutput(i);
        //          }
      }



    std::cout << "Testing multi Vector Poisson Solve";
    std::cout << " Value of matrixFreeComponent = " << matrixFreeVectorComponent
              << "\n";
    std::cout << " Value of QuadRhs             = "
              << matrixFreeQuadratureComponentRhsDensity << "\n";
    std::cout << " Value of QuadAX              = "
              << matrixFreeQuadratureComponentAX << "\n";
    std::cout << " Size of expectedOutput       = " << expectedOutput.size()
              << "\n";



    multiPoissonSolver.reinit(
      matrixFreeData,
      d_constraintMatrixSingleHangingPeriodicHomogeneous,
      matrixFreeVectorComponent,
      matrixFreeQuadratureComponentRhsDensity,
      matrixFreeQuadratureComponentAX,
      true,
      true);

    //        if(this_process == 0) {
    //      boundaryValues = 0.0;
    std::cout << "Size of input  = " << inputVec.size() << "\n";
    std::cout << "Size of output = " << multiVectorOutput.locallyOwnedSize()
              << "\n";
    std::cout << "Size of boundary = " << boundaryValues.locallyOwnedSize()
              << "\n";
    //    std::cout << " Norm of boundary = " << boundaryValues.l2Norm() <<
    //    "\n";
    //        }

    //    linearSolver.solve(multiPoissonSolver,
    //                   rhoQuadInputValues,
    //                   multiVectorOutput,
    //                   boundaryValues,
    //                   blockSizeInput,
    //                   1e-10,
    //                   1000,
    //                   4,
    //                   true);


    linearSolver.solve(multiPoissonSolver,
                       rhoQuadInputValues,
                       multiVectorOutput,
                       boundaryValues,
                       blockSizeInput,
                       1e-10,
                       10000,
                       4,
                       true);

    //    std::cout << "Norm of expected input = " <<
    //    multiExpectedOutput.l2Norm()
    //              << "\n";
    //    std::cout<<"Norm of output from multiPoisson =
    //    "<<multiVectorOutput.l2Norm()<<"\n";

    //    multiVectorOutput -= multiExpectedOutput;
    //      std::cout << "Vector L2 norm of the difference is : "
    //                << multiVectorOutput.l2Norm() << "\n";

    //
    //      // Changing the block size
    //      blockSizeInput = 3;
    //      std::cout<<" Changing block Size to "<<blockSizeInput<<"\n";
    //
    //      vectorTools::createDealiiVector<double>(
    //              matrixFreeData.get_vector_partitioner(matrixFreeVectorComponent),
    //              1,
    //              expectedOutput);
    //      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
    //              matrixFreeData.get_vector_partitioner(matrixFreeVectorComponent),
    //              blockSizeInput,
    //              multiExpectedOutput);
    //      multiVectorOutput.reinit(multiExpectedOutput);
    //      boundaryValues.reinit(multiExpectedOutput);
    //      boundaryValues = 0.0;
    //
    //      for (unsigned int iBlockId = 0; iBlockId < blockSizeInput;
    //      iBlockId++)
    //      {
    //          distributedCPUMultiVec<double> singleBoundaryCond;
    //          singleBoundaryCond.reinit(expectedOutput);
    //          singleBoundaryCond  = 0.0;
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.clear();
    //          dealii::DoFTools::make_hanging_node_constraints(*d_dofHandler,
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous);
    //
    //          const unsigned int vertices_per_cell =
    //                  dealii::GeometryInfo<3>::vertices_per_cell;
    //          const unsigned int dofs_per_cell  =
    //          d_dofHandler->get_fe().dofs_per_cell; const unsigned int
    //          faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell; const
    //          unsigned int dofs_per_face  =
    //          d_dofHandler->get_fe().dofs_per_face;
    //
    //          std::vector<dealii::types::global_dof_index>
    //          cellGlobalDofIndices(dofs_per_cell);
    //          std::vector<dealii::types::global_dof_index>
    //          iFaceGlobalDofIndices(dofs_per_face);
    //
    //          std::vector<bool> dofs_touched(d_dofHandler->n_dofs(), false);
    //          dealii::DoFHandler<3>::active_cell_iterator cell =
    //          d_dofHandler->begin_active(),
    //                  endc = d_dofHandler->end();
    //          for (; cell != endc; ++cell)
    //              if (cell->is_locally_owned() || cell->is_ghost())
    //              {
    //                  cell->get_dof_indices(cellGlobalDofIndices);
    //                  for (unsigned int iFace = 0; iFace < faces_per_cell;
    //                  ++iFace)
    //                  {
    //                      const unsigned int boundaryId =
    //                      cell->face(iFace)->boundary_id(); if (boundaryId ==
    //                      0)
    //                      {
    //                          cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
    //                          for (unsigned int iFaceDof = 0; iFaceDof <
    //                          dofs_per_face;
    //                               ++iFaceDof)
    //                          {
    //                              const dealii::types::global_dof_index nodeId
    //                              =
    //                                      iFaceGlobalDofIndices[iFaceDof];
    //                              if (dofs_touched[nodeId])
    //                                  continue;
    //                              dofs_touched[nodeId] = true;
    //                              if
    //                              (!d_constraintMatrixSingleHangingPeriodicInhomogeneous.is_constrained(nodeId))
    //                              {
    //                                  d_constraintMatrixSingleHangingPeriodicInhomogeneous.add_line(nodeId);
    //                                  d_constraintMatrixSingleHangingPeriodicInhomogeneous.set_inhomogeneity(nodeId,
    //                                  iBlockId );
    //                              } // non-hanging node check
    //
    //                          }     // Face dof loop
    //                      }         // non-periodic boundary id
    //                  }             // Face loop
    //              }                 // cell locally owned
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.close();
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.distribute(singleBoundaryCond);
    //          for (unsigned int iNodeId = 0; iNodeId <
    //          singleBoundaryCond.local_size(); iNodeId ++)
    //          {
    //              boundaryValues.data()[iNodeId*blockSizeInput + iBlockId ] =
    //              singleBoundaryCond.local_element(iNodeId);
    //          }
    //          boundaryValues.updateGhostValues();
    //      }
    //
    //
    //      iElem = 0;
    //      cell = d_dofHandler->begin_active();
    //      for (; cell != endc; ++cell)
    //          if (cell->is_locally_owned() )
    //          {
    //              std::vector<double> &cellLevelQuadInput =
    //              inputVec.find(cell->id())->second;
    //              rhoQuadInputValues[iElem].resize(cellLevelQuadInput.size() *
    //              blockSizeInput); for (unsigned int i = 0; i <
    //              cellLevelQuadInput.size(); i++)
    //              {
    //                  for (unsigned int k = 0; k < blockSizeInput; k++)
    //                  {
    //                      rhoQuadInputValues[iElem][i * blockSizeInput + k] =
    //                              cellLevelQuadInput[i] * (k + 1);
    //                  }
    //              }
    //              iElem++;
    //
    //          }
    //
    //      for (unsigned int k = 0; k < blockSizeInput; k++)
    //      {
    //          expectedOutput = 0;
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.clear();
    //          dealii::DoFTools::make_hanging_node_constraints(*d_dofHandler,
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous);
    //
    //          const unsigned int vertices_per_cell =
    //                  dealii::GeometryInfo<3>::vertices_per_cell;
    //          const unsigned int dofs_per_cell  =
    //          d_dofHandler->get_fe().dofs_per_cell; const unsigned int
    //          faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell; const
    //          unsigned int dofs_per_face  =
    //          d_dofHandler->get_fe().dofs_per_face;
    //
    //          std::vector<dealii::types::global_dof_index>
    //          cellGlobalDofIndices(dofs_per_cell);
    //          std::vector<dealii::types::global_dof_index>
    //          iFaceGlobalDofIndices(dofs_per_face);
    //
    //          std::vector<bool> dofs_touched(d_dofHandler->n_dofs(), false);
    //          dealii::DoFHandler<3>::active_cell_iterator cell =
    //          d_dofHandler->begin_active(),
    //                  endc = d_dofHandler->end();
    //          for (; cell != endc; ++cell)
    //              if (cell->is_locally_owned() || cell->is_ghost())
    //              {
    //                  cell->get_dof_indices(cellGlobalDofIndices);
    //                  for (unsigned int iFace = 0; iFace < faces_per_cell;
    //                  ++iFace)
    //                  {
    //                      const unsigned int boundaryId =
    //                      cell->face(iFace)->boundary_id(); if (boundaryId ==
    //                      0)
    //                      {
    //                          cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
    //                          for (unsigned int iFaceDof = 0; iFaceDof <
    //                          dofs_per_face;
    //                               ++iFaceDof)
    //                          {
    //                              const dealii::types::global_dof_index nodeId
    //                              =
    //                                      iFaceGlobalDofIndices[iFaceDof];
    //                              if (dofs_touched[nodeId])
    //                                  continue;
    //                              dofs_touched[nodeId] = true;
    //                              if
    //                              (!d_constraintMatrixSingleHangingPeriodicInhomogeneous.is_constrained(nodeId))
    //                              {
    //                                  d_constraintMatrixSingleHangingPeriodicInhomogeneous.add_line(nodeId);
    //                                  d_constraintMatrixSingleHangingPeriodicInhomogeneous.set_inhomogeneity(nodeId,
    //                                  k );
    //                              } // non-hanging node check
    //
    //                          }     // Face dof loop
    //                      }         // non-periodic boundary id
    //                  }             // Face loop
    //              }                 // cell locally owned
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.close();
    //          for (std::map<dealii::CellId, std::vector<double>>::iterator it
    //          = inputVec.begin(); it != inputVec.end(); it++)
    //          {
    //              std::vector<double> &cellLevelQuadInput = it->second;
    //
    //              for (unsigned int i = 0; i < cellLevelQuadInput.size(); i++)
    //              {
    //                  if ((k != 0) && (k != 1))
    //                      cellLevelQuadInput[i] = cellLevelQuadInput[i] * (k +
    //                      1) / (k);
    //                  else
    //                      cellLevelQuadInput[i] = cellLevelQuadInput[i] * (k +
    //                      1);
    //              }
    //          }
    //          phiTotalSolverProblem.reinit(matrixFreeData,
    //                                       expectedOutput,
    //                                       d_constraintMatrixSingleHangingPeriodicInhomogeneous,
    //                                       matrixFreeVectorComponent,
    //                                       matrixFreeQuadratureComponentRhsDensity,
    //                                       matrixFreeQuadratureComponentAX,
    //                                       atoms,
    //                                       smearedChargeValues,
    //                                       matrixFreeQuadratureComponentAX,
    //                                       inputVec,
    //                                       true,   // isComputeDiagonalA
    //                                       false,  //
    //                                       isComputeMeanValueConstraint false,
    //                                       // smearedNuclearCharges true,   //
    //                                       isRhoValues false,  //
    //                                       isGradSmearedChargeRhs 0,      //
    //                                       smearedChargeGradientComponentId
    //                                       false,  // storeSmearedChargeRhs
    //                                       false, // reuseSmearedChargeRhs
    //                                       true);
    //                                       //reinitializeFastConstraints
    //
    //          dealiiCGSolver.solve(phiTotalSolverProblem, 1e-10, 1000, 4);
    //
    //          dealii::types::global_dof_index indexVec;
    //          for (dealii::types::global_dof_index i = 0; i <
    //          expectedOutput.size();
    //               i++)
    //          {
    //              indexVec = i * blockSizeInput + k;
    //              if (multiExpectedOutput.in_local_range(indexVec))
    //                  multiExpectedOutput(indexVec) = expectedOutput(i);
    //          }
    //      }
    //
    //
    //
    //
    //      std::cout << "Testing multi Vector Poisson Solve";
    //      std::cout << " Value of matrixFreeComponent = " <<
    //      matrixFreeVectorComponent
    //                << "\n";
    //      std::cout << " Value of QuadRhs             = "
    //                << matrixFreeQuadratureComponentRhsDensity << "\n";
    //      std::cout << " Value of QuadAX              = "
    //                << matrixFreeQuadratureComponentAX <<
    //      "\n";
    //      std::cout << " Size of expectedOutput       = " <<
    //      expectedOutput.size()
    //                << "\n";
    //
    //      //        if(this_process == 0) {
    ////      boundaryValues = 0.0;
    //      std::cout << "Size of input  = " << inputVec.size() << "\n";
    //      std::cout << "Size of output = " <<
    //      multiVectorOutput.locallyOwnedSize() << "\n"; std::cout << "Size of
    //      boundary = " << boundaryValues.locallyOwnedSize() << "\n"; std::cout
    //      << " Norm of boundary = " << boundaryValues.l2Norm() << "\n";
    //      //        }
    //
    //      linearSolver.solve(multiPoissonSolver,
    //                         rhoQuadInputValues,
    //                         multiVectorOutput,
    //                         boundaryValues,
    //                         blockSizeInput,
    //                         1e-10,
    //                         1000,
    //                         4,
    //                         true);
    //
    ////      shift.resize(blockSizeInput,0.0);
    ////      linearSolver.solve(multiPoissonSolver,
    ////                     rhoQuadInputValues,
    ////                     multiVectorOutput,
    ////                     boundaryValues,
    ////                     blockSizeInput,
    ////                     1e-10,
    ////                     1000,
    ////                     4,
    ////                     true);
    //
    //      std::cout << "Norm of expected input = " <<
    //      multiExpectedOutput.l2Norm()
    //                << "\n";
    //      std::cout<<"Norm of output from multiPoisson =
    //      "<<multiVectorOutput.l2Norm()<<"\n";
    //
    //      multiVectorOutput -= multiExpectedOutput;
    //      std::cout << "Vector L2 norm of the difference is : "
    //                << multiVectorOutput.l2Norm() << "\n";
    //
    //
    //      // Changing blck size again.
    //      // Changing the block size
    //      blockSizeInput = 5;
    //      std::cout<<" Changing block Size to "<<blockSizeInput<<"\n";
    //
    //
    //      vectorTools::createDealiiVector<double>(
    //              matrixFreeData.get_vector_partitioner(matrixFreeVectorComponent),
    //              1,
    //              expectedOutput);
    //      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
    //              matrixFreeData.get_vector_partitioner(matrixFreeVectorComponent),
    //              blockSizeInput,
    //              multiExpectedOutput);
    //      multiVectorOutput.reinit(multiExpectedOutput);
    //      boundaryValues.reinit(multiExpectedOutput);
    //      boundaryValues = 0.0;
    //
    //      for (unsigned int iBlockId = 0; iBlockId < blockSizeInput;
    //      iBlockId++)
    //      {
    //          distributedCPUMultiVec<double> singleBoundaryCond;
    //          singleBoundaryCond.reinit(expectedOutput);
    //          singleBoundaryCond  = 0.0;
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.clear();
    //          dealii::DoFTools::make_hanging_node_constraints(*d_dofHandler,
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous);
    //
    //          const unsigned int vertices_per_cell =
    //                  dealii::GeometryInfo<3>::vertices_per_cell;
    //          const unsigned int dofs_per_cell  =
    //          d_dofHandler->get_fe().dofs_per_cell; const unsigned int
    //          faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell; const
    //          unsigned int dofs_per_face  =
    //          d_dofHandler->get_fe().dofs_per_face;
    //
    //          std::vector<dealii::types::global_dof_index>
    //          cellGlobalDofIndices(dofs_per_cell);
    //          std::vector<dealii::types::global_dof_index>
    //          iFaceGlobalDofIndices(dofs_per_face);
    //
    //          std::vector<bool> dofs_touched(d_dofHandler->n_dofs(), false);
    //          dealii::DoFHandler<3>::active_cell_iterator cell =
    //          d_dofHandler->begin_active(),
    //                  endc = d_dofHandler->end();
    //          for (; cell != endc; ++cell)
    //              if (cell->is_locally_owned() || cell->is_ghost())
    //              {
    //                  cell->get_dof_indices(cellGlobalDofIndices);
    //                  for (unsigned int iFace = 0; iFace < faces_per_cell;
    //                  ++iFace)
    //                  {
    //                      const unsigned int boundaryId =
    //                      cell->face(iFace)->boundary_id(); if (boundaryId ==
    //                      0)
    //                      {
    //                          cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
    //                          for (unsigned int iFaceDof = 0; iFaceDof <
    //                          dofs_per_face;
    //                               ++iFaceDof)
    //                          {
    //                              const dealii::types::global_dof_index nodeId
    //                              =
    //                                      iFaceGlobalDofIndices[iFaceDof];
    //                              if (dofs_touched[nodeId])
    //                                  continue;
    //                              dofs_touched[nodeId] = true;
    //                              if
    //                              (!d_constraintMatrixSingleHangingPeriodicInhomogeneous.is_constrained(nodeId))
    //                              {
    //                                  d_constraintMatrixSingleHangingPeriodicInhomogeneous.add_line(nodeId);
    //                                  d_constraintMatrixSingleHangingPeriodicInhomogeneous.set_inhomogeneity(nodeId,
    //                                  iBlockId );
    //                              } // non-hanging node check
    //
    //                          }     // Face dof loop
    //                      }         // non-periodic boundary id
    //                  }             // Face loop
    //              }                 // cell locally owned
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.close();
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.distribute(singleBoundaryCond);
    //          for (unsigned int iNodeId = 0; iNodeId <
    //          singleBoundaryCond.local_size(); iNodeId ++)
    //          {
    //              boundaryValues.data()[iNodeId*blockSizeInput + iBlockId ] =
    //              singleBoundaryCond.local_element(iNodeId);
    //          }
    //          boundaryValues.updateGhostValues();
    //      }
    //
    //
    //      iElem = 0;
    //      cell = d_dofHandler->begin_active();
    //      for (; cell != endc; ++cell)
    //          if (cell->is_locally_owned() )
    //          {
    //              std::vector<double> &cellLevelQuadInput =
    //              inputVec.find(cell->id())->second;
    //              rhoQuadInputValues[iElem].resize(cellLevelQuadInput.size() *
    //              blockSizeInput); for (unsigned int i = 0; i <
    //              cellLevelQuadInput.size(); i++)
    //              {
    //                  for (unsigned int k = 0; k < blockSizeInput; k++)
    //                  {
    //                      rhoQuadInputValues[iElem][i * blockSizeInput + k] =
    //                              cellLevelQuadInput[i] * (k + 1);
    //                  }
    //              }
    //              iElem++;
    //
    //          }
    //
    //      for (unsigned int k = 0; k < blockSizeInput; k++)
    //      {
    //          expectedOutput = 0;
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.clear();
    //          dealii::DoFTools::make_hanging_node_constraints(*d_dofHandler,
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous);
    //
    //          const unsigned int vertices_per_cell =
    //                  dealii::GeometryInfo<3>::vertices_per_cell;
    //          const unsigned int dofs_per_cell  =
    //          d_dofHandler->get_fe().dofs_per_cell; const unsigned int
    //          faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell; const
    //          unsigned int dofs_per_face  =
    //          d_dofHandler->get_fe().dofs_per_face;
    //
    //          std::vector<dealii::types::global_dof_index>
    //          cellGlobalDofIndices(dofs_per_cell);
    //          std::vector<dealii::types::global_dof_index>
    //          iFaceGlobalDofIndices(dofs_per_face);
    //
    //          std::vector<bool> dofs_touched(d_dofHandler->n_dofs(), false);
    //          dealii::DoFHandler<3>::active_cell_iterator cell =
    //          d_dofHandler->begin_active(),
    //                  endc = d_dofHandler->end();
    //          for (; cell != endc; ++cell)
    //              if (cell->is_locally_owned() || cell->is_ghost())
    //              {
    //                  cell->get_dof_indices(cellGlobalDofIndices);
    //                  for (unsigned int iFace = 0; iFace < faces_per_cell;
    //                  ++iFace)
    //                  {
    //                      const unsigned int boundaryId =
    //                      cell->face(iFace)->boundary_id(); if (boundaryId ==
    //                      0)
    //                      {
    //                          cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
    //                          for (unsigned int iFaceDof = 0; iFaceDof <
    //                          dofs_per_face;
    //                               ++iFaceDof)
    //                          {
    //                              const dealii::types::global_dof_index nodeId
    //                              =
    //                                      iFaceGlobalDofIndices[iFaceDof];
    //                              if (dofs_touched[nodeId])
    //                                  continue;
    //                              dofs_touched[nodeId] = true;
    //                              if
    //                              (!d_constraintMatrixSingleHangingPeriodicInhomogeneous.is_constrained(nodeId))
    //                              {
    //                                  d_constraintMatrixSingleHangingPeriodicInhomogeneous.add_line(nodeId);
    //                                  d_constraintMatrixSingleHangingPeriodicInhomogeneous.set_inhomogeneity(nodeId,
    //                                  k );
    //                              } // non-hanging node check
    //
    //                          }     // Face dof loop
    //                      }         // non-periodic boundary id
    //                  }             // Face loop
    //              }                 // cell locally owned
    //          d_constraintMatrixSingleHangingPeriodicInhomogeneous.close();
    //          for (std::map<dealii::CellId, std::vector<double>>::iterator it
    //          = inputVec.begin(); it != inputVec.end(); it++)
    //          {
    //              std::vector<double> &cellLevelQuadInput = it->second;
    //
    //              for (unsigned int i = 0; i < cellLevelQuadInput.size(); i++)
    //              {
    //                  if ((k != 0) && (k != 1))
    //                      cellLevelQuadInput[i] = cellLevelQuadInput[i] * (k +
    //                      1) / (k);
    //                  else
    //                      cellLevelQuadInput[i] = cellLevelQuadInput[i] * (k +
    //                      1);
    //              }
    //          }
    //          phiTotalSolverProblem.reinit(matrixFreeData,
    //                                       expectedOutput,
    //                                       d_constraintMatrixSingleHangingPeriodicInhomogeneous,
    //                                       matrixFreeVectorComponent,
    //                                       matrixFreeQuadratureComponentRhsDensity,
    //                                       matrixFreeQuadratureComponentAX,
    //                                       atoms,
    //                                       smearedChargeValues,
    //                                       matrixFreeQuadratureComponentAX,
    //                                       inputVec,
    //                                       true,   // isComputeDiagonalA
    //                                       false,  //
    //                                       isComputeMeanValueConstraint false,
    //                                       // smearedNuclearCharges true,   //
    //                                       isRhoValues false,  //
    //                                       isGradSmearedChargeRhs 0,      //
    //                                       smearedChargeGradientComponentId
    //                                       false,  // storeSmearedChargeRhs
    //                                       false, // reuseSmearedChargeRhs
    //                                       true);
    //                                       ////reinitializeFastConstraints
    //
    //          dealiiCGSolver.solve(phiTotalSolverProblem, 1e-10, 1000, 4);
    //
    //          dealii::types::global_dof_index indexVec;
    //          for (dealii::types::global_dof_index i = 0; i <
    //          expectedOutput.size();
    //               i++)
    //          {
    //              indexVec = i * blockSizeInput + k;
    //              if (multiExpectedOutput.in_local_range(indexVec))
    //                  multiExpectedOutput(indexVec) = expectedOutput(i);
    //          }
    //      }
    //
    //
    //
    //
    //      std::cout << "Testing multi Vector Poisson Solve";
    //      std::cout << " Value of matrixFreeComponent = " <<
    //      matrixFreeVectorComponent
    //                << "\n";
    //      std::cout << " Value of QuadRhs             = "
    //                << matrixFreeQuadratureComponentRhsDensity << "\n";
    //      std::cout << " Value of QuadAX              = "
    //                << matrixFreeQuadratureComponentAX <<
    //      "\n";
    //      std::cout << " Size of expectedOutput       = " <<
    //      expectedOutput.size()
    //                << "\n";
    //
    //      //        if(this_process == 0) {
    ////      boundaryValues = 0.0;
    //      std::cout << "Size of input  = " << inputVec.size() << "\n";
    //      std::cout << "Size of output = " <<
    //      multiVectorOutput.locallyOwnedSize() << "\n"; std::cout << "Size of
    //      boundary = " << boundaryValues.locallyOwnedSize() << "\n"; std::cout
    //      << " Norm of boundary = " << boundaryValues.l2Norm() << "\n";
    //      //        }
    //
    //      linearSolver.solve(multiPoissonSolver,
    //                         rhoQuadInputValues,
    //                         multiVectorOutput,
    //                         boundaryValues,
    //                         blockSizeInput,
    //                         1e-10,
    //                         1000,
    //                         4,
    //                         true);
    //
    ////      shift.resize(blockSizeInput,0.0);
    ////      linearSolver.solve(multiPoissonSolver,
    ////                     rhoQuadInputValues,
    ////                     multiVectorOutput,
    ////                     boundaryValues,
    ////                     blockSizeInput,
    ////                     1e-10,
    ////                     1000,
    ////                     4,
    ////                     true);
    //
    //      std::cout << "Norm of expected input = " <<
    //      multiExpectedOutput.l2Norm()
    //                << "\n";
    //      std::cout<<"Norm of output from multiPoisson =
    //      "<<multiVectorOutput.l2Norm()<<"\n";
    //
    //      multiVectorOutput -= multiExpectedOutput;
    //      std::cout << "Vector L2 norm of the difference is : "
    //                << multiVectorOutput.l2Norm() << "\n";
  }
} // end of namespace unitTest