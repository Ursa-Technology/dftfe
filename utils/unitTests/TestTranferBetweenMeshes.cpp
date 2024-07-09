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

#include "TestTranferBetweenMeshes.h"
#include "MemoryTransfer.h"
namespace dftfe
{
  namespace {
    double
    value(double x, double y, double z, unsigned int index)
    {
      double val = 1;
      val        = x * x + y * y + z * z;
      return val;
    }
  }


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
    const bool                              generateElectrostaticsTria)
  {
    // create triangulation


    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);

    triangulationManager dftMesh(mpi_comm_parent,
                                 mpi_comm_domain,
                                 interpoolcomm,
                                 interbandgroup_comm,
                                 FEOrder,
                                 dftParams);

    dftParameters dftParamsVxc(dftParams);

    dftParamsVxc.innerAtomBallRadius = dftParams.VxcInnerDomain;
    dftParamsVxc.meshSizeInnerBall = dftParams.VxcInnerMeshSize;
    dftParamsVxc.meshSizeOuterDomain = dftParams.meshSizeOuterDomain;
    dftParamsVxc.outerAtomBallRadius = dftParams.outerAtomBallRadius;
    dftParamsVxc.meshSizeOuterBall = dftParams.meshSizeOuterBall;

    triangulationManager VxcMesh(mpi_comm_parent,
                                 mpi_comm_domain,
                                 interpoolcomm,
                                 interbandgroup_comm,
                                 FEOrder,
                                 dftParamsVxc);
    //
    dftMesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(
      atomLocations,
      imageAtomLocations,
      imageIds,
      nearestAtomDistances,
      domainBoundingVectors,
      false,  // generateSerialTria
      false); // generateElectrostaticsTria

    const parallel::distributed::Triangulation<3> &parallelMeshUnmoved =
      dftMesh.getParallelMeshUnmoved();
    const parallel::distributed::Triangulation<3> &parallelMeshMoved =
      dftMesh.getParallelMeshMoved();

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);


    //      triangulationManagerVxc VxcMesh(mpi_comm_parent,
    //                                      mpi_comm_domain,
    //                                      interpoolcomm,
    //                                      interbandgroup_comm,
    //                                      dftParams);
    //
    //      VxcMesh.generateParallelUnmovedMeshVxc(
    //              dftParams.meshSizeOuterBall / 2,
    //      parallelMeshUnmoved,
    //      atomLocations, // This is compatible with only non-periodic boundary
    //                     // conditions as imageAtomLocations is not considered
    //      dftMesh);


    //      VxcMesh.generateParallelMovedMeshVxc(parallelMeshUnmoved,
    //                                                parallelMeshMoved);
    //
    //    const parallel::distributed::Triangulation<3> &parallelMeshMovedVxc =
    //            VxcMesh.getParallelMovedMeshVxc();
    //
    //    const parallel::distributed::Triangulation<3> &parallelMeshUnmovedVxc =
    //            VxcMesh.getParallelUnmovedMeshVxc();


    // create Vxc mesh

    VxcMesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(
      atomLocations,
      imageAtomLocations,
      imageIds,
      nearestAtomDistances,
      domainBoundingVectors,
      false,  // generateSerialTria
      false); // generateElectrostaticsTria

    const parallel::distributed::Triangulation<3> &parallelMeshMovedVxc =
      VxcMesh.getParallelMeshMoved();

    const parallel::distributed::Triangulation<3> &parallelMeshUnmovedVxc =
      VxcMesh.getParallelMeshUnmoved();

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);
    // construct dofHandler and constraints matrix

    dealii::DoFHandler<3> dofHandlerTria(parallelMeshMoved),
      dofHandlerTriaVxc(parallelMeshMovedVxc);
    const dealii::FE_Q<3> finite_elementHigh(FEOrder + 2);
    const dealii::FE_Q<3> finite_elementLow(FEOrder);

    dofHandlerTria.distribute_dofs(finite_elementHigh);

    dofHandlerTriaVxc.distribute_dofs(finite_elementLow);

    dealii::AffineConstraints<double> constraintMatrix, constraintMatrixVxc;

    dealii::IndexSet locallyRelevantDofs, locallyRelevantDofsVxc;

    dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerTria,
                                                    locallyRelevantDofs);

    dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerTriaVxc,
                                                    locallyRelevantDofsVxc);


    constraintMatrix.clear();
    constraintMatrix.reinit(locallyRelevantDofs);
    dealii::DoFTools::make_hanging_node_constraints(dofHandlerTria,
                                                    constraintMatrix);
    // uncomment this for homogenous BC
    // The test function should also be compatoble
    //    dealii::VectorTools::interpolate_boundary_values(dofHandlerTria,
    //                                                     0,
    //                                                     dealii::Functions::ZeroFunction<3>(),
    //                                                     constraintMatrix);
    constraintMatrix.close();

    constraintMatrixVxc.clear();
    constraintMatrix.reinit(locallyRelevantDofsVxc);
    dealii::DoFTools::make_hanging_node_constraints(dofHandlerTriaVxc,
                                                    constraintMatrixVxc);

    // uncomment this for homogenous BC
    // The test function should also be compatoble
    //    dealii::VectorTools::interpolate_boundary_values(dofHandlerTriaVxc,
    //                                                     0,
    //                                                     dealii::Functions::ZeroFunction<3>(),
    //                                                     constraintMatrixVxc);

    constraintMatrixVxc.close();

    // create quadrature

    dealii::QGauss<3>             gaussQuadHigh(FEOrder + 3);
    dealii::QGauss<3>             gaussQuadLow(FEOrder + 1);
    unsigned int                  numQuadPointsHigh = gaussQuadHigh.size();
    unsigned int                  numQuadPointsLow  = gaussQuadLow.size();
    dealii::MatrixFree<3, double> matrixFreeData, matrixFreeDataVxc;


    matrixFreeData.reinit(dealii::MappingQ1<3, 3>(),
                          dofHandlerTria,
                          constraintMatrix,
                          gaussQuadHigh);

    matrixFreeDataVxc.reinit(dealii::MappingQ1<3, 3>(),
                             dofHandlerTriaVxc,
                             constraintMatrixVxc,
                             gaussQuadLow);

    distributedCPUMultiVec<double> parentVec, childVec;

    unsigned int blockSize = 1;


    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      matrixFreeData.get_vector_partitioner(0), blockSize, parentVec);

    dftUtils::constraintMatrixInfo multiVectorConstraintsParent;
    multiVectorConstraintsParent.initialize(
      matrixFreeData.get_vector_partitioner(0), constraintMatrix);

    multiVectorConstraintsParent.precomputeMaps(parentVec.getMPIPatternP2P(),
                                                blockSize);

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      quadValuesChildAnalytical, quadValuesChildComputed;

    unsigned int totalLocallyOwnedCellsVxc =
      matrixFreeDataVxc.n_physical_cells();
    quadValuesChildAnalytical.resize(totalLocallyOwnedCellsVxc *
                                     numQuadPointsLow * blockSize);
    quadValuesChildComputed.resize(totalLocallyOwnedCellsVxc *
                                   numQuadPointsLow * blockSize);


    std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coord;
    dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      dealii::MappingQ1<3, 3>(), dofHandlerTria, dof_coord);


    dealii::types::global_dof_index numberDofsParent = dofHandlerTria.n_dofs();

    std::shared_ptr<
      const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
      parentMPIPattern = parentVec.getMPIPatternP2P();
    const std::pair<global_size_type, global_size_type>
      &locallyOwnedRangeParent = parentMPIPattern->getLocallyOwnedRange();

    for (dealii::types::global_dof_index iNode = locallyOwnedRangeParent.first;
         iNode < locallyOwnedRangeParent.second;
         iNode++)
      {
        for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
          {
            unsigned int indexVec =
              (iNode - locallyOwnedRangeParent.first) * blockSize + iBlock;
            if ((!constraintMatrix.is_constrained(iNode)))
              *(parentVec.data() + indexVec) = value(dof_coord[iNode][0],
                                                     dof_coord[iNode][1],
                                                     dof_coord[iNode][2],
                                                     iBlock);
          }
      }

    multiVectorConstraintsParent.distribute(parentVec, blockSize);

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);


    //    parentVec.update_ghost_values();



    TransferDataBetweenMeshesIncompatiblePartitioning inverseDftDoFManagerObj(matrixFreeData,
                                                                              0,
                                                                              0,
                                                                              matrixFreeDataVxc,
                                                                              0,
                                                                              0,
                                                                              mpi_comm_domain,
                                                                              false);

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);

    std::vector<dealii::types::global_dof_index>
      fullFlattenedArrayCellLocalProcIndexIdMapParent;
    vectorTools::computeCellLocalIndexSetMap(
      parentVec.getMPIPatternP2P(),
      matrixFreeData,
      0,
      blockSize,
      fullFlattenedArrayCellLocalProcIndexIdMapParent);

    global_size_type numPointsChild = (global_size_type)(totalLocallyOwnedCellsVxc *
                                                         numQuadPointsLow * blockSize);

    MPI_Allreduce(MPI_IN_PLACE,
                  &numPointsChild,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numPointsChild),
                  MPI_SUM,
                  mpi_comm_domain);

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);


    double startTimeMesh1ToMesh2 = MPI_Wtime();
    inverseDftDoFManagerObj.interpolateMesh1DataToMesh2QuadPoints(
      parentVec,
      blockSize,
      fullFlattenedArrayCellLocalProcIndexIdMapParent,
      quadValuesChildComputed,
      true);


    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);
    double endTimeMesh1ToMesh2 = MPI_Wtime();

    if(  dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0 ) {

        std::cout<<" Num of points  = "<<numPointsChild<<"\n";
        std::cout << " Time taken to transfer from Mesh 1 to Mesh 2 = " << endTimeMesh1ToMesh2 - startTimeMesh1ToMesh2
                  << "\n";

      }

    dealii::FEValues<3> fe_valuesChild(dofHandlerTriaVxc.get_fe(),
                                       gaussQuadLow,
                                       dealii::update_values |
                                         dealii::update_quadrature_points);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellChild = dofHandlerTriaVxc.begin_active(),
      endcChild = dofHandlerTriaVxc.end();

    unsigned int iCellChildIndex = 0;
    for (; cellChild != endcChild; cellChild++)
      {
        if (cellChild->is_locally_owned())
          {
            fe_valuesChild.reinit(cellChild);
            for (unsigned int iQuad = 0; iQuad < numQuadPointsLow; iQuad++)
              {
                dealii::Point<3, double> qPointVal =
                  fe_valuesChild.quadrature_point(iQuad);
                for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
                  {
                    quadValuesChildAnalytical
                      [(iCellChildIndex * numQuadPointsLow + iQuad) *
                         blockSize +
                       iBlock] =
                        value(qPointVal[0], qPointVal[1], qPointVal[2], iBlock);
                  }
              }
            iCellChildIndex++;
          }
      }

    double l2Error = 0.0;
    for (unsigned int iQuad = 0; iQuad < quadValuesChildAnalytical.size();
         iQuad++)
      {
        l2Error +=
          ((quadValuesChildComputed[iQuad] - quadValuesChildAnalytical[iQuad]) *
           (quadValuesChildComputed[iQuad] - quadValuesChildAnalytical[iQuad]));
      }
    MPI_Allreduce(
      MPI_IN_PLACE, &l2Error, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_domain);
    l2Error = std::sqrt(l2Error);
    if(  dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0 ) {
        std::cout << " Error while interpolating to quad points of child = "
                  << l2Error << "\n";
      }



    // test transfer child to parent

    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      matrixFreeDataVxc.get_vector_partitioner(0), blockSize, childVec);

    dftUtils::constraintMatrixInfo multiVectorConstraintsChild;
    multiVectorConstraintsChild.initialize(
      matrixFreeDataVxc.get_vector_partitioner(0), constraintMatrixVxc);

    multiVectorConstraintsChild.precomputeMaps(childVec.getMPIPatternP2P(),
                                               blockSize);

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      quadValuesParentAnalytical, quadValuesParentComputed;

    unsigned int totalLocallyOwnedCellsParent =
      matrixFreeData.n_physical_cells();
    quadValuesParentAnalytical.resize(totalLocallyOwnedCellsParent *
                                      numQuadPointsHigh * blockSize);
    quadValuesParentComputed.resize(totalLocallyOwnedCellsParent *
                                    numQuadPointsHigh * blockSize);


    std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coord_child;
    dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      dealii::MappingQ1<3, 3>(), dofHandlerTriaVxc, dof_coord_child);


    dealii::types::global_dof_index numberDofsChild =
      dofHandlerTriaVxc.n_dofs();

    std::shared_ptr<
      const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
      childMPIPattern = childVec.getMPIPatternP2P();
    const std::pair<global_size_type, global_size_type>
      &locallyOwnedRangeChild = childMPIPattern->getLocallyOwnedRange();

    for (dealii::types::global_dof_index iNode = locallyOwnedRangeChild.first;
         iNode < locallyOwnedRangeChild.second;
         iNode++)
      {
        for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
          {
            unsigned int indexVec =
              (iNode - locallyOwnedRangeChild.first) * blockSize + iBlock;
            if ((!constraintMatrixVxc.is_constrained(iNode)))
              *(childVec.data() + indexVec) = value(dof_coord_child[iNode][0],
                                                    dof_coord_child[iNode][1],
                                                    dof_coord_child[iNode][2],
                                                    iBlock);
            //            if ((childVec.in_local_range(indexVec)))
            //              childVec(indexVec) =
            //              value(dof_coord_child[iNode][0],dof_coord_child[iNode][1],dof_coord_child[iNode][2],iBlock);
          }
      }

    multiVectorConstraintsChild.distribute(childVec, blockSize);
    //    childVec.updateGhostValues();

    global_size_type numPointsParent = totalLocallyOwnedCellsParent *
                                       numQuadPointsHigh * blockSize;

    MPI_Allreduce(MPI_IN_PLACE,
                  &numPointsParent,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numPointsParent),
                  MPI_SUM,
                  mpi_comm_domain);
    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);

    double startTimeMesh2ToMesh1 = MPI_Wtime();
    inverseDftDoFManagerObj.interpolateMesh2DataToMesh1QuadPoints(
      childVec, blockSize, quadValuesParentComputed, true);

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);
    double endTimeMesh2ToMesh1 = MPI_Wtime();

    if(  dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0 ) {
        std::cout<<" Number of points parent = "<<numPointsParent<<"\n";
        std::cout << " Time taken to transfer from Mesh 2 to Mesh 1 = " << endTimeMesh2ToMesh1 - startTimeMesh2ToMesh1
                  << "\n";
      }

    dealii::FEValues<3> fe_valuesParent(dofHandlerTria.get_fe(),
                                        gaussQuadHigh,
                                        dealii::update_values |
                                          dealii::update_quadrature_points);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellParent = dofHandlerTria.begin_active(),
      endcParent = dofHandlerTria.end();

    unsigned int iCellParentIndex = 0;
    for (; cellParent != endcParent; cellParent++)
      {
        if (cellParent->is_locally_owned())
          {
            fe_valuesParent.reinit(cellParent);
            for (unsigned int iQuad = 0; iQuad < numQuadPointsHigh; iQuad++)
              {
                dealii::Point<3, double> qPointVal =
                  fe_valuesParent.quadrature_point(iQuad);
                for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
                  {
                    quadValuesParentAnalytical
                      [(iCellParentIndex * numQuadPointsHigh + iQuad) *
                         blockSize +
                       iBlock] =
                        value(qPointVal[0], qPointVal[1], qPointVal[2], iBlock);
                  }
              }
            iCellParentIndex++;
          }
      }

    l2Error = 0.0;
    for (unsigned int iQuad = 0; iQuad < quadValuesParentAnalytical.size();
         iQuad++)
      {
        double diff = ((quadValuesParentComputed[iQuad] -
                        quadValuesParentAnalytical[iQuad]) *
                       (quadValuesParentComputed[iQuad] -
                        quadValuesParentAnalytical[iQuad]));

        l2Error += diff;

        //        if ( diff > 1e-5)
        //          {
        //            std::cout<<"iQuad = " <<iQuad<<" anal =
        //            "<<quadValuesParentAnalytical[iQuad]<<" comp =
        //            "<<quadValuesParentComputed[iQuad]<<"\n";
        //          }
      }
    MPI_Allreduce(
      MPI_IN_PLACE, &l2Error, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_domain);
    l2Error = std::sqrt(l2Error);

    if(  dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0 ) {
        std::cout << " Error while interpolating to quad points of parent = "
                  << l2Error << "\n";
      }


    //    exit(0);
  }

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
    const bool                              generateElectrostaticsTria)
  {
    // create triangulation


    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);

    triangulationManager dftMesh(mpi_comm_parent,
                                 mpi_comm_domain,
                                 interpoolcomm,
                                 interbandgroup_comm,
                                 FEOrder,
                                 dftParams);

    dftParameters dftParamsVxc(dftParams);

    dftParamsVxc.innerAtomBallRadius = dftParams.VxcInnerDomain;
    dftParamsVxc.meshSizeInnerBall = dftParams.VxcInnerMeshSize;
    dftParamsVxc.meshSizeOuterDomain = dftParams.meshSizeOuterDomain;
    dftParamsVxc.outerAtomBallRadius = dftParams.outerAtomBallRadius;
    dftParamsVxc.meshSizeOuterBall = dftParams.meshSizeOuterBall;

    triangulationManager VxcMesh(mpi_comm_parent,
                                 mpi_comm_domain,
                                 interpoolcomm,
                                 interbandgroup_comm,
                                 FEOrder,
                                 dftParamsVxc);
    //
    dftMesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(
      atomLocations,
      imageAtomLocations,
      imageIds,
      nearestAtomDistances,
      domainBoundingVectors,
      false,  // generateSerialTria
      false); // generateElectrostaticsTria

    const parallel::distributed::Triangulation<3> &parallelMeshUnmoved =
      dftMesh.getParallelMeshUnmoved();
    const parallel::distributed::Triangulation<3> &parallelMeshMoved =
      dftMesh.getParallelMeshMoved();

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);


    // create Vxc mesh

    VxcMesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(
      atomLocations,
      imageAtomLocations,
      imageIds,
      nearestAtomDistances,
      domainBoundingVectors,
      false,  // generateSerialTria
      false); // generateElectrostaticsTria

    const parallel::distributed::Triangulation<3> &parallelMeshMovedVxc =
      VxcMesh.getParallelMeshMoved();

    const parallel::distributed::Triangulation<3> &parallelMeshUnmovedVxc =
      VxcMesh.getParallelMeshUnmoved();

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);
    // construct dofHandler and constraints matrix

    dealii::DoFHandler<3> dofHandlerTria(parallelMeshMoved),
      dofHandlerTriaVxc(parallelMeshMovedVxc);
    const dealii::FE_Q<3> finite_elementHigh(FEOrder + 2);
    const dealii::FE_Q<3> finite_elementLow(FEOrder);

    dofHandlerTria.distribute_dofs(finite_elementHigh);

    dofHandlerTriaVxc.distribute_dofs(finite_elementLow);

    dealii::AffineConstraints<double> constraintMatrix, constraintMatrixVxc;

    dealii::IndexSet locallyRelevantDofs, locallyRelevantDofsVxc;

    dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerTria,
                                                    locallyRelevantDofs);

    dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerTriaVxc,
                                                    locallyRelevantDofsVxc);


    constraintMatrix.clear();
    constraintMatrix.reinit(locallyRelevantDofs);
    dealii::DoFTools::make_hanging_node_constraints(dofHandlerTria,
                                                    constraintMatrix);
    // uncomment this for homogenous BC
    // The test function should also be compatoble
    //    dealii::VectorTools::interpolate_boundary_values(dofHandlerTria,
    //                                                     0,
    //                                                     dealii::Functions::ZeroFunction<3>(),
    //                                                     constraintMatrix);
    constraintMatrix.close();

    constraintMatrixVxc.clear();
    constraintMatrix.reinit(locallyRelevantDofsVxc);
    dealii::DoFTools::make_hanging_node_constraints(dofHandlerTriaVxc,
                                                    constraintMatrixVxc);

    // uncomment this for homogenous BC
    // The test function should also be compatoble
    //    dealii::VectorTools::interpolate_boundary_values(dofHandlerTriaVxc,
    //                                                     0,
    //                                                     dealii::Functions::ZeroFunction<3>(),
    //                                                     constraintMatrixVxc);

    constraintMatrixVxc.close();

    // create quadrature

    dealii::QGauss<3>             gaussQuadHigh(FEOrder + 3);
    dealii::QGauss<3>             gaussQuadLow(FEOrder + 1);
    unsigned int                  numQuadPointsHigh = gaussQuadHigh.size();
    unsigned int                  numQuadPointsLow  = gaussQuadLow.size();
    dealii::MatrixFree<3, double> matrixFreeData, matrixFreeDataVxc;


    matrixFreeData.reinit(dealii::MappingQ1<3, 3>(),
                          dofHandlerTria,
                          constraintMatrix,
                          gaussQuadHigh);

    matrixFreeDataVxc.reinit(dealii::MappingQ1<3, 3>(),
                             dofHandlerTriaVxc,
                             constraintMatrixVxc,
                             gaussQuadLow);

    distributedCPUMultiVec<double> parentVec, childVec;

    distributedDeviceVec<double> parentVecDevice, childVecDevice;

    unsigned int blockSize = 4;


    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      matrixFreeData.get_vector_partitioner(0), blockSize, parentVec);

    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      matrixFreeData.get_vector_partitioner(0), blockSize, parentVecDevice);

    dftUtils::constraintMatrixInfo multiVectorConstraintsParent;
    multiVectorConstraintsParent.initialize(
      matrixFreeData.get_vector_partitioner(0), constraintMatrix);

    multiVectorConstraintsParent.precomputeMaps(parentVec.getMPIPatternP2P(),
                                                blockSize);

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      quadValuesChildAnalytical, quadValuesChildComputed;

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      quadValuesChildAnalyticalDevice, quadValuesChildComputedDevice;

    unsigned int totalLocallyOwnedCellsVxc =
      matrixFreeDataVxc.n_physical_cells();
    quadValuesChildAnalytical.resize(totalLocallyOwnedCellsVxc *
                                     numQuadPointsLow * blockSize);
    quadValuesChildComputed.resize(totalLocallyOwnedCellsVxc *
                                   numQuadPointsLow * blockSize);


    quadValuesChildAnalyticalDevice.resize(totalLocallyOwnedCellsVxc *
                                           numQuadPointsLow * blockSize);
    quadValuesChildComputedDevice.resize(totalLocallyOwnedCellsVxc *
                                         numQuadPointsLow * blockSize);


    std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coord;
    dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      dealii::MappingQ1<3, 3>(), dofHandlerTria, dof_coord);


    dealii::types::global_dof_index numberDofsParent = dofHandlerTria.n_dofs();

    std::shared_ptr<
      const utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
      parentMPIPattern = parentVec.getMPIPatternP2P();
    const std::pair<global_size_type, global_size_type>
      &locallyOwnedRangeParent = parentMPIPattern->getLocallyOwnedRange();

    for (dealii::types::global_dof_index iNode = locallyOwnedRangeParent.first;
         iNode < locallyOwnedRangeParent.second;
         iNode++)
      {
        for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
          {
            unsigned int indexVec =
              (iNode - locallyOwnedRangeParent.first) * blockSize + iBlock;
            if ((!constraintMatrix.is_constrained(iNode)))
              *(parentVec.data() + indexVec) = value(dof_coord[iNode][0],
                                                     dof_coord[iNode][1],
                                                     dof_coord[iNode][2],
                                                     iBlock);
          }
      }

    multiVectorConstraintsParent.distribute(parentVec, blockSize);

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);


    //    parentVec.update_ghost_values();


    dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE, dftfe::utils::MemorySpace::HOST>
      memoryTransferHostToDevice;

    dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::HOST, dftfe::utils::MemorySpace::DEVICE>
      memoryTransferDeviceToHost;

    memoryTransferHostToDevice.copy((parentVec.ghostSize() + parentVec.locallyOwnedSize())*parentVec.numVectors(),
                                    parentVecDevice.begin(),
                                    parentVec.begin());

    TransferDataBetweenMeshesIncompatiblePartitioning inverseDftDoFManagerObj(matrixFreeData,
                                                                              0,
                                                                              0,
                                                                              matrixFreeDataVxc,
                                                                              0,
                                                                              0,
                                                                              mpi_comm_domain,
                                                                              true); // useDevice

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);

    std::vector<dealii::types::global_dof_index>
      fullFlattenedArrayCellLocalProcIndexIdMapParent;
    vectorTools::computeCellLocalIndexSetMap(
      parentVec.getMPIPatternP2P(),
      matrixFreeData,
      0,
      blockSize,
      fullFlattenedArrayCellLocalProcIndexIdMapParent);

    dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                dftfe::utils::MemorySpace::DEVICE> fullFlattenedArrayCellLocalProcIndexIdMapParentDevice;

    fullFlattenedArrayCellLocalProcIndexIdMapParentDevice.resize(fullFlattenedArrayCellLocalProcIndexIdMapParent.size());

    fullFlattenedArrayCellLocalProcIndexIdMapParentDevice.copyFrom(fullFlattenedArrayCellLocalProcIndexIdMapParent);

    global_size_type numPointsChild = (global_size_type)(totalLocallyOwnedCellsVxc *
                                                         numQuadPointsLow * blockSize);

    MPI_Allreduce(MPI_IN_PLACE,
                  &numPointsChild,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numPointsChild),
                  MPI_SUM,
                  mpi_comm_domain);

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);


    double startTimeMesh1ToMesh2 = MPI_Wtime();
    inverseDftDoFManagerObj.interpolateMesh1DataToMesh2QuadPoints(
      parentVecDevice,
      blockSize,
      fullFlattenedArrayCellLocalProcIndexIdMapParentDevice,
      quadValuesChildComputedDevice,
      true);

    std::cout<<std::flush;
    MPI_Barrier(mpi_comm_domain);
    double endTimeMesh1ToMesh2 = MPI_Wtime();

    memoryTransferDeviceToHost.copy(totalLocallyOwnedCellsVxc *
                                      numQuadPointsLow * blockSize,
                                    quadValuesChildComputed.begin(),
                                    quadValuesChildComputedDevice.begin());

    if(  dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0 ) {

        std::cout<<" Num of points  = "<<numPointsChild<<"\n";
        std::cout << " Time taken to transfer from Mesh 1 to Mesh 2 = " << endTimeMesh1ToMesh2 - startTimeMesh1ToMesh2
                  << "\n";

      }

    dealii::FEValues<3> fe_valuesChild(dofHandlerTriaVxc.get_fe(),
                                       gaussQuadLow,
                                       dealii::update_values |
                                         dealii::update_quadrature_points);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellChild = dofHandlerTriaVxc.begin_active(),
      endcChild = dofHandlerTriaVxc.end();

    unsigned int iCellChildIndex = 0;
    for (; cellChild != endcChild; cellChild++)
      {
        if (cellChild->is_locally_owned())
          {
            fe_valuesChild.reinit(cellChild);
            for (unsigned int iQuad = 0; iQuad < numQuadPointsLow; iQuad++)
              {
                dealii::Point<3, double> qPointVal =
                  fe_valuesChild.quadrature_point(iQuad);
                for (unsigned int iBlock = 0; iBlock < blockSize; iBlock++)
                  {
                    quadValuesChildAnalytical
                      [(iCellChildIndex * numQuadPointsLow + iQuad) *
                         blockSize +
                       iBlock] =
                        value(qPointVal[0], qPointVal[1], qPointVal[2], iBlock);
                  }
              }
            iCellChildIndex++;
          }
      }

    double l2Error = 0.0;
    for (unsigned int iQuad = 0; iQuad < quadValuesChildAnalytical.size();
         iQuad++)
      {
        l2Error +=
          ((quadValuesChildComputed[iQuad] - quadValuesChildAnalytical[iQuad]) *
           (quadValuesChildComputed[iQuad] - quadValuesChildAnalytical[iQuad]));
      }
    MPI_Allreduce(
      MPI_IN_PLACE, &l2Error, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_domain);
    l2Error = std::sqrt(l2Error);
    if(  dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain) == 0 ) {
        std::cout << " Error while interpolating to quad points of child = "
                  << l2Error << "\n";
      }

    exit(0);
  }

#endif
} // namespace dftfe
