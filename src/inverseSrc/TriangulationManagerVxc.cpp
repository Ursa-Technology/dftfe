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
/** @file triangulationManagerVxc.cc
 *
 *  @brief Source file for triangulationManager.h
 *
 *  @author Bikash Kanungo, Vishal Subramanian
 */


#include <constants.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <meshGenUtils.h>
#include <TriangulationManagerVxc.h>

//#include "generateMesh.cc"
//#include "restartUtils.cc"

namespace dftfe
{
  namespace
  {
    void
    getSystemExtent(const std::vector<std::vector<double>> &atomLocations,
                    const unsigned int                      coordIndexOffset,
                    const double                            innerDomainSize,
                    std::vector<double> &                   lo,
                    std::vector<double> &                   hi)
    {
      const unsigned int N = atomLocations.size();
      lo.resize(3, 0.0);
      hi.resize(3, 0.0);
      for (unsigned int i = 0; i < 3; ++i)
        {
          lo[i] = atomLocations[0][i + coordIndexOffset];
          hi[i] = atomLocations[0][i + coordIndexOffset];
        }

      for (unsigned int i = 0; i < N; ++i)
        {
          for (unsigned int j = 0; j < 3; ++j)
            {
              if (atomLocations[i][j + coordIndexOffset] - innerDomainSize <
                  lo[j])
                lo[j] =
                  atomLocations[i][j + coordIndexOffset] - innerDomainSize;

              if (atomLocations[i][j + coordIndexOffset] + innerDomainSize >
                  hi[j])
                hi[j] =
                  atomLocations[i][j + coordIndexOffset] + innerDomainSize;
            }
        }
    }

    double
    getMinCellSize(const parallel::distributed::Triangulation<3> &parallelMesh,
                   const MPI_Comm &mpi_comm_domain)
    {
      unsigned int iCell      = 0;
      double       minCellDia = 0.0;
      for (auto &cellIter : parallelMesh.active_cell_iterators())
        {
          if (iCell == 0)
            {
              minCellDia = cellIter->diameter();
            }
          if (minCellDia > cellIter->diameter())
            minCellDia = cellIter->diameter();

          iCell++;
        }
      MPI_Allreduce(
        MPI_IN_PLACE, &minCellDia, 1, MPI_DOUBLE, MPI_MIN, mpi_comm_domain);

      return minCellDia;
    }

  } // namespace

  //
  //
  // constructor
  //
  TriangulationManagerVxc::TriangulationManagerVxc(
    const MPI_Comm &     mpi_comm_parent,
    const MPI_Comm &     mpi_comm_domain,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interbandgroup_comm,
    const dftParameters &dftParams,
    const inverseDFTParameters & inverseDFTParams,
    const dealii::parallel::distributed::Triangulation<3, 3>::Settings repartitionFlag)
    : d_mpi_comm_parent(mpi_comm_parent)
    , d_mpi_comm_domain(mpi_comm_domain)
    , d_interpoolcomm(interpoolcomm)
    , d_interbandgroup_comm(interbandgroup_comm)
    , d_parallelTriangulationUnmovedVxc(
        mpi_comm_domain,
        dealii::Triangulation<3, 3>::none,
        repartitionFlag)
    , d_parallelTriangulationMovedVxc(
        mpi_comm_domain,
        dealii::Triangulation<3, 3>::none,
        repartitionFlag)
    , d_serialTriangulationVxc(MPI_COMM_SELF)
    , d_serialTriangulationElectrostaticsVxc(MPI_COMM_SELF)
    , d_electrostaticsTriangulationRhoVxc(mpi_comm_domain)
    , d_electrostaticsTriangulationDispVxc(mpi_comm_domain)
    , d_electrostaticsTriangulationForceVxc(mpi_comm_domain)
    , d_dftParams(dftParams)
    , d_inverseDFTParams(inverseDFTParams)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , computing_timer(pcout, TimerOutput::never, TimerOutput::wall_times)
  {}

  //
  // destructor
  //
  TriangulationManagerVxc::~TriangulationManagerVxc()
  {}


  void
  TriangulationManagerVxc::generateParallelUnmovedMeshVxc(
    const std::vector<std::vector<double>> &       atomPositions,
    triangulationManager &                         dftTria)
  {
    std::vector<double> minCoord, maxCoord;
    // TODO var innerDomainSize is hard coded, read it from dftParams
    double innerDomainSize = d_inverseDFTParams.VxcInnerDomain;
    getSystemExtent(atomPositions, 2, innerDomainSize, minCoord, maxCoord);
    d_parallelTriangulationUnmovedVxc.clear();
    // copy the mesh
    //    generateMeshWithManualRepartitioning(d_parallelTriangulationUnmovedVxc);


    dftTria.generateMesh(d_parallelTriangulationUnmovedVxc,
                         d_serialTriangulationVxc,
                         d_parallelTriaCurrentRefinement,
                         d_serialTriaCurrentRefinement,
                         false,
                         true);
    /*
        AssertThrow(
          parallelMeshUnmoved.n_active_cells() ==
            d_parallelTriangulationUnmovedVxc.n_active_cells(),
          ExcMessage(
            "DFT-FE error:  Vxc mesh partitioning is not consistent with the wave function mesh "));

        AssertThrow(
          parallelMeshUnmoved.n_active_cells() ==
            d_parallelTriangulationUnmovedVxc.n_active_cells(),
          ExcMessage(
            "DFT-FE error:  Vxc mesh partitioning is not consistent with the wave function mesh "));
    */
    double minCellSize = getMinCellSize(d_parallelTriangulationUnmovedVxc, d_mpi_comm_domain);
    bool   meshSatisfied = false;
    // TODO change the flag for refinement to ensure no change in parallel
    // layout

    unsigned int refineStepIter = 0;
    while (!meshSatisfied)
      {
        pcout << " refinement step = " << refineStepIter << "\n";
        meshSatisfied = true;
        for (auto &cellIter :
             d_parallelTriangulationUnmovedVxc.active_cell_iterators())
          {
            if (cellIter->is_locally_owned())
              {
                dealii::Point<3, double> center = cellIter->center();


                bool refineThisCell = true;
                for (unsigned int iDim = 0; iDim < 3; iDim++)
                  {
                    if ((center[iDim] > maxCoord[iDim]) ||
                        (center[iDim] < minCoord[iDim]))
                      {
                        refineThisCell = false;
                        break;
                      }
                  }
                if (refineThisCell && (cellIter->minimum_vertex_distance() >
                                       (minCellSize + 1e-4)))
                  {
                    cellIter->set_refine_flag();
                    meshSatisfied = false;
                  }
              }
          }
        // DO not call repartition(). this ensures that the partitioning is
        // consistent
        d_parallelTriangulationUnmovedVxc.execute_coarsening_and_refinement();
        d_parallelTriangulationUnmovedVxc.repartition();
        meshSatisfied =
          Utilities::MPI::min((unsigned int)meshSatisfied, d_mpi_comm_domain);

        refineStepIter++;
      };

    if (d_dftParams.verbosity >= 4)
      pcout << std::endl
            << "Final triangulation number of elements: "
            << d_parallelTriangulationUnmovedVxc.n_global_active_cells()
            << std::endl;

    double minElemLength = d_dftParams.meshSizeOuterDomain;
    double maxElemLength = 0.0;
    typename parallel::distributed::Triangulation<3>::active_cell_iterator cell,
      endc;
    cell = d_parallelTriangulationUnmovedVxc.begin_active();
    endc = d_parallelTriangulationUnmovedVxc.end();
    unsigned int numLocallyOwnedCells = 0;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            numLocallyOwnedCells++;
            if (cell->minimum_vertex_distance() < minElemLength)
              minElemLength = cell->minimum_vertex_distance();

            if (cell->minimum_vertex_distance() > maxElemLength)
              maxElemLength = cell->minimum_vertex_distance();
          }
      }

    minElemLength = Utilities::MPI::min(minElemLength, d_mpi_comm_domain);
    maxElemLength = Utilities::MPI::max(maxElemLength, d_mpi_comm_domain);

    //
    // print out adaptive mesh metrics and check mesh generation synchronization
    // across pools
    //
    if (d_dftParams.verbosity >= 4)
      {
        pcout << "Vxc Triangulation generation summary: " << std::endl
              << " num elements: "
              << d_parallelTriangulationUnmovedVxc.n_global_active_cells()
              << ", min element length: " << minElemLength
              << ", max element length: " << maxElemLength << std::endl;
      }
  }

  void
  TriangulationManagerVxc::generateParallelMovedMeshVxc(
    const parallel::distributed::Triangulation<3> &parallelMeshUnmoved,
    const parallel::distributed::Triangulation<3> &parallelMeshMoved)
  {
    //        d_parallelTriangulationUnmovedVxc.clear();
    //        d_parallelTriangulationUnmovedVxc.copy_triangulation(parallelMeshUnmoved);
    //
    //
    //        d_parallelTriangulationMovedVxc.clear();
    //        d_parallelTriangulationMovedVxc.copy_triangulation(parallelMeshMoved);
    //
    //        if (d_dftParams.verbosity >= 4)
    //          pcout << std::endl
    //                << "Final triangulation number of elements: "
    //                << d_parallelTriangulationMovedVxc.n_global_active_cells()
    //                << std::endl;



    d_parallelTriangulationMovedVxc.copy_triangulation(
      d_parallelTriangulationUnmovedVxc);

    const std::vector<bool> locally_owned_vertices =
      dealii::GridTools::get_locally_owned_vertices(
        d_parallelTriangulationMovedVxc);

    std::vector<bool> vertex_moved(d_parallelTriangulationMovedVxc.n_vertices(),
                                   false);
    std::vector<bool> gridPointTouched(
      d_parallelTriangulationMovedVxc.n_vertices(), false);
    // using a linear mapping
    dealii::MappingQGeneric<3, 3> mapping(1);

    std::vector<std::vector<unsigned int>> cellToVertexIndexMap;
    cellToVertexIndexMap.resize(
      parallelMeshUnmoved.n_locally_owned_active_cells());

    std::vector<std::vector<dealii::Point<3>>> cellToVertexParamCoordMap;
    cellToVertexParamCoordMap.resize(
      parallelMeshUnmoved.n_locally_owned_active_cells());

    std::vector<dealii::Point<3>> newVertexPosition;
    newVertexPosition.resize(d_parallelTriangulationMovedVxc.n_vertices());
    for (auto &cellIterVxc :
         d_parallelTriangulationMovedVxc.active_cell_iterators())
      {
        if (cellIterVxc->is_locally_owned())
          {
            for (unsigned int vertex_no = 0;
                 vertex_no < GeometryInfo<3>::vertices_per_cell;
                 ++vertex_no)
              {
                const unsigned global_vertex_no =
                  cellIterVxc->vertex_index(vertex_no);

                if (gridPointTouched[global_vertex_no] ||
                    !locally_owned_vertices[global_vertex_no])
                  continue;

                dealii::Point<3> P_real = cellIterVxc->vertex(vertex_no);
                dealii::Point<3> P_ref;
                // can be made optimal by not going through all local cell ??
                unsigned int iElem = 0;
                for (auto &cellIter :
                     parallelMeshUnmoved.active_cell_iterators())
                  {
                    if (cellIter->is_locally_owned())
                      {
                        try
                          {
                            P_ref =
                              mapping.transform_real_to_unit_cell(cellIter,
                                                                  P_real);
                            bool x_coord = false, y_coord = false,
                                 z_coord = false;
                            if ((P_ref[0] > -1e-7) && (P_ref[0] < 1 + 1e-7))
                              {
                                x_coord = true;
                              }
                            if ((P_ref[1] > -1e-7) && (P_ref[1] < 1 + 1e-7))
                              {
                                y_coord = true;
                              }
                            if ((P_ref[2] > -1e-7) && (P_ref[2] < 1 + 1e-7))
                              {
                                z_coord = true;
                              }
                            if (x_coord && y_coord && z_coord)
                              {
                                // store necessary data here
                                cellToVertexIndexMap[iElem].push_back(
                                  global_vertex_no);
                                cellToVertexParamCoordMap[iElem].push_back(
                                  P_ref);
                                gridPointTouched[global_vertex_no] = true;
                              }
                          }
                        catch (...)
                          {}
                        iElem++;
                      }
                  }
              }
          }
      }

    unsigned int iCell = 0;
    for (auto &cellIter : parallelMeshMoved.active_cell_iterators())
      {
        if (cellIter->is_locally_owned())
          {
            for (unsigned int vertexIter = 0;
                 vertexIter < cellToVertexIndexMap[iCell].size();
                 vertexIter++)
              {
                dealii::Point<3> P_ref =
                  cellToVertexParamCoordMap[iCell][vertexIter];
                dealii::Point<3> P_real =
                  mapping.transform_unit_to_real_cell(cellIter, P_ref);
                newVertexPosition[cellToVertexIndexMap[iCell][vertexIter]] =
                  P_real;
              }
            iCell++;
          }
      }

    for (auto &cellIterVxc :
         d_parallelTriangulationMovedVxc.active_cell_iterators())
      {
        if (cellIterVxc->is_locally_owned())
          {
            for (unsigned int vertex_no = 0;
                 vertex_no < GeometryInfo<3>::vertices_per_cell;
                 ++vertex_no)
              {
                const unsigned global_vertex_no =
                  cellIterVxc->vertex_index(vertex_no);
                if (locally_owned_vertices[global_vertex_no] &&
                    !vertex_moved[global_vertex_no])
                  {
                    cellIterVxc->vertex(vertex_no) =
                      newVertexPosition[global_vertex_no];
                    vertex_moved[global_vertex_no] = true;
                  }
              }
          }
      }

    d_parallelTriangulationMovedVxc.communicate_locally_moved_vertices(
      locally_owned_vertices);
  }

  void
  TriangulationManagerVxc::computeMapBetweenParentAndChildMesh(
    const parallel::distributed::Triangulation<3> &parallelParentMesh,
    const parallel::distributed::Triangulation<3> &parallelChildMesh,
    std::vector<std::vector<unsigned int>> &       mapParentCellToChildCells,
    std::vector<std::map<unsigned int,
                         typename dealii::DoFHandler<3>::active_cell_iterator>>
                              &                        mapParentCellToChildCellsIter,
    std::vector<unsigned int> &mapChildCellsToParentCell,
    unsigned int &             maxChildCellRefinementWithParent)
  {
    maxChildCellRefinementWithParent = 0;
    mapChildCellsToParentCell.resize(
      parallelChildMesh.n_locally_owned_active_cells());
    mapParentCellToChildCells.resize(
      parallelParentMesh.n_locally_owned_active_cells());
    mapParentCellToChildCellsIter.resize(
      parallelParentMesh.n_locally_owned_active_cells());

    std::vector<unsigned int> numChildCellsInParentCell(
      parallelParentMesh.n_locally_owned_active_cells());
    std::fill(numChildCellsInParentCell.begin(),
              numChildCellsInParentCell.end(),
              0);
    // using a linear mapping
    dealii::MappingQGeneric<3, 3> mapping(1);

    unsigned int childCellIndex = 0;
    for (auto &cellChildIter : parallelChildMesh.active_cell_iterators())
      {
        if (cellChildIter->is_locally_owned())
          {
            unsigned int     parentCellIndex = 0;
            dealii::Point<3> childCellCenter = cellChildIter->center();
            for (auto &parentChildIter :
                 parallelParentMesh.active_cell_iterators())
              {
                if (parentChildIter->is_locally_owned())
                  {
                    try
                      {
                        dealii::Point<3> P_ref =
                          mapping.transform_real_to_unit_cell(parentChildIter,
                                                              childCellCenter);
                        bool x_coord = false, y_coord = false, z_coord = false;
                        if ((P_ref[0] > -1e-10) && (P_ref[0] < 1 + 1e-10))
                          {
                            x_coord = true;
                          }
                        if ((P_ref[1] > -1e-10) && (P_ref[1] < 1 + 1e-10))
                          {
                            y_coord = true;
                          }
                        if ((P_ref[2] > -1e-10) && (P_ref[2] < 1 + 1e-10))
                          {
                            z_coord = true;
                          }
                        if (x_coord && y_coord && z_coord)
                          {
                            mapChildCellsToParentCell[childCellIndex] =
                              parentCellIndex;
                            mapParentCellToChildCells[parentCellIndex]
                              .push_back(childCellIndex);
                            //                                mapParentCellToChildCellsIter[parentCellIndex].push_back(cellChildIter);
                            mapParentCellToChildCellsIter
                              [parentCellIndex]
                              [numChildCellsInParentCell[parentCellIndex]] =
                                cellChildIter;
                            unsigned int relativeRefinement =
                              std::abs(cellChildIter->level() -
                                       parentChildIter->level());
                            if (relativeRefinement >
                                maxChildCellRefinementWithParent)
                              {
                                maxChildCellRefinementWithParent =
                                  relativeRefinement;
                              }
                            numChildCellsInParentCell[parentCellIndex] =
                              numChildCellsInParentCell[parentCellIndex] + 1;
                          }
                      }
                    catch (...)
                      {}
                    parentCellIndex++;
                  }
              }
            childCellIndex++;
          }
      }
  }

  //
  // get moved parallel mesh
  //
  parallel::distributed::Triangulation<3> &
  TriangulationManagerVxc::getParallelMovedMeshVxc()
  {
    return d_parallelTriangulationMovedVxc;
  }

  //
  // get unmoved parallel mesh
  //
  parallel::distributed::Triangulation<3> &
  TriangulationManagerVxc::getParallelUnmovedMeshVxc()
  {
    return d_parallelTriangulationUnmovedVxc;
  }
} // namespace dftfe

//  bool triangulationManagerVxc::refinementAlgorithmAWithManualRepartition(
//    parallel::distributed::Triangulation<3> &parallelTriangulation,
//    std::vector<unsigned int> &              locallyOwnedCellsRefineFlags,
//    std::map<dealii::CellId, unsigned int> & cellIdToCellRefineFlagMapLocal,
//    const bool                               smoothenCellsOnPeriodicBoundary,
//    const double                             smootheningFactor)
//  {
//    //
//    // compute magnitudes of domainBounding Vectors
//    //
//    const double domainBoundingVectorMag1 =
//      sqrt(d_domainBoundingVectors[0][0] * d_domainBoundingVectors[0][0] +
//           d_domainBoundingVectors[0][1] * d_domainBoundingVectors[0][1] +
//           d_domainBoundingVectors[0][2] * d_domainBoundingVectors[0][2]);
//    const double domainBoundingVectorMag2 =
//      sqrt(d_domainBoundingVectors[1][0] * d_domainBoundingVectors[1][0] +
//           d_domainBoundingVectors[1][1] * d_domainBoundingVectors[1][1] +
//           d_domainBoundingVectors[1][2] * d_domainBoundingVectors[1][2]);
//    const double domainBoundingVectorMag3 =
//      sqrt(d_domainBoundingVectors[2][0] * d_domainBoundingVectors[2][0] +
//           d_domainBoundingVectors[2][1] * d_domainBoundingVectors[2][1] +
//           d_domainBoundingVectors[2][2] * d_domainBoundingVectors[2][2]);
//
//    locallyOwnedCellsRefineFlags.clear();
//    cellIdToCellRefineFlagMapLocal.clear();
//    typename parallel::distributed::Triangulation<3>::active_cell_iterator
//    cell,
//      endc;
//    cell = parallelTriangulation.begin_active();
//    endc = parallelTriangulation.end();
//
//    std::map<dealii::CellId, unsigned int> cellIdToLocallyOwnedId;
//    unsigned int                           locallyOwnedCount = 0;
//
//    bool   isAnyCellRefined           = false;
//    double smallestMeshSizeAroundAtom = d_dftParams.meshSizeOuterBall;
//
//    if (d_dftParams.useMeshSizesFromAtomsFile)
//      {
//        smallestMeshSizeAroundAtom = 1e+6;
//        for (unsigned int n = 0; n < d_atomPositions.size(); n++)
//          {
//            if (d_atomPositions[n][5] < smallestMeshSizeAroundAtom)
//              smallestMeshSizeAroundAtom = d_atomPositions[n][5];
//          }
//      }
//
//    std::vector<double>       atomPointsLocal;
//    std::vector<unsigned int> atomIdsLocal;
//    std::vector<double>       meshSizeAroundAtomLocalAtoms;
//    std::vector<double>       outerAtomBallRadiusLocalAtoms;
//    for (unsigned int iAtom = 0;
//         iAtom < (d_atomPositions.size() + d_imageAtomPositions.size());
//         iAtom++)
//      {
//        if (iAtom < d_atomPositions.size())
//          {
//            atomPointsLocal.push_back(d_atomPositions[iAtom][2]);
//            atomPointsLocal.push_back(d_atomPositions[iAtom][3]);
//            atomPointsLocal.push_back(d_atomPositions[iAtom][4]);
//            atomIdsLocal.push_back(iAtom);
//
//            meshSizeAroundAtomLocalAtoms.push_back(
//              d_dftParams.useMeshSizesFromAtomsFile ?
//                d_atomPositions[iAtom][5] :
//                d_dftParams.meshSizeOuterBall);
//            outerAtomBallRadiusLocalAtoms.push_back(
//              d_dftParams.useMeshSizesFromAtomsFile ?
//                d_atomPositions[iAtom][6] :
//                d_dftParams.outerAtomBallRadius);
//          }
//        else
//          {
//            const unsigned int iImageCharge = iAtom - d_atomPositions.size();
//            atomPointsLocal.push_back(d_imageAtomPositions[iImageCharge][0]);
//            atomPointsLocal.push_back(d_imageAtomPositions[iImageCharge][1]);
//            atomPointsLocal.push_back(d_imageAtomPositions[iImageCharge][2]);
//            const unsigned int imageChargeId = d_imageIds[iImageCharge];
//            atomIdsLocal.push_back(imageChargeId);
//
//            meshSizeAroundAtomLocalAtoms.push_back(
//              d_dftParams.useMeshSizesFromAtomsFile ?
//                d_atomPositions[imageChargeId][5] :
//                d_dftParams.meshSizeOuterBall);
//            outerAtomBallRadiusLocalAtoms.push_back(
//              d_dftParams.useMeshSizesFromAtomsFile ?
//                d_atomPositions[imageChargeId][6] :
//                d_dftParams.outerAtomBallRadius);
//          }
//      }
//
//    //
//    //
//    //
//    for (; cell != endc; ++cell)
//      {
//        if (cell->is_locally_owned())
//          {
//            cellIdToLocallyOwnedId[cell->id()] = locallyOwnedCount;
//            locallyOwnedCount++;
//
//            const dealii::Point<3> center(cell->center());
//            double currentMeshSize = cell->minimum_vertex_distance();
//
//            //
//            // compute projection of the vector joining the center of domain
//            and
//            // centroid of cell onto each of the domain bounding vectors
//            //
//            double projComponent_1 =
//              (center[0] * d_domainBoundingVectors[0][0] +
//               center[1] * d_domainBoundingVectors[0][1] +
//               center[2] * d_domainBoundingVectors[0][2]) /
//              domainBoundingVectorMag1;
//            double projComponent_2 =
//              (center[0] * d_domainBoundingVectors[1][0] +
//               center[1] * d_domainBoundingVectors[1][1] +
//               center[2] * d_domainBoundingVectors[1][2]) /
//              domainBoundingVectorMag2;
//            double projComponent_3 =
//              (center[0] * d_domainBoundingVectors[2][0] +
//               center[1] * d_domainBoundingVectors[2][1] +
//               center[2] * d_domainBoundingVectors[2][2]) /
//              domainBoundingVectorMag3;
//
//
//            bool cellRefineFlag = false;
//
//
//            // loop over all atoms
//            double       distanceToClosestAtom = 1e8;
//            Point<3>     closestAtom;
//            unsigned int closestId = 0;
//            for (unsigned int n = 0; n < atomPointsLocal.size() / 3; n++)
//              {
//                Point<3> atom(atomPointsLocal[3 * n],
//                              atomPointsLocal[3 * n + 1],
//                              atomPointsLocal[3 * n + 2]);
//                if (center.distance(atom) < distanceToClosestAtom)
//                  {
//                    distanceToClosestAtom = center.distance(atom);
//                    closestAtom           = atom;
//                    closestId             = n;
//                  }
//              }
//
//            if (d_dftParams.autoAdaptBaseMeshSize)
//              {
//                bool inOuterAtomBall = false;
//
//                if (distanceToClosestAtom <=
//                    outerAtomBallRadiusLocalAtoms[closestId])
//                  inOuterAtomBall = true;
//
//                if (inOuterAtomBall &&
//                    (currentMeshSize >
//                     1.2 * meshSizeAroundAtomLocalAtoms[closestId]))
//                  cellRefineFlag = true;
//
//                bool inInnerAtomBall = false;
//
//                if (distanceToClosestAtom <= d_dftParams.innerAtomBallRadius)
//                  inInnerAtomBall = true;
//
//                if (inInnerAtomBall &&
//                    currentMeshSize > 1.2 * d_dftParams.meshSizeInnerBall)
//                  cellRefineFlag = true;
//              }
//            else
//              {
//                bool inOuterAtomBall = false;
//
//                if (distanceToClosestAtom <=
//                    outerAtomBallRadiusLocalAtoms[closestId])
//                  inOuterAtomBall = true;
//
//                if (inOuterAtomBall &&
//                    (currentMeshSize >
//                    meshSizeAroundAtomLocalAtoms[closestId]))
//                  cellRefineFlag = true;
//
//                bool inInnerAtomBall = false;
//
//                if (distanceToClosestAtom <= d_dftParams.innerAtomBallRadius)
//                  inInnerAtomBall = true;
//
//                if (inInnerAtomBall &&
//                    currentMeshSize > d_dftParams.meshSizeInnerBall)
//                  cellRefineFlag = true;
//              }
//
//            /*
//            if (d_dftParams.autoAdaptBaseMeshSize  &&
//            !d_dftParams.reproducible_output)
//            {
//              bool inBiggerAtomBall = false;
//
//              if(distanceToClosestAtom <= 10.0)
//                inBiggerAtomBall = true;
//
//              if(inBiggerAtomBall && currentMeshSize > 6.0)
//                cellRefineFlag = true;
//            }
//            */
//
//            MappingQ1<3, 3> mapping;
//            try
//              {
//                Point<3> p_cell =
//                  mapping.transform_real_to_unit_cell(cell, closestAtom);
//                double dist = GeometryInfo<3>::distance_to_unit_cell(p_cell);
//
//                if (dist < 1e-08 &&
//                    ((currentMeshSize > d_dftParams.meshSizeInnerBall) ||
//                     (currentMeshSize >
//                      1.2 * meshSizeAroundAtomLocalAtoms[closestId])))
//                  cellRefineFlag = true;
//              }
//            catch (MappingQ1<3>::ExcTransformationFailed)
//              {}
//
//            cellRefineFlag =
//              Utilities::MPI::max((unsigned int)cellRefineFlag,
//              interpoolcomm);
//            cellRefineFlag = Utilities::MPI::max((unsigned int)cellRefineFlag,
//                                                 interBandGroupComm);
//
//            //
//            // set refine flags
//            if (cellRefineFlag)
//              {
//                locallyOwnedCellsRefineFlags.push_back(1);
//                cellIdToCellRefineFlagMapLocal[cell->id()] = 1;
//                cell->set_refine_flag();
//                isAnyCellRefined = true;
//              }
//            else
//              {
//                cellIdToCellRefineFlagMapLocal[cell->id()] = 0;
//                locallyOwnedCellsRefineFlags.push_back(0);
//              }
//          }
//      }
//
//
//    //
//    // refine cells on periodic boundary if their length is greater than
//    // mesh size around atom by a factor (set by smootheningFactor)
//    //
//    if (smoothenCellsOnPeriodicBoundary)
//      {
//        locallyOwnedCount = 0;
//        cell              = parallelTriangulation.begin_active();
//        endc              = parallelTriangulation.end();
//
//        const unsigned int faces_per_cell =
//          dealii::GeometryInfo<3>::faces_per_cell;
//
//        for (; cell != endc; ++cell)
//          {
//            if (cell->is_locally_owned())
//              {
//                if (cell->at_boundary() &&
//                    cell->minimum_vertex_distance() >
//                      (d_dftParams.autoAdaptBaseMeshSize ? 1.5 : 1) *
//                        smootheningFactor * smallestMeshSizeAroundAtom &&
//                    !cell->refine_flag_set())
//                  for (unsigned int iFace = 0; iFace < faces_per_cell;
//                  ++iFace)
//                    if (cell->has_periodic_neighbor(iFace))
//                      {
//                        cell->set_refine_flag();
//                        isAnyCellRefined = true;
//                        locallyOwnedCellsRefineFlags
//                          [cellIdToLocallyOwnedId[cell->id()]]     = 1;
//                        cellIdToCellRefineFlagMapLocal[cell->id()] = 1;
//                        break;
//                      }
//                locallyOwnedCount++;
//              }
//          }
//      }
//
//    return isAnyCellRefined;
//  }

//  bool triangulationManagerVxc::consistentPeriodicBoundaryRefinementForVxc(
//    parallel::distributed::Triangulation<3> &parallelTriangulation,
//    std::vector<unsigned int> &              locallyOwnedCellsRefineFlags,
//    std::map<dealii::CellId, unsigned int> & cellIdToCellRefineFlagMapLocal)
//  {
//    locallyOwnedCellsRefineFlags.clear();
//    cellIdToCellRefineFlagMapLocal.clear();
//    typename parallel::distributed::Triangulation<3>::active_cell_iterator
//    cell,
//      endc;
//    cell = parallelTriangulation.begin_active();
//    endc = parallelTriangulation.end();
//
//    //
//    // populate maps refinement flag maps to zero values
//    //
//    std::map<dealii::CellId, unsigned int> cellIdToLocallyOwnedId;
//    unsigned int                           locallyOwnedCount = 0;
//    for (; cell != endc; ++cell)
//      if (cell->is_locally_owned())
//        {
//          cellIdToLocallyOwnedId[cell->id()] = locallyOwnedCount;
//          locallyOwnedCellsRefineFlags.push_back(0);
//          cellIdToCellRefineFlagMapLocal[cell->id()] = 0;
//          locallyOwnedCount++;
//        }
//
//
//    cell = parallelTriangulation.begin_active();
//    endc = parallelTriangulation.end();
//
//
//    //
//    // go to each locally owned or ghost cell which has a face on the periodic
//    // boundary-> query if cell has a periodic neighbour which is coarser ->
//    if
//    // yes and the coarse cell is locally owned set refinement flag on that
//    cell
//    //
//    const unsigned int faces_per_cell =
//    dealii::GeometryInfo<3>::faces_per_cell; bool isAnyCellRefined = false;
//    for (; cell != endc; ++cell)
//      {
//        if ((cell->is_locally_owned() || cell->is_ghost()) &&
//            cell->at_boundary())
//          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
//            if (cell->has_periodic_neighbor(iFace))
//              if (cell->periodic_neighbor_is_coarser(iFace))
//                {
//                  typename parallel::distributed::Triangulation<
//                    3>::active_cell_iterator periodicCell =
//                    cell->periodic_neighbor(iFace);
//
//                  if (periodicCell->is_locally_owned())
//                    {
//                      locallyOwnedCellsRefineFlags
//                        [cellIdToLocallyOwnedId[periodicCell->id()]]     = 1;
//                      cellIdToCellRefineFlagMapLocal[periodicCell->id()] = 1;
//                      periodicCell->set_refine_flag();
//
//                      isAnyCellRefined = true;
//                    }
//                }
//      }
//    return isAnyCellRefined;
//  }

//  void triangulationManagerVxc::generateCoarseMeshForVxc(
//    parallel::distributed::Triangulation<3> &parallelTriangulation)
//  {
//    //generate coarse mesh
//    //
//    // compute magnitudes of domainBounding Vectors
//    //
//    const double domainBoundingVectorMag1 =
//      sqrt(d_domainBoundingVectors[0][0] * d_domainBoundingVectors[0][0] +
//           d_domainBoundingVectors[0][1] * d_domainBoundingVectors[0][1] +
//           d_domainBoundingVectors[0][2] * d_domainBoundingVectors[0][2]);
//    const double domainBoundingVectorMag2 =
//      sqrt(d_domainBoundingVectors[1][0] * d_domainBoundingVectors[1][0] +
//           d_domainBoundingVectors[1][1] * d_domainBoundingVectors[1][1] +
//           d_domainBoundingVectors[1][2] * d_domainBoundingVectors[1][2]);
//    const double domainBoundingVectorMag3 =
//      sqrt(d_domainBoundingVectors[2][0] * d_domainBoundingVectors[2][0] +
//           d_domainBoundingVectors[2][1] * d_domainBoundingVectors[2][1] +
//           d_domainBoundingVectors[2][2] * d_domainBoundingVectors[2][2]);
//
//    unsigned int subdivisions[3];
//    subdivisions[0] = 1.0;
//    subdivisions[1] = 1.0;
//    subdivisions[2] = 1.0;
//
//    std::vector<double> numberIntervalsEachDirection;
//
//    double largestMeshSizeAroundAtom = d_dftParams.meshSizeOuterBall;
//
//    if (d_dftParams.useMeshSizesFromAtomsFile)
//      {
//        largestMeshSizeAroundAtom = 1e-6;
//        for (unsigned int n = 0; n < d_atomPositions.size(); n++)
//          {
//            if (d_atomPositions[n][5] > largestMeshSizeAroundAtom)
//              largestMeshSizeAroundAtom = d_atomPositions[n][5];
//          }
//      }
//
//    if (d_dftParams.autoAdaptBaseMeshSize)
//      {
//        double baseMeshSize1, baseMeshSize2, baseMeshSize3;
//        if (d_dftParams.periodicX || d_dftParams.periodicY ||
//            d_dftParams.periodicZ)
//          {
//            const double targetBaseMeshSize =
//              (std::min(std::min(domainBoundingVectorMag1,
//                                 domainBoundingVectorMag2),
//                        domainBoundingVectorMag3) > 50.0) ?
//                7.0 :
//                std::max(2.0, largestMeshSizeAroundAtom);
//            baseMeshSize1 = std::pow(2,
//                                     round(log2(targetBaseMeshSize /
//                                                largestMeshSizeAroundAtom))) *
//                            largestMeshSizeAroundAtom;
//            baseMeshSize2 = std::pow(2,
//                                     round(log2(targetBaseMeshSize /
//                                                largestMeshSizeAroundAtom))) *
//                            largestMeshSizeAroundAtom;
//            baseMeshSize3 = std::pow(2,
//                                     round(log2(targetBaseMeshSize /
//                                                largestMeshSizeAroundAtom))) *
//                            largestMeshSizeAroundAtom;
//          }
//        else
//          {
//            baseMeshSize1 =
//              std::pow(2,
//                       round(
//                         log2(std::min(domainBoundingVectorMag1 / 8.0, 8.0) /
//                              largestMeshSizeAroundAtom))) *
//              largestMeshSizeAroundAtom;
//            baseMeshSize2 =
//              std::pow(2,
//                       round(
//                         log2(std::min(domainBoundingVectorMag2 / 8.0, 8.0) /
//                              largestMeshSizeAroundAtom))) *
//              largestMeshSizeAroundAtom;
//            baseMeshSize3 =
//              std::pow(2,
//                       round(
//                         log2(std::min(domainBoundingVectorMag3 / 8.0, 8.0) /
//                              largestMeshSizeAroundAtom))) *
//              largestMeshSizeAroundAtom;
//          }
//
//        numberIntervalsEachDirection.push_back(domainBoundingVectorMag1 /
//                                               baseMeshSize1);
//        numberIntervalsEachDirection.push_back(domainBoundingVectorMag2 /
//                                               baseMeshSize2);
//        numberIntervalsEachDirection.push_back(domainBoundingVectorMag3 /
//                                               baseMeshSize3);
//      }
//    else
//      {
//        numberIntervalsEachDirection.push_back(domainBoundingVectorMag1 /
//                                               d_dftParams.meshSizeOuterDomain);
//        numberIntervalsEachDirection.push_back(domainBoundingVectorMag2 /
//                                               d_dftParams.meshSizeOuterDomain);
//        numberIntervalsEachDirection.push_back(domainBoundingVectorMag3 /
//                                               d_dftParams.meshSizeOuterDomain);
//      }
//
//    Point<3> vector1(d_domainBoundingVectors[0][0],
//                     d_domainBoundingVectors[0][1],
//                     d_domainBoundingVectors[0][2]);
//    Point<3> vector2(d_domainBoundingVectors[1][0],
//                     d_domainBoundingVectors[1][1],
//                     d_domainBoundingVectors[1][2]);
//    Point<3> vector3(d_domainBoundingVectors[2][0],
//                     d_domainBoundingVectors[2][1],
//                     d_domainBoundingVectors[2][2]);
//
//    //
//    // Generate coarse mesh
//    //
//    Point<3> basisVectors[3] = {vector1, vector2, vector3};
//
//
//    for (unsigned int i = 0; i < 3; i++)
//      {
//        const double temp = numberIntervalsEachDirection[i] -
//                            std::floor(numberIntervalsEachDirection[i]);
//        if (temp >= 0.5)
//          subdivisions[i] = std::ceil(numberIntervalsEachDirection[i]);
//        else
//          subdivisions[i] = std::floor(numberIntervalsEachDirection[i]);
//      }
//
//
//    //TODO check if the repartition() has to be called here ?
//    GridGenerator::subdivided_parallelepiped<3>(parallelTriangulation,
//                                                subdivisions,
//                                                basisVectors);
////    parallelTriangulation.repartition();
//
//    //
//    // Translate the main grid so that midpoint is at center
//    //
//    const Point<3> translation = 0.5 * (vector1 + vector2 + vector3);
//    GridTools::shift(-translation, parallelTriangulation);
//
//    //
//    // collect periodic faces of the first level mesh to set up periodic
//    // boundary conditions later
//    //
//    meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,
//                                                 d_domainBoundingVectors,
//                                                 d_mpiCommParent,
//                                                 d_dftParams);
//
//    if (d_dftParams.verbosity >= 4)
//      pcout << std::endl
//            << "Coarse triangulation number of elements: "
//            << parallelTriangulation.n_global_active_cells() << std::endl;
//
//
//  }

//  void
//  triangulationManagerVxc::generateMeshWithManualRepartitioning(parallel::distributed::Triangulation<3>
//  & parallelMesh)
//  {
//
//    generateCoarseMeshForVxc(parallelMesh) ;
//
//
//
//    unsigned int numLevels  = 0;
//    bool         refineFlag = true;
//    while (refineFlag)
//      {
//        refineFlag = false;
//        std::vector<unsigned int>              locallyOwnedCellsRefineFlags;
//        std::map<dealii::CellId, unsigned int> cellIdToCellRefineFlagMapLocal;
//
//        refineFlag = refinementAlgorithmAWithManualRepartition(parallelMesh,
//                                          locallyOwnedCellsRefineFlags,
//                                          cellIdToCellRefineFlagMapLocal);
//
//        // This sets the global refinement sweep flag
//        refineFlag =
//          Utilities::MPI::max((unsigned int)refineFlag, mpi_communicator);
//
//        // Refine
//        if (refineFlag)
//          {
//            if (numLevels < d_max_refinement_steps)
//              {
//                if (d_dftParams.verbosity >= 4)
//                  pcout << "refinement in progress, level: " << numLevels
//                        << std::endl;
//
//                d_parallelTriaVxcCurrentRefinement.push_back(std::vector<bool>());
//                parallelMesh.save_refine_flags(
//                  d_parallelTriaVxcCurrentRefinement[numLevels]);
//
//                parallelMesh.execute_coarsening_and_refinement();
//                parallelMesh.repartition();
//
//                numLevels++;
//              }
//            else
//              {
//                refineFlag = false;
//              }
//          }
//      }
//
//    //TODO is this required and possible
//    if (!d_dftParams.reproducible_output &&
//        !d_dftParams.createConstraintsFromSerialDofhandler)
//      {
//        //
//        // STAGE1: This stage is only activated if combined periodic and
//        hanging
//        // node constraints are not consistent in parallel. Call
//        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement
//        // alternatively. In the call to refinementAlgorithmA there is no
//        // additional reduction of adaptivity performed on the periodic
//        // boundary. Multilevel refinement is performed until both
//        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement do
//        not
//        // set refinement flags on any cell.
//        //
//        if (!checkConstraintsConsistency(parallelMesh))
//          {
//            refineFlag = true;
//            while (refineFlag)
//              {
//                refineFlag = false;
//                std::vector<unsigned int> locallyOwnedCellsRefineFlags;
//                std::map<dealii::CellId, unsigned int>
//                  cellIdToCellRefineFlagMapLocal;
//                if (numLevels % 2 == 0)
//                  {
//                    refineFlag =
//                      refinementAlgorithmAWithManualRepartition(parallelMesh,
//                                           locallyOwnedCellsRefineFlags,
//                                           cellIdToCellRefineFlagMapLocal);
//
//                    // This sets the global refinement sweep flag
//                    refineFlag = Utilities::MPI::max((unsigned int)refineFlag,
//                                                     mpi_communicator);
//
//                    // try the other type of refinement to prevent while loop
//                    // from ending prematurely
//                    if (!refineFlag)
//                      {
//                        // call refinement algorithm  which sets refinement
//                        // flags such as to create consistent refinement
//                        across
//                        // periodic boundary
//                        refineFlag =
//                        consistentPeriodicBoundaryRefinementForVxc(
//                          parallelMesh,
//                          locallyOwnedCellsRefineFlags,
//                          cellIdToCellRefineFlagMapLocal);
//
//                        // This sets the global refinement sweep flag
//                        refineFlag =
//                          Utilities::MPI::max((unsigned int)refineFlag,
//                                              mpi_communicator);
//                      }
//                  }
//                else
//                  {
//                    // call refinement algorithm  which sets refinement flags
//                    // such as to create consistent refinement across periodic
//                    // boundary
//                    refineFlag = consistentPeriodicBoundaryRefinementForVxc(
//                      parallelMesh,
//                      locallyOwnedCellsRefineFlags,
//                      cellIdToCellRefineFlagMapLocal);
//
//                    // This sets the global refinement sweep flag
//                    refineFlag = Utilities::MPI::max((unsigned int)refineFlag,
//                                                     mpi_communicator);
//
//                    // try the other type of refinement to prevent while loop
//                    // from ending prematurely
//                    if (!refineFlag)
//                      {
//                        refineFlag =
//                          refinementAlgorithmAWithManualRepartition(parallelMesh,
//                                               locallyOwnedCellsRefineFlags,
//                                               cellIdToCellRefineFlagMapLocal);
//
//                        // This sets the global refinement sweep flag
//                        refineFlag =
//                          Utilities::MPI::max((unsigned int)refineFlag,
//                                              mpi_communicator);
//                      }
//                  }
//
//                // Refine
//                if (refineFlag)
//                  {
//                    if (numLevels < d_max_refinement_steps)
//                      {
//                        if (d_dftParams.verbosity >= 4)
//                          pcout
//                            << "refinement in progress, level: " << numLevels
//                            << std::endl;
//
//
//                        d_parallelTriaVxcCurrentRefinement.push_back(
//                          std::vector<bool>());
//                        parallelMesh.save_refine_flags(
//                          d_parallelTriaVxcCurrentRefinement[numLevels]);
//
//                        parallelMesh
//                          .execute_coarsening_and_refinement();
//                        parallelMesh.repartition();
//                        numLevels++;
//                      }
//                    else
//                      {
//                        refineFlag = false;
//                      }
//                  }
//              }
//          }
//
//        //
//        // STAGE2: This stage is only activated if combined periodic and
//        hanging
//        // node constraints are still not consistent in parallel. Call
//        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement
//        // alternatively. In the call to refinementAlgorithmA there is an
//        // additional reduction of adaptivity performed on the periodic
//        boundary
//        // such that the maximum cell length on the periodic boundary is less
//        // than two times the MESH SIZE AROUND ATOM. Multilevel refinement is
//        // performed until both refinementAlgorithmAand
//        // consistentPeriodicBoundaryRefinement do not set refinement flags on
//        // any cell.
//        //
//        if (!checkConstraintsConsistency(parallelMesh))
//          {
//            refineFlag = true;
//            while (refineFlag)
//              {
//                refineFlag = false;
//                std::vector<unsigned int> locallyOwnedCellsRefineFlags;
//                std::map<dealii::CellId, unsigned int>
//                  cellIdToCellRefineFlagMapLocal;
//                if (numLevels % 2 == 0)
//                  {
//                    refineFlag =
//                      refinementAlgorithmAWithManualRepartition(parallelMesh,
//                                           locallyOwnedCellsRefineFlags,
//                                           cellIdToCellRefineFlagMapLocal,
//                                           true,
//                                           2.0);
//
//                    // This sets the global refinement sweep flag
//                    refineFlag = Utilities::MPI::max((unsigned int)refineFlag,
//                                                     mpi_communicator);
//
//                    // try the other type of refinement to prevent while loop
//                    // from ending prematurely
//                    if (!refineFlag)
//                      {
//                        // call refinement algorithm  which sets refinement
//                        // flags such as to create consistent refinement
//                        across
//                        // periodic boundary
//                        refineFlag =
//                        consistentPeriodicBoundaryRefinementForVxc(
//                          parallelMesh,
//                          locallyOwnedCellsRefineFlags,
//                          cellIdToCellRefineFlagMapLocal);
//
//                        // This sets the global refinement sweep flag
//                        refineFlag =
//                          Utilities::MPI::max((unsigned int)refineFlag,
//                                              mpi_communicator);
//                      }
//                  }
//                else
//                  {
//                    // call refinement algorithm  which sets refinement flags
//                    // such as to create consistent refinement across periodic
//                    // boundary
//                    refineFlag = consistentPeriodicBoundaryRefinementForVxc(
//                      parallelMesh,
//                      locallyOwnedCellsRefineFlags,
//                      cellIdToCellRefineFlagMapLocal);
//
//                    // This sets the global refinement sweep flag
//                    refineFlag = Utilities::MPI::max((unsigned int)refineFlag,
//                                                     mpi_communicator);
//
//                    // try the other type of refinement to prevent while loop
//                    // from ending prematurely
//                    if (!refineFlag)
//                      {
//                        refineFlag =
//                          refinementAlgorithmAWithManualRepartition(parallelMesh,
//                                               locallyOwnedCellsRefineFlags,
//                                               cellIdToCellRefineFlagMapLocal,
//                                               true,
//                                               2.0);
//
//                        // This sets the global refinement sweep flag
//                        refineFlag =
//                          Utilities::MPI::max((unsigned int)refineFlag,
//                                              mpi_communicator);
//                      }
//                  }
//
//                // Refine
//                if (refineFlag)
//                  {
//                    if (numLevels < d_max_refinement_steps)
//                      {
//                        if (d_dftParams.verbosity >= 4)
//                          pcout
//                            << "refinement in progress, level: " << numLevels
//                            << std::endl;
//
//
//                        d_parallelTriaVxcCurrentRefinement.push_back(
//                          std::vector<bool>());
//                        parallelMesh.save_refine_flags(
//                          d_parallelTriaVxcCurrentRefinement[numLevels]);
//
//                        parallelMesh
//                          .execute_coarsening_and_refinement();
//                        parallelMesh.repartition();
//
//                        numLevels++;
//                      }
//                    else
//                      {
//                        refineFlag = false;
//                      }
//                  }
//              }
//          }
//
//        //
//        // STAGE3: This stage is only activated if combined periodic and
//        hanging
//        // node constraints are still not consistent in parallel. Call
//        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement
//        // alternatively. In the call to refinementAlgorithmA there is an
//        // additional reduction of adaptivity performed on the periodic
//        boundary
//        // such that the maximum cell length on the periodic boundary is less
//        // than MESH SIZE AROUND ATOM essentially ensuring uniform refinement
//        on
//        // the periodic boundary in the case of MESH SIZE AROUND ATOM being
//        same
//        // as MESH SIZE AT ATOM. Multilevel refinement is performed until both
//        // refinementAlgorithmA and consistentPeriodicBoundaryRefinement do
//        not
//        // set refinement flags on any cell.
//        //
//        if (!checkConstraintsConsistency(parallelMesh))
//          {
//            refineFlag = true;
//            while (refineFlag)
//              {
//                refineFlag = false;
//                std::vector<unsigned int> locallyOwnedCellsRefineFlags;
//                std::map<dealii::CellId, unsigned int>
//                  cellIdToCellRefineFlagMapLocal;
//                if (numLevels % 2 == 0)
//                  {
//                    refineFlag =
//                      refinementAlgorithmAWithManualRepartition(parallelMesh,
//                                           locallyOwnedCellsRefineFlags,
//                                           cellIdToCellRefineFlagMapLocal,
//                                           true,
//                                           1.0);
//
//                    // This sets the global refinement sweep flag
//                    refineFlag = Utilities::MPI::max((unsigned int)refineFlag,
//                                                     mpi_communicator);
//
//                    // try the other type of refinement to prevent while loop
//                    // from ending prematurely
//                    if (!refineFlag)
//                      {
//                        // call refinement algorithm  which sets refinement
//                        // flags such as to create consistent refinement
//                        across
//                        // periodic boundary
//                        refineFlag =
//                        consistentPeriodicBoundaryRefinementForVxc(
//                          parallelMesh,
//                          locallyOwnedCellsRefineFlags,
//                          cellIdToCellRefineFlagMapLocal);
//
//                        // This sets the global refinement sweep flag
//                        refineFlag =
//                          Utilities::MPI::max((unsigned int)refineFlag,
//                                              mpi_communicator);
//                      }
//                  }
//                else
//                  {
//                    // call refinement algorithm  which sets refinement flags
//                    // such as to create consistent refinement across periodic
//                    // boundary
//                    refineFlag = consistentPeriodicBoundaryRefinementForVxc(
//                      parallelMesh,
//                      locallyOwnedCellsRefineFlags,
//                      cellIdToCellRefineFlagMapLocal);
//
//                    // This sets the global refinement sweep flag
//                    refineFlag = Utilities::MPI::max((unsigned int)refineFlag,
//                                                     mpi_communicator);
//
//                    // try the other type of refinement to prevent while loop
//                    // from ending prematurely
//                    if (!refineFlag)
//                      {
//                        refineFlag =
//                          refinementAlgorithmAWithManualRepartition(parallelMesh,
//                                               locallyOwnedCellsRefineFlags,
//                                               cellIdToCellRefineFlagMapLocal,
//                                               true,
//                                               1.0);
//
//                        // This sets the global refinement sweep flag
//                        refineFlag =
//                          Utilities::MPI::max((unsigned int)refineFlag,
//                                              mpi_communicator);
//                      }
//                  }
//
//                // Refine
//                if (refineFlag)
//                  {
//                    if (numLevels < d_max_refinement_steps)
//                      {
//                        if (d_dftParams.verbosity >= 4)
//                          pcout
//                            << "refinement in progress, level: " << numLevels
//                            << std::endl;
//
//
//                        d_parallelTriaVxcCurrentRefinement.push_back(
//                          std::vector<bool>());
//                        parallelMesh.save_refine_flags(
//                          d_parallelTriaVxcCurrentRefinement[numLevels]);
//
//                        parallelMesh
//                          .execute_coarsening_and_refinement();
//                        parallelMesh.repartition();
//
//                        numLevels++;
//                      }
//                    else
//                      {
//                        refineFlag = false;
//                      }
//                  }
//              }
//          }
//
//        if (checkConstraintsConsistency(parallelMesh))
//          {
//            if (d_dftParams.verbosity >= 4)
//              pcout
//                << "Hanging node and periodic constraints parallel consistency
//                achieved."
//                << std::endl;
//          }
//        else
//          {
//            if (d_dftParams.verbosity >= 4)
//              pcout
//                << "Hanging node and periodic constraints parallel consistency
//                not achieved."
//                << std::endl;
//
//            AssertThrow(
//              d_dftParams.createConstraintsFromSerialDofhandler,
//              ExcMessage(
//                "DFT-FE error: this is due to a known issue related to hanging
//                node constraints in dealii. Please set CONSTRAINTS FROM SERIAL
//                DOFHANDLER = true under the Boundary conditions subsection in
//                the input parameters file to circumvent this issue."));
//          }
//      }
//
//
//  }
