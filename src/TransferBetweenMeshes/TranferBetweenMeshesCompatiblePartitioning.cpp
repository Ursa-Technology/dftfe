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

#include "inverseDFTDoFManager.h"
//#include "mkl.h"

#ifdef DFTFE_WITH_DEVICE
#  include "deviceDirectCCLWrapper.h"
#  include "operatorDevice.h"
#  include "elpaScalaManager.h"
#  include "dftParameters.h"
#  include <chebyshevOrthogonalizedSubspaceIterationSolverDevice.h>
#  include <dftUtils.h>
#  include <deviceKernelsGeneric.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <linearAlgebraOperations.h>
#  include <linearAlgebraOperationsDevice.h>
#  include <vectorUtilities.h>
#  include <deviceKernelsGeneric.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <vectorUtilities.h>
#endif

namespace dftfe
{
  namespace
  {
    unsigned int
    getChildCellIndexAtLevel(const dealii::Point<3> &cellCenter,
                             unsigned int            iLevel)
    {
      double cellSize = std::pow(2.0, iLevel);
      cellSize        = 1.0 / cellSize;

      unsigned int numCellsX = std::pow(2, iLevel);

      unsigned int cellIdX = cellCenter[0] / cellSize;
      unsigned int cellIdY = cellCenter[1] / cellSize;
      unsigned int cellIdZ = cellCenter[2] / cellSize;

      unsigned int cellId =
        cellIdX + (cellIdY + cellIdZ * numCellsX) * numCellsX;
      return cellId;
    }
  } // namespace
  double
  quadValue(double x, double y, double z, unsigned int index)
  {
    double val = 1;
    val        = x * x + y * y + z * z;
    return val;
  }


  TransferDataBetweenMeshesCompatiblePartitioning::TransferDataBetweenMeshesCompatiblePartitioning(
    const dealii::MatrixFree<3, double> &   matrixFreeParentData,
    const unsigned int                      matrixFreeParentVectorComponent,
    const unsigned int                      matrixFreeParentQuadratureComponent,
    const dealii::MatrixFree<3, double> &   matrixFreeChildData,
    const unsigned int                      matrixFreeChildVectorComponent,
    const unsigned int                      matrixFreeChildQuadratureComponent,
    std::vector<std::vector<unsigned int>> &mapParentCellsToChild,
    std::vector<std::map<unsigned int,
                         typename dealii::DoFHandler<3>::active_cell_iterator>>
                              &                        mapParentCellToChildCellsIter,
    std::vector<unsigned int> &mapChildCellsToParent,
    unsigned int               maxRelativeRefinement)
    : d_numInputVectorsForInterpolationFromParentNodesToChildQuad(0)
    , d_numInputVectorsForInterpolationFromChildNodesToParentQuad(0)
  {
    double startMap                       = MPI_Wtime();
    d_matrixFreeDataParentPtr             = &matrixFreeParentData;
    d_matrixFreeParentVectorComponent     = matrixFreeParentVectorComponent;
    d_matrixFreeParentQuadratureComponent = matrixFreeParentQuadratureComponent;

    d_dofHandlerParent = &d_matrixFreeDataParentPtr->get_dof_handler(
      d_matrixFreeParentVectorComponent);

    d_quadratureParent = d_matrixFreeDataParentPtr->get_quadrature(
      matrixFreeParentVectorComponent);

    d_numberQuadraturePointsParent = d_quadratureParent.size();

    d_totallyOwnedCellsParent = d_matrixFreeDataParentPtr->n_physical_cells();

    unsigned int numberDofsPerElementParent =
      d_dofHandlerParent->get_fe().dofs_per_cell;

    d_matrixFreeDataChildPtr             = &matrixFreeChildData;
    d_matrixFreeChildVectorComponent     = matrixFreeChildVectorComponent;
    d_matrixFreeChildQuadratureComponent = matrixFreeChildQuadratureComponent;

    d_dofHandlerChild = &d_matrixFreeDataChildPtr->get_dof_handler(
      d_matrixFreeChildVectorComponent);

    d_quadratureChild = d_matrixFreeDataChildPtr->get_quadrature(
      d_matrixFreeChildQuadratureComponent);

    d_numberQuadraturePointsChild = d_quadratureChild.size();

    d_totallyOwnedCellsChild = d_matrixFreeDataChildPtr->n_physical_cells();

    // TODO copy the reference instead of the whole vector
    d_mapChildCellsToParent         = mapChildCellsToParent;
    d_mapParentCellToChildCellsIter = mapParentCellToChildCellsIter;
    d_mapParentCellsToChild         = mapParentCellsToChild;

    if (d_mapChildCellsToParent.size() != d_totallyOwnedCellsChild)
      {
        std::cout << "Error ::::: number of child cells dont match\n";
      }

    if (d_mapParentCellsToChild.size() != d_totallyOwnedCellsParent)
      {
        std::cout << "Error ::::: number of parent cells dont match\n";
      }

    std::cout << " child cells = " << d_totallyOwnedCellsChild << " "
              << d_mapChildCellsToParent.size() << "\n";
    std::cout << " parent cell = " << d_totallyOwnedCellsParent << " "
              << d_mapParentCellsToChild.size() << "\n";

    double startParam = MPI_Wtime();

    std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>>>
      parentCellQuadDataIndex;
    std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
      parentCellQuadDataShapeVal;
    std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
      childCellQuadDataShapeVal;

    getShapeFuncValsForParametricCell(maxRelativeRefinement,
                                      parentCellQuadDataIndex,
                                      parentCellQuadDataShapeVal,
                                      childCellQuadDataShapeVal);

    double startParent = MPI_Wtime();
    computeShapeFuncValuesOnParentCell(childCellQuadDataShapeVal);

    double startChild = MPI_Wtime();
    computeShapeFuncValuesOnChildCell(parentCellQuadDataIndex,
                                      parentCellQuadDataShapeVal);

    double endChild = MPI_Wtime();
    std::cout << " map start = " << startParam - startMap
              << "param = " << startParent - startParam
              << "  parent = " << startChild - startParent
              << " child = " << endChild - startChild << "\n";
  }

  void
  TransferDataBetweenMeshesCompatiblePartitioning::getShapeFuncValsForParametricCell(
    unsigned int maxRelativeRefinement,
    std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>>>
      &parentCellQuadDataIndex,
    std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
      &parentCellQuadDataShapeVal,
    std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
      &childCellQuadDataShapeVal)
  {
    // create parent triangulation and dofHandler
    // and quad points and parentFeValues

    dealii::Triangulation<3> parentTria;

    std::vector<unsigned int> subdivisions = {1, 1, 1};

    unsigned int dealiiSubdivisions[3];
    std::copy(subdivisions.begin(), subdivisions.end(), dealiiSubdivisions);

    dealii::Point<3, double> domainVectors[3];


    double xmin = 1.0;
    double ymin = 1.0;
    double zmin = 1.0;

    domainVectors[0][0] = xmin;
    domainVectors[1][1] = ymin;
    domainVectors[2][2] = zmin;

    dealii::GridGenerator::subdivided_parallelepiped<3>(parentTria,
                                                        dealiiSubdivisions,
                                                        domainVectors);

    dealii::DoFHandler<3> dofHandlerParent(parentTria);
    dofHandlerParent.distribute_dofs(d_dofHandlerParent->get_fe());

    dealii::FEValues<3> fe_valuesParent(dofHandlerParent.get_fe(),
                                        d_quadratureParent,
                                        dealii::update_values |
                                          dealii::update_quadrature_points);


    const dealii::FiniteElement<3> &feParent = d_dofHandlerParent->get_fe();

    unsigned int numberDofsPerElementParent = feParent.dofs_per_cell;

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellParent = dofHandlerParent.begin_active(),
      endcParent = dofHandlerParent.end();

    unsigned int numParentQuadPoints = d_quadratureParent.size();

    fe_valuesParent.reinit(cellParent);

    // create child triangulation

    dealii::Triangulation<3> childTria;

    dealii::GridGenerator::subdivided_parallelepiped<3>(childTria,
                                                        dealiiSubdivisions,
                                                        domainVectors);

    unsigned int          numChildQuadPoints = d_quadratureChild.size();
    dealii::DoFHandler<3> dofHandlerChild(childTria);

    dealii::MappingQGeneric<3, 3> mapping(1);
    // iterate through the levels
    std::cout<<" max refinement level = "<<maxRelativeRefinement<<"\n";
    for (unsigned int iLevel = 0; iLevel <= maxRelativeRefinement; iLevel++)
      {
        //std::cout << " iLevel = " << iLevel << "\n";
        // create child dofHandler
        dofHandlerChild.distribute_dofs(d_dofHandlerChild->get_fe());

        const dealii::FiniteElement<3> &feChild = dofHandlerChild.get_fe();

        dealii::FEValues<3> fe_valuesChild(dofHandlerChild.get_fe(),
                                           d_quadratureChild,
                                           dealii::update_values |
                                             dealii::update_quadrature_points);

        //        std::vector<bool> quadPointsTouched(numParentQuadPoints,
        //        false);

        unsigned int numberDofsPerElementChild =
          dofHandlerChild.get_fe().dofs_per_cell;

        typename dealii::DoFHandler<3>::active_cell_iterator
          cellChild = dofHandlerChild.begin_active(),
          endcChild = dofHandlerChild.end();
        // iterate through child cells
        for (; cellChild != endcChild; cellChild++)
          {
            fe_valuesChild.reinit(cellChild);
            dealii::Point<3, double> cellCenter = cellChild->center();

            //std::cout << " cell center = (" << cellCenter[0] << ","
            //          << cellCenter[1] << "," << cellCenter[2] << ")\n";

            dealii::Point<3> cellCenterRef =
              mapping.transform_real_to_unit_cell(cellParent, cellCenter);

            //std::cout << " cell center param = (" << cellCenterRef[0] << ","
            //          << cellCenterRef[1] << "," << cellCenterRef[2] << ")\n";

            unsigned int childCellIndex =
              getChildCellIndexAtLevel(cellCenterRef, iLevel);

            //std::cout << "cell index = " << childCellIndex << "\n";

            std::vector<double> cellWiseShapeVal;
            cellWiseShapeVal.resize(numberDofsPerElementParent *
                                    numChildQuadPoints);
            // check the child cell Id
            for (unsigned int iQuad = 0; iQuad < numChildQuadPoints; iQuad++)
              {
                dealii::Point<3, double> qPointVal =
                  fe_valuesChild.quadrature_point(iQuad);

                dealii::Point<3> qPointRef =
                  mapping.transform_real_to_unit_cell(cellParent, qPointVal);

                for (unsigned int iNode = 0; iNode < numberDofsPerElementParent;
                     iNode++)
                  {
                    cellWiseShapeVal[iNode +
                                     iQuad * numberDofsPerElementParent] =
                      feParent.shape_value(iNode, qPointRef);
                  }
              }
            childCellQuadDataShapeVal[iLevel][childCellIndex] =
              cellWiseShapeVal;

            for (unsigned int iQuad = 0; iQuad < numParentQuadPoints; iQuad++)
              {
                // check if the quad point is in the child cell
                // if so then get the shape function value and push it to index
                // as well if not do nothing

                //                if (!quadPointsTouched[iQuad])
                //                  {
                dealii::Point<3, double> qPointVal =
                  fe_valuesParent.quadrature_point(iQuad);

                try
                  {
                    dealii::Point<3> qpointRef =
                      mapping.transform_real_to_unit_cell(cellChild, qPointVal);
                    bool x_coord = false, y_coord = false, z_coord = false;
                    if ((qpointRef[0] > -1e-7) && (qpointRef[0] < 1 + 1e-7))
                      {
                        x_coord = true;
                      }
                    if ((qpointRef[1] > -1e-7) && (qpointRef[1] < 1 + 1e-7))
                      {
                        y_coord = true;
                      }
                    if ((qpointRef[2] > -1e-7) && (qpointRef[2] < 1 + 1e-7))
                      {
                        z_coord = true;
                      }
                    if (x_coord && y_coord && z_coord)
                      {
                        parentCellQuadDataIndex[iLevel][childCellIndex]
                          .push_back(iQuad);
                        for (unsigned int iNode = 0;
                             iNode < numberDofsPerElementChild;
                             iNode++)
                          {
                            parentCellQuadDataShapeVal[iLevel][childCellIndex]
                              .push_back(feChild.shape_value(iNode, qpointRef));
                          }
                        //                        quadPointsTouched[iQuad] = true;
                      }
                  }
                catch (...)
                  {}
                //                  }
              }
          }
        childTria.refine_global();
      }
  }

  TransferDataBetweenMeshesCompatiblePartitioning::~TransferDataBetweenMeshesCompatiblePartitioning()
  {}

  void
  TransferDataBetweenMeshesCompatiblePartitioning::computeShapeFuncValuesOnParentCell(
    std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
      &childCellQuadDataShapeVal)
  {
    const unsigned int inc = 1;

    dealii::MappingQGeneric<3, 3> mapping(1);

    dealii::FEValues<3> fe_valuesParent(d_dofHandlerParent->get_fe(),
                                        d_quadratureParent,
                                        dealii::update_values |
                                          dealii::update_quadrature_points);

    dealii::FEValues<3> fe_valuesChild(d_dofHandlerChild->get_fe(),
                                       d_quadratureChild,
                                       dealii::update_values |
                                         dealii::update_quadrature_points);
    d_mapParentShapeFuncMemLocation.resize(d_totallyOwnedCellsParent);

    if (d_totallyOwnedCellsParent > 0)
      d_mapParentShapeFuncMemLocation[0] = 0;

    unsigned int numberDofsPerElementParent =
      d_dofHandlerParent->get_fe().dofs_per_cell;

    for (unsigned int iElem = 1; iElem < d_totallyOwnedCellsParent; iElem++)
      {
        d_mapParentShapeFuncMemLocation[iElem] =
          d_mapParentShapeFuncMemLocation[iElem - 1] +
          (d_mapParentCellsToChild[iElem - 1].size() *
           d_numberQuadraturePointsChild * numberDofsPerElementParent);
      }



    d_shapeValueParentCells.resize(numberDofsPerElementParent *
                                   d_numberQuadraturePointsChild *
                                   d_totallyOwnedCellsChild);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellParent = d_dofHandlerParent->begin_active(),
      endcParent = d_dofHandlerParent->end();

    typename dealii::DoFHandler<3>::active_cell_iterator cellChildBegin =
      d_dofHandlerChild->begin_active();

    const dealii::FiniteElement<3> &fe = d_dofHandlerParent->get_fe();


    std::vector<double> cellLevelShapeFuncValue(numberDofsPerElementParent *
                                                d_numberQuadraturePointsChild);
    unsigned int        sizeToTransfer =
      numberDofsPerElementParent * d_numberQuadraturePointsChild;
    unsigned int cellIndexParent = 0;

    std::vector<bool> allQuadPointsTouched(d_totallyOwnedCellsChild *
                                             d_numberQuadraturePointsChild,
                                           false);

    d_mapChildCellToParentShapeFuncIndex.resize(d_totallyOwnedCellsChild, 0);
    for (; cellParent != endcParent; ++cellParent)
      {
        if (cellParent->is_locally_owned())
          {
            fe_valuesParent.reinit(cellParent);
            unsigned int cellIndexChild = 0;
            unsigned int numChildCells =
              d_mapParentCellsToChild[cellIndexParent].size();
            for (unsigned int childIndex = 0; childIndex < numChildCells;
                 childIndex++)
              {
                cellIndexChild =
                  d_mapParentCellsToChild[cellIndexParent][childIndex];
                //                typename
                //                dealii::DoFHandler<3>::active_cell_iterator
                //                  cellChild = cellChildBegin +cellIndexChild;

                typename dealii::DoFHandler<3>::active_cell_iterator cellChild =
                  d_mapParentCellToChildCellsIter[cellIndexParent][childIndex];
                fe_valuesChild.reinit(cellChild);

                unsigned int refinementLevel =
                  cellChild->level() - cellParent->level();

                dealii::Point<3, double> cellCenter = cellChild->center();

                dealii::Point<3> cellCenterRef =
                  mapping.transform_real_to_unit_cell(cellParent, cellCenter);

                unsigned int childCellIndex =
                  getChildCellIndexAtLevel(cellCenterRef, refinementLevel);

                for (unsigned int qPoint = 0;
                     qPoint < d_numberQuadraturePointsChild;
                     qPoint++)
                  {
                    for (unsigned int iNode = 0;
                         iNode < numberDofsPerElementParent;
                         iNode++)
                      {
                        cellLevelShapeFuncValue[iNode +
                                                qPoint *
                                                  numberDofsPerElementParent] =
                          childCellQuadDataShapeVal
                            [refinementLevel][childCellIndex]
                            [qPoint * numberDofsPerElementParent + iNode];
                      }

                    //                    dealii::Point<3, double> qPointVal =
                    //                      fe_valuesChild.quadrature_point(qPoint);
                    //
                    //                    dealii::Point<3> qPointRef =
                    //                      mapping.transform_real_to_unit_cell(cellParent,
                    //                                                          qPointVal);
                    //
                    //                    for (unsigned int iNode = 0;
                    //                         iNode <
                    //                         numberDofsPerElementParent;
                    //                         iNode++)
                    //                      {
                    //                        double feShapeVal =
                    //                        fe.shape_value(iNode, qPointRef);
                    //                        cellLevelShapeFuncValue[iNode +
                    //                                                qPoint *
                    //                                                  numberDofsPerElementParent]
                    //                                                  =
                    //                          feShapeVal;
                    //                        if (std::abs(
                    //                              feShapeVal -
                    //                              childCellQuadDataShapeVal
                    //                                [refinementLevel][childCellIndex]
                    //                                [qPoint *
                    //                                numberDofsPerElementParent
                    //                                + iNode]) >
                    //                            1e-5)
                    //                          {
                    //                            std::cout << " Error in child
                    //                            quad points \n";
                    //                          }
                    //                      }
                    allQuadPointsTouched[cellIndexChild *
                                           d_numberQuadraturePointsChild +
                                         qPoint] = true;
                    //
                    //
                    //                    // TODO get the child cell index
                    //
                    //                    //                      for( unsigned
                    //                    int iNode = 0;
                    //                    //
                    //                    iNode<numberDofsPerElementParent;iNode++)
                    //                    //                      {
                    //                    // cellLevelShapeFuncValue[iNode +
                    //                    // qPoint*numberDofsPerElementParent]
                    //                    //                          =
                    //                    //
                    //                    childCellQuadDataShapeVal[refinementLevel][childCellIndex][qPoint*numberDofsPerElementParent
                    //                    //                                  +
                    //                    iNode];
                    //                    //                      }
                  }
                dcopy_(&sizeToTransfer,
                       &cellLevelShapeFuncValue[0],
                       &inc,
                       &d_shapeValueParentCells
                         [d_mapParentShapeFuncMemLocation[cellIndexParent] +
                          childIndex * sizeToTransfer],
                       &inc);
                d_mapChildCellToParentShapeFuncIndex[cellIndexChild] =
                  d_mapParentShapeFuncMemLocation[cellIndexParent] +
                  childIndex * sizeToTransfer;
              }

            //            typename dealii::DoFHandler<3>::active_cell_iterator
            //              cellChild    = d_dofHandlerChild->begin_active(),
            //              endcChild    = d_dofHandlerChild->end();
            //            unsigned int cellIndexChild = 0 ;
            //            for( ; cellChild!= endcChild; ++cellChild)
            //              {
            //                if (cellChild->is_locally_owned())
            //                  {
            //                    for( unsigned int cellIndex = 0; cellIndex <
            //                    d_mapParentCellsToChild[cellIndexParent].size();
            //                    cellIndex++)
            //                      {
            //                        if(d_mapParentCellsToChild[cellIndexParent][cellIndex]
            //                        == cellIndexChild)
            //                          {
            //                            fe_valuesChild.reinit(cellChild);
            //                            for (unsigned int qPoint = 0; qPoint <
            //                            d_numberQuadraturePointsChild;
            //                            qPoint++)
            //                              {
            //                                dealii::Point<3, double> qPointVal
            //                                =
            //                                  fe_valuesChild.quadrature_point(qPoint);
            //
            //                                dealii::Point<3> qPointRef =
            //                                mapping.transform_real_to_unit_cell(cellParent,qPointVal);
            //
            //                                for( unsigned int iNode = 0;
            //                                iNode<numberDofsPerElementParent;iNode++)
            //                                  {
            //                                    cellLevelShapeFuncValue[iNode
            //                                    +
            //                                    qPoint*numberDofsPerElementParent]
            //                                    =
            //                                      fe.shape_value(iNode,qPointRef);
            //                                  }
            //                                allQuadPointsTouched[cellIndexChild*d_numberQuadraturePointsChild
            //                                + qPoint] = true;
            //                              }
            //                            dcopy_(&sizeToTransfer,
            //                                   &cellLevelShapeFuncValue[0],
            //                                   &inc,
            //                                   &d_shapeValueParentCells[
            //                                     d_mapParentShapeFuncMemLocation[cellIndexParent]
            //                                     + cellIndex*sizeToTransfer],
            //                                   &inc);
            //                              d_mapChildCellToParentShapeFuncIndex[cellIndexChild]
            //                              =
            //                              d_mapParentShapeFuncMemLocation[cellIndexParent]
            //                                                                                + cellIndex*sizeToTransfer;
            //
            //                          }
            //                      }
            //
            //                    cellIndexChild++;
            //                  }
            //              }
            cellIndexParent++;
          }
      }

    bool quadPointsTestPass = true;
    for (unsigned int iQuad = 0; iQuad < allQuadPointsTouched.size(); iQuad++)
      {
        if (allQuadPointsTouched[iQuad] == false)
          {
            quadPointsTestPass = false;
          }
      }
    if (!quadPointsTestPass)
      {
        std::cout
          << " Error :::::: Not all quad points are mapped in child cell \n";
      }
    else
      {
        std::cout << " All quad points are mapped in child cell \n ";
      }

#ifdef DFTFE_WITH_DEVICE
    if (d_useDevice)
      {
        d_mapChildCellToParentShapeFuncIndexDevice.resize(
          d_mapChildCellToParentShapeFuncIndex.size());
        d_mapChildCellToParentShapeFuncIndexDevice.copyFrom(
          d_mapChildCellToParentShapeFuncIndex);

        d_mapChildCellsToParentDevice.resize(d_mapChildCellsToParent.size());
        d_mapChildCellsToParentDevice.copyFrom(d_mapChildCellsToParent);

        d_shapeValueParentCellsDevice.resize(d_shapeValueParentCells.size());
        d_shapeValueParentCellsDevice.copyFrom(d_shapeValueParentCells);
      }

#endif
  }

  void
  TransferDataBetweenMeshesCompatiblePartitioning::computeShapeFuncValuesOnChildCell(
    std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>>>
      &parentCellQuadDataIndex,
    std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
      &parentCellQuadDataShapeVal)
  {
    dealii::MappingQGeneric<3, 3> mapping(1);

    dealii::FEValues<3> fe_valuesParent(d_dofHandlerParent->get_fe(),
                                        d_quadratureParent,
                                        dealii::update_values |
                                          dealii::update_quadrature_points);

    dealii::FEValues<3> fe_valuesChild(d_dofHandlerChild->get_fe(),
                                       d_quadratureChild,
                                       dealii::update_values |
                                         dealii::update_quadrature_points);

    const dealii::FiniteElement<3> &fe = d_dofHandlerChild->get_fe();

    const unsigned int inc = 1;

    unsigned int numberDofsPerElementChild =
      d_dofHandlerChild->get_fe().dofs_per_cell;
    listOfQuadPointsInChildCell.resize(d_totallyOwnedCellsChild);
    numberOfParentQuadPointsInChildCell.resize(d_totallyOwnedCellsChild, 0);
    std::vector<std::vector<double>> cellLevelShapeFuncValue(
      d_totallyOwnedCellsChild);
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellParent = d_dofHandlerParent->begin_active(),
      endcParent = d_dofHandlerParent->end();

    std::vector<bool> allQuadPointsTouched(d_totallyOwnedCellsParent *
                                             d_numberQuadraturePointsParent,
                                           false);

    unsigned int cellIndexParent = 0;
    for (; cellParent != endcParent; ++cellParent)
      {
        if (cellParent->is_locally_owned())
          {
            fe_valuesParent.reinit(cellParent);

            std::vector<bool> quadPointsTouched(d_numberQuadraturePointsParent,
                                                false);

            typename dealii::DoFHandler<3>::active_cell_iterator
              cellChildBegin = d_dofHandlerChild->begin_active();
            //              endcChild    = d_dofHandlerChild->end();

            unsigned int cellIndexChild = 0;
            unsigned int numChildCells =
              d_mapParentCellsToChild[cellIndexParent].size();
            for (unsigned int childIndex = 0; childIndex < numChildCells;
                 childIndex++)
              {
                cellIndexChild =
                  d_mapParentCellsToChild[cellIndexParent][childIndex];
                //                typename
                //                dealii::DoFHandler<3>::active_cell_iterator
                //                  cellChild = cellChildBegin + cellIndexChild;
                //                  //d_mapParentCellToChildCellsIter[cellIndexParent][childIndex];

                typename dealii::DoFHandler<3>::active_cell_iterator cellChild =
                  d_mapParentCellToChildCellsIter[cellIndexParent][childIndex];
                fe_valuesChild.reinit(cellChild);

                unsigned int refinementLevel =
                  cellChild->level() - cellParent->level();

                dealii::Point<3, double> cellCenter = cellChild->center();

                dealii::Point<3> cellCenterRef =
                  mapping.transform_real_to_unit_cell(cellParent, cellCenter);


                unsigned int childCellIndex =
                  getChildCellIndexAtLevel(cellCenterRef, refinementLevel);


                unsigned int numQuadPointsInChildCell =
                  parentCellQuadDataIndex[refinementLevel][childCellIndex]
                    .size();

                for (unsigned int quadPointIndex = 0;
                     quadPointIndex < numQuadPointsInChildCell;
                     quadPointIndex++)
                  {
                    unsigned int qPoint =
                      parentCellQuadDataIndex[refinementLevel][childCellIndex]
                                             [quadPointIndex];
                    if (!quadPointsTouched[qPoint])
                      {
                        listOfQuadPointsInChildCell[cellIndexChild].push_back(
                          qPoint);
                        //                        std::cout << " quad Id = " <<
                        //                        qPoint << " map index = "
                        //                                  <<
                        //                                  parentCellQuadDataIndex[refinementLevel]
                        //                                                            [childCellIndex]
                        //                                                            [quadPointIndex]
                        //                                  << "\n";
                        quadPointsTouched[qPoint] = true;
                        for (unsigned int iNode = 0;
                             iNode < numberDofsPerElementChild;
                             iNode++)
                          {
                            cellLevelShapeFuncValue[cellIndexChild].push_back(
                              parentCellQuadDataShapeVal
                                [refinementLevel][childCellIndex]
                                [quadPointIndex * numberDofsPerElementChild +
                                 iNode]);
                          }
                        numberOfParentQuadPointsInChildCell[cellIndexChild]++;
                        allQuadPointsTouched[cellIndexParent *
                                               d_numberQuadraturePointsParent +
                                             qPoint] = true;
                      }
                  }

                //                for (unsigned int qPoint = 0;
                //                     qPoint < d_numberQuadraturePointsParent;
                //                     qPoint++)
                //                  {
                //                    if (!quadPointsTouched[qPoint])
                //                      {
                //                        dealii::Point<3, double> qPointVal =
                //                          fe_valuesParent.quadrature_point(qPoint);
                //
                //                        try
                //                          {
                //                            dealii::Point<3> qpointRef =
                //                              mapping.transform_real_to_unit_cell(cellChild,
                //                                                                  qPointVal);
                //                            bool x_coord = false, y_coord =
                //                            false,
                //                                 z_coord = false;
                //                            if ((qpointRef[0] > -1e-7) &&
                //                                (qpointRef[0] < 1 + 1e-7))
                //                              {
                //                                x_coord = true;
                //                              }
                //                            if ((qpointRef[1] > -1e-7) &&
                //                                (qpointRef[1] < 1 + 1e-7))
                //                              {
                //                                y_coord = true;
                //                              }
                //                            if ((qpointRef[2] > -1e-7) &&
                //                                (qpointRef[2] < 1 + 1e-7))
                //                              {
                //                                z_coord = true;
                //                              }
                //                            if (x_coord && y_coord && z_coord)
                //                              {
                //                                listOfQuadPointsInChildCell[cellIndexChild]
                //                                  .push_back(qPoint);
                //                                std::cout
                //                                  << " quad Id = " << qPoint
                //                                  << " map index = "
                //                                  <<
                //                                  parentCellQuadDataIndex[refinementLevel]
                //                                                            [childCellIndex]
                //                                                            [quadPointIndex]
                //                                  << "\n";
                //                                quadPointsTouched[qPoint] =
                //                                true; for (unsigned int iNode
                //                                = 0;
                //                                     iNode <
                //                                     numberDofsPerElementChild;
                //                                     iNode++)
                //                                  {
                //                                    double feShapeVal =
                //                                      fe.shape_value(iNode,
                //                                      qpointRef);
                //                                    cellLevelShapeFuncValue[cellIndexChild]
                //                                      .push_back(feShapeVal);
                //                                    if (std::abs(
                //                                          feShapeVal -
                //                                          parentCellQuadDataShapeVal
                //                                            [refinementLevel][childCellIndex]
                //                                            [quadPointIndex *
                //                                               numberDofsPerElementChild
                //                                               +
                //                                             iNode]) > 1e-5)
                //                                      {
                //                                        std::cout
                //                                          << " Error in parent
                //                                          quad points \n";
                //                                      }
                //                                  }
                //                                numberOfParentQuadPointsInChildCell
                //                                  [cellIndexChild]++;
                //                                allQuadPointsTouched
                //                                  [cellIndexParent *
                //                                     d_numberQuadraturePointsParent
                //                                     +
                //                                   qPoint] = true;
                //
                //                                quadPointIndex++;
                //                              }
                //                          }
                //                        catch (...)
                //                          {}
                //                      }
                //                  }
              }

            //            typename dealii::DoFHandler<3>::active_cell_iterator
            //              cellChild    = d_dofHandlerChild->begin_active(),
            //              endcChild    = d_dofHandlerChild->end();
            //
            //            unsigned int cellIndexChild = 0 ;
            //            for( ; cellChild!= endcChild; ++cellChild)
            //              {
            //                if (cellChild->is_locally_owned())
            //                  {
            //                    for( unsigned int cellIndex = 0; cellIndex <
            //                    d_mapParentCellsToChild[cellIndexParent].size();
            //                    cellIndex++)
            //                      {
            //                        if(d_mapParentCellsToChild[cellIndexParent][cellIndex]
            //                        == cellIndexChild)
            //                          {
            //                            fe_valuesChild.reinit(cellChild);
            //                            for (unsigned int qPoint = 0; qPoint <
            //                            d_numberQuadraturePointsParent;
            //                            qPoint++)
            //                              {
            //                                if(!quadPointsTouched[qPoint])
            //                                  {
            //                                    dealii::Point<3, double>
            //                                    qPointVal =
            //                                      fe_valuesParent.quadrature_point(qPoint);
            //
            //                                    try
            //                                      {
            //                                        dealii::Point<3> qpointRef
            //                                        =
            //                                        mapping.transform_real_to_unit_cell(cellChild,qPointVal);
            //                                        bool x_coord = false,
            //                                        y_coord = false, z_coord =
            //                                        false; if ((qpointRef[0] >
            //                                        -1e-7) && (qpointRef[0] <
            //                                        1+ 1e-7))
            //                                          {
            //                                            x_coord = true;
            //                                          }
            //                                        if ((qpointRef[1] > -1e-7)
            //                                        && (qpointRef[1] < 1+
            //                                        1e-7))
            //                                          {
            //                                            y_coord = true;
            //                                          }
            //                                        if ((qpointRef[2] > -1e-7)
            //                                        && (qpointRef[2] < 1+
            //                                        1e-7))
            //                                          {
            //                                            z_coord = true;
            //                                          }
            //                                        if (x_coord && y_coord &&
            //                                        z_coord)
            //                                          {
            //                                            listOfQuadPointsInChildCell[cellIndexChild].push_back(qPoint);
            //                                            quadPointsTouched[qPoint]
            //                                            = true; for( unsigned
            //                                            int iNode = 0;
            //                                            iNode<numberDofsPerElementChild;iNode++)
            //                                              {
            //                                                cellLevelShapeFuncValue[cellIndexChild].push_back(fe.shape_value(iNode,qpointRef));
            //                                              }
            //                                            numberOfParentQuadPointsInChildCell[cellIndexChild]++;
            //                                            allQuadPointsTouched[
            //                                            cellIndexParent*
            //                                            d_numberQuadraturePointsParent+qPoint]
            //                                            = true;
            //
            //                                          }
            //                                      }catch(...){}
            //
            //                                  }
            //                              }
            //                          }
            //
            //                      }
            //                    cellIndexChild++;
            //                  }
            //              }
            cellIndexParent++;
          }
      }

    d_shapeValueChildCells.resize(d_totallyOwnedCellsParent *
                                  d_numberQuadraturePointsParent *
                                  numberDofsPerElementChild);

    d_mapChildShapeFuncMemLocation.resize(d_totallyOwnedCellsChild);

    d_maxNumberQuadPointsInChildCell = 0;
    if (d_totallyOwnedCellsChild > 0)
      {
        d_maxNumberQuadPointsInChildCell =
          numberOfParentQuadPointsInChildCell[0];
        d_mapChildShapeFuncMemLocation[0] = 0;
      }

    for (unsigned int iCellChild = 1; iCellChild < d_totallyOwnedCellsChild;
         iCellChild++)
      {
        if (d_maxNumberQuadPointsInChildCell <
            numberOfParentQuadPointsInChildCell[iCellChild])
          {
            d_maxNumberQuadPointsInChildCell =
              numberOfParentQuadPointsInChildCell[iCellChild];
          }
        d_mapChildShapeFuncMemLocation[iCellChild] =
          d_mapChildShapeFuncMemLocation[iCellChild - 1] +
          (numberOfParentQuadPointsInChildCell[iCellChild - 1] *
           numberDofsPerElementChild);
      }
    d_listOfChildCellsWithQuadPoint.resize(0);
    for (unsigned int iCellChild = 0; iCellChild < d_totallyOwnedCellsChild;
         iCellChild++)
      {
        if (numberOfParentQuadPointsInChildCell[iCellChild] > 0)
          {
            d_listOfChildCellsWithQuadPoint.push_back(iCellChild);
          }
        unsigned int sizeTransfer =
          numberOfParentQuadPointsInChildCell[iCellChild] *
          numberDofsPerElementChild;
        dcopy_(
          &sizeTransfer,
          &cellLevelShapeFuncValue[iCellChild][0],
          &inc,
          &d_shapeValueChildCells[d_mapChildShapeFuncMemLocation[iCellChild]],
          &inc);
      }

    // setting up the shape functions for strided gemm
    //    d_shapeFunctionValuesChildToParentStrided.resize(d_totallyOwnedCellsChild*numberDofsPerElementChild*d_maxNumberQuadPointsInChildCell);
    //    std::fill(d_shapeFunctionValuesChildToParentStrided.begin(),d_shapeFunctionValuesChildToParentStrided.end(),0.0);
    //    for (unsigned int iCellChild = 0 ; iCellChild <
    //    d_totallyOwnedCellsChild; iCellChild++)
    //      {
    //        unsigned int sizeTransfer =
    //        numberOfParentQuadPointsInChildCell[iCellChild]*numberDofsPerElementChild;
    //        dcopy_(&sizeTransfer,
    //               &cellLevelShapeFuncValue[iCellChild][0],
    //               &inc,
    //               &d_shapeFunctionValuesChildToParentStrided[iCellChild*d_maxNumberQuadPointsInChildCell*numberDofsPerElementChild],
    //               &inc);
    //
    //      }

    //std::cout << " max quad pt = " << d_maxNumberQuadPointsInChildCell
    //          << " num child = " << d_totallyOwnedCellsChild << "\n";

    // check if all the quad points are touched

    bool quadPointsTestPass = true;
    for (unsigned int iQuad = 0; iQuad < allQuadPointsTouched.size(); iQuad++)
      {
        if (allQuadPointsTouched[iQuad] == false)
          {
            quadPointsTestPass = false;
          }
      }
    if (!quadPointsTestPass)
      {
        std::cout
          << " Error :::::: Not all quad points are mapped in parent cell \n";
      }
    else
      {
        std::cout << " All quad points are mapped in parent cell \n ";
      }
  }

  void
  TransferDataBetweenMeshesCompatiblePartitioning::interpolateMesh1DataToMesh2QuadPoints(
    const distributedCPUMultiVec<double> &inputVec,
    const unsigned int                    numberOfVectors,
    const std::vector<dealii::types::global_dof_index>
                                                                 &                  fullFlattenedArrayCellLocalProcIndexIdMapParent,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &outputQuadData,
    bool resizeOutputVec)
  {
    if(resizeOutputVec)
      {
        outputQuadData.resize(d_numberQuadraturePointsChild*d_totallyOwnedCellsChild*numberOfVectors);
      }
    outputQuadData.setValue(0.0);
    //    if (d_numInputVectorsForInterpolationFromParentNodesToChildQuad !=
    //    numberOfVectors)
    //      {
    //        d_numInputVectorsForInterpolationFromParentNodesToChildQuad =
    //        numberOfVectors; vectorTools::computeCellLocalIndexSetMap(
    //          inputVec.getMPIPatternP2P(),
    //          *d_matrixFreeDataParentPtr,
    //          d_matrixFreeParentVectorComponent,
    //          numberOfVectors,
    //          d_fullFlattenedArrayMacroCellLocalProcIndexIdMapParent,
    //          d_normalCellIdToMacroCellIdMapParent,
    //          d_macroCellIdToNormalCellIdMapParent,
    //          d_fullFlattenedArrayCellLocalProcIndexIdMapParent);
    //      }


    const double       scalarCoeffAlpha = 1.0;
    const double       scalarCoeffBeta  = 0.0;
    const char         transA = 'N', transB = 'N';
    const unsigned int inc = 1;


    unsigned int iElemParent = 0;

    unsigned int numberDofsPerElementParent =
      d_dofHandlerParent->get_fe().dofs_per_cell;

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerParent->begin_active(),
      endc = d_dofHandlerParent->end();

    std::vector<double> cellLevelInputVec(numberDofsPerElementParent *
                                            numberOfVectors,
                                          0.0);


    std::vector<double> cellLevelInputVecQuad;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            unsigned int numberOfQuadPointsInParentCell =
              d_mapParentCellsToChild[iElemParent].size() *
              d_numberQuadraturePointsChild;

            cellLevelInputVecQuad.resize(numberOfQuadPointsInParentCell *
                                         numberOfVectors);

            for (unsigned int iNode = 0; iNode < numberDofsPerElementParent;
                 iNode++)
              {
                dcopy_(&numberOfVectors,
                       inputVec.data() +
                         fullFlattenedArrayCellLocalProcIndexIdMapParent
                           [iElemParent * numberDofsPerElementParent + iNode],
                       &inc,
                       &cellLevelInputVec[numberOfVectors * iNode],
                       &inc);
              }

            xgemm(&transA,
                  &transB,
                  &numberOfVectors,
                  &numberOfQuadPointsInParentCell,
                  &numberDofsPerElementParent,
                  &scalarCoeffAlpha,
                  &cellLevelInputVec[0],
                  &numberOfVectors,
                  &d_shapeValueParentCells
                    [d_mapParentShapeFuncMemLocation[iElemParent]],
                  &numberDofsPerElementParent,
                  &scalarCoeffBeta,
                  &cellLevelInputVecQuad[0],
                  &numberOfVectors);

            unsigned int numberQuadPointsTimesVectors =
              d_numberQuadraturePointsChild * numberOfVectors;
            for (unsigned int childCellIter = 0;
                 childCellIter < d_mapParentCellsToChild[iElemParent].size();
                 childCellIter++)
              {
                unsigned int childCellIndex =
                  d_mapParentCellsToChild[iElemParent][childCellIter];
                dcopy_(&numberQuadPointsTimesVectors,
                       &cellLevelInputVecQuad[childCellIter *
                                              numberQuadPointsTimesVectors],
                       &inc,
                       &outputQuadData[childCellIndex *
                                       numberQuadPointsTimesVectors],
                       &inc);
              }
            iElemParent++;
          }
      }
  }

  void
  TransferDataBetweenMeshesCompatiblePartitioning::interpolateMesh2DataToMesh1QuadPoints(
    const distributedCPUMultiVec<double> &inputVec,
    const unsigned int                    numberOfVectors,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &                 outputQuadData,
    bool resizeOutputVec)
  {
    if(resizeOutputVec)
      {
        outputQuadData.resize(d_numberQuadraturePointsParent*d_totallyOwnedCellsParent*numberOfVectors);
      }
    outputQuadData.setValue(0.0);
    double startMap = MPI_Wtime();
    if (d_numInputVectorsForInterpolationFromChildNodesToParentQuad !=
        numberOfVectors)
      {
        d_numInputVectorsForInterpolationFromChildNodesToParentQuad =
          numberOfVectors;
        vectorTools::computeCellLocalIndexSetMap(
          inputVec.getMPIPatternP2P(),
          *d_matrixFreeDataChildPtr,
          d_matrixFreeChildVectorComponent,
          numberOfVectors,
          d_fullFlattenedArrayCellLocalProcIndexIdMapChild);

        //        vectorTools::computeCellLocalIndexSetMap(
        //          inputVec.getMPIPatternP2P(),
        //          *d_matrixFreeDataChildPtr,
        //	  d_matrixFreeChildVectorComponent,
        //          numberOfVectors,
        //          d_fullFlattenedArrayMacroCellLocalProcIndexIdMapChild,
        //          d_normalCellIdToMacroCellIdMapChild,
        //          d_macroCellIdToNormalCellIdMapChild,
        //          d_fullFlattenedArrayCellLocalProcIndexIdMapChild);
      }

    double endMap = MPI_Wtime();

    double timeForMapCreation = endMap - startMap;

    const double       scalarCoeffAlpha = 1.0;
    const double       scalarCoeffBeta  = 0.0;
    const char         transA = 'N', transB = 'N';
    const unsigned int inc = 1;

    unsigned int iElemChild = 0;

    unsigned int numberDofsPerElementChild =
      d_dofHandlerChild->get_fe().dofs_per_cell;


    std::vector<double> cellLevelInputVec(numberDofsPerElementChild *
                                            numberOfVectors,
                                          0.0);

    std::vector<double> cellLevelInputVecQuad;
    double              timeCopyNodes = 0.0, timeGemm = 0.0, timeCopyQuad = 0.0;
    unsigned int        numChildCells = d_listOfChildCellsWithQuadPoint.size();

    for (unsigned int iCell = 0; iCell < numChildCells; iCell++)
      {
        double startCopyTime = MPI_Wtime();
        iElemChild           = d_listOfChildCellsWithQuadPoint[iCell];
        unsigned int numberOfQuadPointsInChildCell =
          numberOfParentQuadPointsInChildCell[iElemChild];

        cellLevelInputVecQuad.resize(numberOfQuadPointsInChildCell *
                                     numberOfVectors);


        for (unsigned int iNode = 0; iNode < numberDofsPerElementChild; iNode++)
          {
            dcopy_(&numberOfVectors,
                   inputVec.data() +
                     d_fullFlattenedArrayCellLocalProcIndexIdMapChild
                       [iElemChild * numberDofsPerElementChild + iNode],
                   &inc,
                   &cellLevelInputVec[numberOfVectors * iNode],
                   &inc);
          }
        double endCopyTime = MPI_Wtime();

        xgemm(
          &transA,
          &transB,
          &numberOfVectors,
          &numberOfQuadPointsInChildCell,
          &numberDofsPerElementChild,
          &scalarCoeffAlpha,
          &cellLevelInputVec[0],
          &numberOfVectors,
          &d_shapeValueChildCells[d_mapChildShapeFuncMemLocation[iElemChild]],
          &numberDofsPerElementChild,
          &scalarCoeffBeta,
          &cellLevelInputVecQuad[0],
          &numberOfVectors);

        double endGemmTime = MPI_Wtime();

        for (unsigned int quadIndex = 0;
             quadIndex < numberOfQuadPointsInChildCell;
             quadIndex++)
          {
            dcopy_(&numberOfVectors,
                   &cellLevelInputVecQuad[quadIndex * numberOfVectors],
                   &inc,
                   &outputQuadData[(d_mapChildCellsToParent[iElemChild] *
                                      d_numberQuadraturePointsParent +
                                    listOfQuadPointsInChildCell[iElemChild]
                                                               [quadIndex]) *
                                   numberOfVectors],
                   &inc);
          }
        double endCopyQuadTime = MPI_Wtime();
        timeCopyNodes += (endCopyTime - startCopyTime);
        timeGemm += (endGemmTime - endCopyTime);
        timeCopyQuad += (endCopyQuadTime - endGemmTime);
      }


    std::cout << " Time for Map = " << timeForMapCreation
              << " nodes = " << timeCopyNodes << " gemm = " << timeGemm
              << " quad = " << timeCopyQuad << "\n";


    //   typename dealii::DoFHandler<3>::active_cell_iterator
    //     cell    = d_dofHandlerChild->begin_active(),
    //     endc    = d_dofHandlerChild->end();
    //   for( ; cell!= endc; ++cell)
    //     {
    //       if (cell->is_locally_owned())
    //         {
    //
    //           unsigned int numberOfQuadPointsInChildCell =
    //      		   numberOfParentQuadPointsInChildCell[iElemChild];
    //
    //           cellLevelInputVecQuad.resize(numberOfQuadPointsInChildCell
    //                                        *numberOfVectors);
    //
    //           for (unsigned int iNode = 0; iNode < numberDofsPerElementChild;
    //           iNode++)
    //             {
    //               dcopy_(&numberOfVectors,
    //                      inputVec.begin() +
    //                        d_fullFlattenedArrayCellLocalProcIndexIdMapChild
    //                          [iElemChild * numberDofsPerElementChild +
    //                          iNode],
    //                      &inc,
    //                      &cellLevelInputVec[numberOfVectors * iNode],
    //                      &inc);
    //             }
    //
    //           xgemm(&transA,
    //                 &transB,
    //                &numberOfVectors,
    //                 &numberOfQuadPointsInChildCell,
    //                 &numberDofsPerElementChild,
    //                 &scalarCoeffAlpha,
    //                 &cellLevelInputVec[0],
    //                 &numberOfVectors,
    //                 &d_shapeValueChildCells[d_mapChildShapeFuncMemLocation[iElemChild]],
    //                 &numberDofsPerElementChild,
    //                 &scalarCoeffBeta,
    //                 &cellLevelInputVecQuad[0],
    //                 &numberOfVectors);
    //
    //           for (unsigned int quadIndex = 0 ;quadIndex
    //           <numberOfQuadPointsInChildCell; quadIndex++)
    //             {
    //               dcopy_(&numberOfVectors,
    //                      &cellLevelInputVecQuad[quadIndex*numberOfVectors],
    //                      &inc,
    //                      &outputQuadData[
    //                        (d_mapChildCellsToParent[iElemChild]*
    //                          d_numberQuadraturePointsParent+
    //                        listOfQuadPointsInChildCell[iElemChild][quadIndex])
    //                        *numberOfVectors],
    //                      &inc);
    //             }
    //
    //           iElemChild++;
    //         }
    //     }
  }
#ifdef DFTFE_WITH_DEVICE
  void
  TransferDataBetweenMeshesCompatiblePartitioning::interpolateMesh1DataToMesh2QuadPoints(
    const distributedDeviceVec<double> &inputVec,
    const unsigned int                  numberOfVectors,
    const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                      dftfe::utils::MemorySpace::DEVICE>
      &flattenedArrayCellLocalProcIndexIdMapDevice,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
        &outputQuadData,
    bool resizeOutputVec)
  {
    if(resizeOutputVec)
      {
        outputQuadData.resize(d_numberQuadraturePointsChild*d_totallyOwnedCellsChild*numberOfVectors);
      }
    outputQuadData.setValue(0.0);
    unsigned int numberDofsPerElementParent =
      d_dofHandlerParent->get_fe().dofs_per_cell;
    d_cellLevelParentNodalDevice.resize(d_totallyOwnedCellsParent *
                                          numberDofsPerElementParent *
                                          numberOfVectors,
                                        0.0);
    dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
      numberOfVectors,
      d_totallyOwnedCellsParent * numberDofsPerElementParent,
      inputVec.data(),
      d_cellLevelParentNodalDevice.begin(),
      flattenedArrayCellLocalProcIndexIdMapDevice.begin());

    dftfe::utils::deviceKernelsGeneric::interpolateParentNodeToChildQuadDevice(
      d_totallyOwnedCellsChild,
      numberDofsPerElementParent,
      d_numberQuadraturePointsChild,
      numberOfVectors,
      d_mapChildCellToParentShapeFuncIndexDevice.data(),
      d_mapChildCellsToParentDevice.data(),
      d_shapeValueParentCellsDevice.data(),
      d_cellLevelParentNodalDevice.data(),
      outputQuadData.data());
  }
#endif

  void
  TransferDataBetweenMeshesCompatiblePartitioning::interpolateMesh1DataToMesh2QuadPoints(
    const distributedCPUVec<double> &inputVec,
    const unsigned int               numberOfVectors,
    const std::vector<dealii::types::global_dof_index>
                                                                 &                  fullFlattenedArrayCellLocalProcIndexIdMapParent,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &outputQuadData,
    bool resizeOutputVec)
  {
    if(resizeOutputVec)
      {
        outputQuadData.resize(d_numberQuadraturePointsChild*d_totallyOwnedCellsChild*numberOfVectors);
      }
    outputQuadData.setValue(0.0);
    //    if (d_numInputVectorsForInterpolationFromParentNodesToChildQuad !=
    //    numberOfVectors)
    //      {
    //        d_numInputVectorsForInterpolationFromParentNodesToChildQuad =
    //        numberOfVectors; vectorTools::computeCellLocalIndexSetMap(
    //          inputVec.getMPIPatternP2P(),
    //          *d_matrixFreeDataParentPtr,
    //          d_matrixFreeParentVectorComponent,
    //          numberOfVectors,
    //          d_fullFlattenedArrayMacroCellLocalProcIndexIdMapParent,
    //          d_normalCellIdToMacroCellIdMapParent,
    //          d_macroCellIdToNormalCellIdMapParent,
    //          d_fullFlattenedArrayCellLocalProcIndexIdMapParent);
    //      }


    const double       scalarCoeffAlpha = 1.0;
    const double       scalarCoeffBeta  = 0.0;
    const char         transA = 'N', transB = 'N';
    const unsigned int inc = 1;


    unsigned int iElemParent = 0;

    unsigned int numberDofsPerElementParent =
      d_dofHandlerParent->get_fe().dofs_per_cell;

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerParent->begin_active(),
      endc = d_dofHandlerParent->end();

    std::vector<double> cellLevelInputVec(numberDofsPerElementParent *
                                            numberOfVectors,
                                          0.0);


    std::vector<double> cellLevelInputVecQuad;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            unsigned int numberOfQuadPointsInParentCell =
              d_mapParentCellsToChild[iElemParent].size() *
              d_numberQuadraturePointsChild;

            cellLevelInputVecQuad.resize(numberOfQuadPointsInParentCell *
                                         numberOfVectors);

            for (unsigned int iNode = 0; iNode < numberDofsPerElementParent;
                 iNode++)
              {
                dcopy_(&numberOfVectors,
                       inputVec.begin() +
                         fullFlattenedArrayCellLocalProcIndexIdMapParent
                           [iElemParent * numberDofsPerElementParent + iNode],
                       &inc,
                       &cellLevelInputVec[numberOfVectors * iNode],
                       &inc);
              }

            xgemm(&transA,
                  &transB,
                  &numberOfVectors,
                  &numberOfQuadPointsInParentCell,
                  &numberDofsPerElementParent,
                  &scalarCoeffAlpha,
                  &cellLevelInputVec[0],
                  &numberOfVectors,
                  &d_shapeValueParentCells
                    [d_mapParentShapeFuncMemLocation[iElemParent]],
                  &numberDofsPerElementParent,
                  &scalarCoeffBeta,
                  &cellLevelInputVecQuad[0],
                  &numberOfVectors);

            unsigned int numberQuadPointsTimesVectors =
              d_numberQuadraturePointsChild * numberOfVectors;
            for (unsigned int childCellIter = 0;
                 childCellIter < d_mapParentCellsToChild[iElemParent].size();
                 childCellIter++)
              {
                unsigned int childCellIndex =
                  d_mapParentCellsToChild[iElemParent][childCellIter];
                dcopy_(&numberQuadPointsTimesVectors,
                       &cellLevelInputVecQuad[childCellIter *
                                              numberQuadPointsTimesVectors],
                       &inc,
                       &outputQuadData[childCellIndex *
                                       numberQuadPointsTimesVectors],
                       &inc);
              }
            iElemParent++;
          }
      }
  }

  void
  TransferDataBetweenMeshesCompatiblePartitioning::interpolateMesh2DataToMesh1QuadPoints(
    const distributedCPUVec<double> &inputVec,
    const unsigned int               numberOfVectors,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &            outputQuadData,
    bool resizeOutputVec)
  {
    if(resizeOutputVec)
      {
        outputQuadData.resize(d_numberQuadraturePointsParent*d_totallyOwnedCellsParent*numberOfVectors);
      }
    outputQuadData.setValue(0.0);
    double startMap = MPI_Wtime();
    if (d_numInputVectorsForInterpolationFromChildNodesToParentQuad !=
        numberOfVectors)
      {
        d_numInputVectorsForInterpolationFromChildNodesToParentQuad =
          numberOfVectors;
        vectorTools::computeCellLocalIndexSetMap(
          inputVec.get_partitioner(),
          *d_matrixFreeDataChildPtr,
          d_matrixFreeChildVectorComponent,
          numberOfVectors,
          d_fullFlattenedArrayCellLocalProcIndexIdMapChild);

        //        vectorTools::computeCellLocalIndexSetMap(
        //          inputVec.getMPIPatternP2P(),
        //          *d_matrixFreeDataChildPtr,
        //	  d_matrixFreeChildVectorComponent,
        //          numberOfVectors,
        //          d_fullFlattenedArrayMacroCellLocalProcIndexIdMapChild,
        //          d_normalCellIdToMacroCellIdMapChild,
        //          d_macroCellIdToNormalCellIdMapChild,
        //          d_fullFlattenedArrayCellLocalProcIndexIdMapChild);
      }

    double endMap = MPI_Wtime();

    double timeForMapCreation = endMap - startMap;

    const double       scalarCoeffAlpha = 1.0;
    const double       scalarCoeffBeta  = 0.0;
    const char         transA = 'N', transB = 'N';
    const unsigned int inc = 1;

    unsigned int iElemChild = 0;

    unsigned int numberDofsPerElementChild =
      d_dofHandlerChild->get_fe().dofs_per_cell;


    std::vector<double> cellLevelInputVec(numberDofsPerElementChild *
                                            numberOfVectors,
                                          0.0);

    std::vector<double> cellLevelInputVecQuad;
    double              timeCopyNodes = 0.0, timeGemm = 0.0, timeCopyQuad = 0.0;
    unsigned int        numChildCells = d_listOfChildCellsWithQuadPoint.size();

    for (unsigned int iCell = 0; iCell < numChildCells; iCell++)
      {
        double startCopyTime = MPI_Wtime();
        iElemChild           = d_listOfChildCellsWithQuadPoint[iCell];
        unsigned int numberOfQuadPointsInChildCell =
          numberOfParentQuadPointsInChildCell[iElemChild];

        cellLevelInputVecQuad.resize(numberOfQuadPointsInChildCell *
                                     numberOfVectors);


        for (unsigned int iNode = 0; iNode < numberDofsPerElementChild; iNode++)
          {
            dcopy_(&numberOfVectors,
                   inputVec.begin() +
                     d_fullFlattenedArrayCellLocalProcIndexIdMapChild
                       [iElemChild * numberDofsPerElementChild + iNode],
                   &inc,
                   &cellLevelInputVec[numberOfVectors * iNode],
                   &inc);
          }
        double endCopyTime = MPI_Wtime();

        xgemm(
          &transA,
          &transB,
          &numberOfVectors,
          &numberOfQuadPointsInChildCell,
          &numberDofsPerElementChild,
          &scalarCoeffAlpha,
          &cellLevelInputVec[0],
          &numberOfVectors,
          &d_shapeValueChildCells[d_mapChildShapeFuncMemLocation[iElemChild]],
          &numberDofsPerElementChild,
          &scalarCoeffBeta,
          &cellLevelInputVecQuad[0],
          &numberOfVectors);

        double endGemmTime = MPI_Wtime();

        for (unsigned int quadIndex = 0;
             quadIndex < numberOfQuadPointsInChildCell;
             quadIndex++)
          {
            dcopy_(&numberOfVectors,
                   &cellLevelInputVecQuad[quadIndex * numberOfVectors],
                   &inc,
                   &outputQuadData[(d_mapChildCellsToParent[iElemChild] *
                                      d_numberQuadraturePointsParent +
                                    listOfQuadPointsInChildCell[iElemChild]
                                                               [quadIndex]) *
                                   numberOfVectors],
                   &inc);
          }
        double endCopyQuadTime = MPI_Wtime();
        timeCopyNodes += (endCopyTime - startCopyTime);
        timeGemm += (endGemmTime - endCopyTime);
        timeCopyQuad += (endCopyQuadTime - endGemmTime);
      }


    //    std::cout<<" Time for Map = "<<timeForMapCreation<<" nodes =
    //    "<<timeCopyNodes<<" gemm = "<<timeGemm<<" quad =
    //    "<<timeCopyQuad<<"\n";


    //   typename dealii::DoFHandler<3>::active_cell_iterator
    //     cell    = d_dofHandlerChild->begin_active(),
    //     endc    = d_dofHandlerChild->end();
    //   for( ; cell!= endc; ++cell)
    //     {
    //       if (cell->is_locally_owned())
    //         {
    //
    //           unsigned int numberOfQuadPointsInChildCell =
    //      		   numberOfParentQuadPointsInChildCell[iElemChild];
    //
    //           cellLevelInputVecQuad.resize(numberOfQuadPointsInChildCell
    //                                        *numberOfVectors);
    //
    //           for (unsigned int iNode = 0; iNode < numberDofsPerElementChild;
    //           iNode++)
    //             {
    //               dcopy_(&numberOfVectors,
    //                      inputVec.begin() +
    //                        d_fullFlattenedArrayCellLocalProcIndexIdMapChild
    //                          [iElemChild * numberDofsPerElementChild +
    //                          iNode],
    //                      &inc,
    //                      &cellLevelInputVec[numberOfVectors * iNode],
    //                      &inc);
    //             }
    //
    //           xgemm(&transA,
    //                 &transB,
    //                &numberOfVectors,
    //                 &numberOfQuadPointsInChildCell,
    //                 &numberDofsPerElementChild,
    //                 &scalarCoeffAlpha,
    //                 &cellLevelInputVec[0],
    //                 &numberOfVectors,
    //                 &d_shapeValueChildCells[d_mapChildShapeFuncMemLocation[iElemChild]],
    //                 &numberDofsPerElementChild,
    //                 &scalarCoeffBeta,
    //                 &cellLevelInputVecQuad[0],
    //                 &numberOfVectors);
    //
    //           for (unsigned int quadIndex = 0 ;quadIndex
    //           <numberOfQuadPointsInChildCell; quadIndex++)
    //             {
    //               dcopy_(&numberOfVectors,
    //                      &cellLevelInputVecQuad[quadIndex*numberOfVectors],
    //                      &inc,
    //                      &outputQuadData[
    //                        (d_mapChildCellsToParent[iElemChild]*
    //                          d_numberQuadraturePointsParent+
    //                        listOfQuadPointsInChildCell[iElemChild][quadIndex])
    //                        *numberOfVectors],
    //                      &inc);
    //             }
    //
    //           iElemChild++;
    //         }
    //     }
  }

  /*
    void TransferDataBetweenMeshesCompatiblePartitioning::interpolateMesh2DataToMesh1QuadPoints(
      const distributedCPUMultiVec<double> &inputVec,
      const unsigned int numberOfVectors,
      dftfe::utils::MemoryStorage<dataTypes::number,
                            dftfe::utils::MemorySpace::HOST> &outputQuadData)
    {
      if( d_numInputVectorsForInterpolationFromChildNodesToParentQuad !=
    numberOfVectors)
        {
          d_numInputVectorsForInterpolationFromChildNodesToParentQuad =
    numberOfVectors; vectorTools::computeCellLocalIndexSetMap(
            inputVec.getMPIPatternP2P(),
            *d_matrixFreeDataChildPtr,
            d_matrixFreeChildVectorComponent,
            numberOfVectors,
            d_fullFlattenedArrayMacroCellLocalProcIndexIdMapChild,
            d_normalCellIdToMacroCellIdMapChild,
            d_macroCellIdToNormalCellIdMapChild,
            d_fullFlattenedArrayCellLocalProcIndexIdMapChild);
        }


      const double scalarCoeffAlpha = 1.0;
      const double scalarCoeffBeta  = 0.0;
      const char transA = 'N', transB = 'N';
      const unsigned int inc = 1;

      unsigned int iElemChild = 0;

      unsigned int numberDofsPerElementChild =
    d_dofHandlerChild->get_fe().dofs_per_cell ;

      typename dealii::DoFHandler<3>::active_cell_iterator
        cell    = d_dofHandlerChild->begin_active(),
        endc    = d_dofHandlerChild->end();

      std::vector<double> cellLevelInputVec(d_totallyOwnedCellsChild
                                              *numberDofsPerElementChild
                                              *numberOfVectors,0.0);


      std::vector<double> cellLevelInputVecQuad;
      cellLevelInputVecQuad.resize(d_totallyOwnedCellsChild
                                   *d_maxNumberQuadPointsInChildCell
                                   *numberOfVectors,0.0);
      iElemChild = 0;
      for(cell = d_dofHandlerChild->begin_active() ; cell!= endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              //            unsigned int numberOfQuadPointsInChildCell =
              //              numberOfParentQuadPointsInChildCell[iElemChild];

              for (unsigned int iNode = 0; iNode < numberDofsPerElementChild;
                   iNode++)
                {
                  dcopy_(
                    &numberOfVectors,
                    inputVec.begin() +
                      d_fullFlattenedArrayCellLocalProcIndexIdMapChild
                        [iElemChild * numberDofsPerElementChild + iNode],
                    &inc,
                    &cellLevelInputVec[iElemChild * numberDofsPerElementChild *
                                         numberOfVectors +
                                       numberOfVectors * iNode],
                    &inc);
                }
              iElemChild++;
            }
        }

      unsigned int stridea = numberDofsPerElementChild*numberOfVectors;
      unsigned int strideb =
    numberDofsPerElementChild*d_maxNumberQuadPointsInChildCell; unsigned int
    stridec = numberOfVectors*d_maxNumberQuadPointsInChildCell;
      dgemm_batch_strided_(&transA,
                          &transB,
                          &numberOfVectors,
                          &d_maxNumberQuadPointsInChildCell,
                          &numberDofsPerElementChild,
                          &scalarCoeffAlpha,
                          &cellLevelInputVec[0],
                          &numberOfVectors,
                          &stridea,
                          &d_shapeFunctionValuesChildToParentStrided[0],
                          &numberDofsPerElementChild,
                          &strideb,
                          &scalarCoeffBeta,
                          &cellLevelInputVecQuad[0],
                          &numberOfVectors,
                          &stridec,
                          &d_totallyOwnedCellsChild);

      iElemChild = 0;
      for(cell = d_dofHandlerChild->begin_active() ; cell!= endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              unsigned int numberOfQuadPointsInChildCell =
    numberOfParentQuadPointsInChildCell[iElemChild]; for (unsigned int quadIndex
    = 0 ;quadIndex <numberOfQuadPointsInChildCell; quadIndex++)
                {
                  dcopy_(&numberOfVectors,
                         &cellLevelInputVecQuad[iElemChild*d_maxNumberQuadPointsInChildCell*numberOfVectors
    + quadIndex*numberOfVectors], &inc, &outputQuadData[
                           (d_mapChildCellsToParent[iElemChild]*
                              d_numberQuadraturePointsParent+
                            listOfQuadPointsInChildCell[iElemChild][quadIndex])
                           *numberOfVectors],
                         &inc);
                }
              iElemChild++;
            }
        }

    }
  */

} // namespace dftfe
