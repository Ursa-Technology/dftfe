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

#ifndef DFTFE_INVERSEDFTDOFMANAGER_H
#define DFTFE_INVERSEDFTDOFMANAGER_H

#include "headers.h"
#include "linearAlgebraOperationsInternal.h"
#include "linearAlgebraOperations.h"
#include "vectorUtilities.h"
#include "triangulationManagerVxc.h"
#include "transferDataBetweenMeshesBase.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class TransferDataBetweenMeshesCompatiblePartitioning<memorySpace>  : public TransferDataBetweenMeshesBase
  {
  public:
    TransferDataBetweenMeshesCompatiblePartitioning(const dealii::MatrixFree<3, double> &matrixFreeParentData,
                                                    const unsigned int                   matrixFreeParentVectorComponent,
                                                    const unsigned int matrixFreeParentQuadratureComponent,
                                                    const dealii::MatrixFree<3, double> &matrixFreeChildData,
                                                    const unsigned int                   matrixFreeChildVectorComponent,
                                                    const unsigned int matrixFreeChildQuadratureComponent,
                                                    std::vector<std::vector<unsigned int>> &mapParentCellsToChild,
                                                    std::vector<
                                                      std::map<unsigned int,
                                                               typename dealii::DoFHandler<3>::active_cell_iterator>>
                                                                              &                        mapParentCellToChildCellsIter,
                                                    std::vector<unsigned int> &mapChildCellsToParent,
                                                    unsigned int               maxRelativeRefinement);

    // inputVec is a vector of appropriate size storing the values at the nodes
    // of the parent mesh outputQuadData stores the interpolated value at the
    // quad points of the child mesh
    void
    interpolateMesh1DataToMesh2QuadPoints(
      const distributedCPUMultiVec<double> &inputVec,
      const unsigned int                    numberOfVectors,
      const std::vector<dealii::types::global_dof_index>
                                                                   &                  fullFlattenedArrayCellLocalProcIndexIdMapParent,
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST> &outputQuadData,
      bool resizeOutputVec) override;

    void
    getShapeFuncValsForParametricCell(
      unsigned int maxRelativeRefinement,
      std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>>>
        &parentCellQuadDataIndex,
      std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
        &parentCellQuadDataShapeVal,
      std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
        &childCellQuadDataShapeVal);

#ifdef DFTFE_WITH_DEVICE
    void
    interpolateMesh1DataToMesh2QuadPoints(
      const distributedDeviceVec<double> &inputVec,
      const unsigned int                  numberOfVectors,
      const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                        dftfe::utils::MemorySpace::DEVICE>
        &flattenedArrayCellLocalProcIndexIdMapDevice,
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
          &outputQuadData,
      bool resizeOutputVec) override;
#endif


    // inputVec is a vector of appropriate size storing the values at the nodes
    // of the child mesh outputQuadData stores the interpolated value at the
    // quad points of the parent mesh
    void
    interpolateMesh2DataToMesh1QuadPoints(
      const distributedCPUMultiVec<double> &inputVec,
      const unsigned int                    numberOfVectors,
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST> &                 outputQuadData,
      bool resizeOutputVec) override;

    void
    interpolateMesh1DataToMesh2QuadPoints(
      const distributedCPUVec<double> &inputVec,
      const unsigned int               numberOfVectors,
      const std::vector<dealii::types::global_dof_index>
                                                                   &                  fullFlattenedArrayCellLocalProcIndexIdMapParent,
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST> &outputQuadData,
      bool resizeOutputVec) override;

    void
    interpolateMesh2DataToMesh1QuadPoints(
      const distributedCPUVec<double> &inputVec,
      const unsigned int               numberOfVectors,
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST> &            outputQuadData,
      bool resizeOutputVec) override;


    void
    computeShapeFuncValuesOnParentCell(
      std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
        &childCellQuadDataShapeVal);

    void
    computeShapeFuncValuesOnChildCell(
      std::map<unsigned int, std::map<unsigned int, std::vector<unsigned int>>>
        &parentCellQuadDataIndex,
      std::map<unsigned int, std::map<unsigned int, std::vector<double>>>
        &parentCellQuadDataShapeVal);

    ~TransferDataBetweenMeshesCompatiblePartitioning();

  private:
    const dealii::MatrixFree<3, double> *d_matrixFreeDataParentPtr;

    const dealii::DoFHandler<3> *d_dofHandlerParent;

    unsigned int d_matrixFreeParentVectorComponent,
      d_matrixFreeParentQuadratureComponent, d_numberQuadraturePointsParent;
    unsigned int d_totallyOwnedCellsParent;

    std::vector<double>       d_shapeValueParentCells;
    std::vector<unsigned int> d_mapParentShapeFuncMemLocation;

    dealii::Quadrature<3> d_quadratureParent;

    //    std::vector<dealii::types::global_dof_index>
    //    d_fullFlattenedArrayMacroCellLocalProcIndexIdMapParent;
    //    std::vector<unsigned int> d_normalCellIdToMacroCellIdMapParent,
    //    d_macroCellIdToNormalCellIdMapParent;
    //    std::vector<dealii::types::global_dof_index>
    //    d_fullFlattenedArrayCellLocalProcIndexIdMapParent;


    const dealii::MatrixFree<3, double> *d_matrixFreeDataChildPtr;

    const dealii::DoFHandler<3> *d_dofHandlerChild;

    unsigned int d_matrixFreeChildVectorComponent,
      d_matrixFreeChildQuadratureComponent, d_numberQuadraturePointsChild;

    unsigned int d_totallyOwnedCellsChild;

    dealii::Quadrature<3> d_quadratureChild;

    std::vector<double> d_shapeValueChildCells;

    std::vector<unsigned int> d_mapChildShapeFuncMemLocation;

    unsigned int d_maxNumberQuadPointsInChildCell;

    //    std::vector<dealii::types::global_dof_index>
    //    d_fullFlattenedArrayMacroCellLocalProcIndexIdMapChild;
    //    std::vector<unsigned int> d_normalCellIdToMacroCellIdMapChild,
    //    d_macroCellIdToNormalCellIdMapChild;
    std::vector<dealii::types::global_dof_index>
      d_fullFlattenedArrayCellLocalProcIndexIdMapChild;



    unsigned int d_numInputVectorsForInterpolationFromParentNodesToChildQuad,
      d_numInputVectorsForInterpolationFromChildNodesToParentQuad;

#ifdef DFTFE_WITH_DEVICE
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
      d_mapChildCellToParentShapeFuncIndexDevice;

    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
      d_mapChildCellsToParentDevice;

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_shapeValueParentCellsDevice;

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_cellLevelParentNodalDevice;
#endif

    std::vector<unsigned int> d_listOfChildCellsWithQuadPoint;


    std::vector<std::vector<unsigned int>> listOfQuadPointsInChildCell;

    std::vector<unsigned int> numberOfParentQuadPointsInChildCell;

    std::vector<double> shapeFunctionValuesFromParentToChild;
    std::vector<double> shapeFunctionValuesChildToParent;

    std::vector<double> d_shapeFunctionValuesChildToParentStrided;

    std::vector<std::map<unsigned int,
                         typename dealii::DoFHandler<3>::active_cell_iterator>>
                                           d_mapParentCellToChildCellsIter;
    std::vector<std::vector<unsigned int>> d_mapParentCellsToChild;
    std::vector<unsigned int>              d_mapChildCellsToParent;
    std::vector<unsigned int>              d_mapChildCellToParentShapeFuncIndex;
  };
} // namespace dftfe

#endif // DFTFE_INVERSEDFTDOFMANAGER_H

