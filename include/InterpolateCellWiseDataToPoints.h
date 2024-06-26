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

/*
 * @author Vishal Subramanian, Bikash Kanungo
 */

#ifndef DFTFE_INTERPOLATECELLWISEDATATOPOINTS_H
#define DFTFE_INTERPOLATECELLWISEDATATOPOINTS_H

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/range/adaptors.hpp>
#include "BLASWrapper.h"
#include "MapPointsToCells.h"
#include "Cell.h"
#include "InterpolateFromCellToLocalPoints.h"


namespace dftfe
{
  template <typename T,dftfe::utils::MemorySpace memorySpace>
  class InterpolateCellWiseDataToPoints
  {
  public:

    InterpolateCellWiseDataToPoints(const std::vector<std::shared_ptr<const dftfe::utils::Cell<3>>> &srcCells,
                                    std::vector<std::shared_ptr<InterpolateFromCellToLocalPoints<memorySpace>>> interpolateLocalObj,
                                    const std::vector<std::vector<double>> & targetPts,
                                    const std::vector<unsigned int> &numDofsPerElem,
                                    const MPI_Comm & mpiComm);

    void interpolateSrcDataToTargetPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>> &
        BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<T,
                                              memorySpace> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                        memorySpace> &mapVecToCells,
      dftfe::utils::MemoryStorage<T,
                                  memorySpace> &outputData,
      bool resizeData = false) ;

    void interpolateSrcDataToTargetPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &
        BLASWrapperPtr,
      const distributedCPUVec<T> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                        dftfe::utils::MemorySpace::HOST>&                  mapVecToCells,
      dftfe::utils::MemoryStorage<T,
                                  dftfe::utils::MemorySpace::HOST> &outputData,
      bool resizeData = false) ;

  private:
    dftfe::utils::MapPointsToCells<3,8> d_mapPoints;/// TODO check if M=8 is optimal
//    std::vector<T> d_shapeFuncValues;
    std::vector<size_type> d_cellPointStartIndex, d_cellShapeFuncStartIndex;
    const MPI_Comm d_mpiComm;

    std::shared_ptr<dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>> d_mpiPatternP2PPtr;

    std::shared_ptr<dftfe::utils::mpi::MPICommunicatorP2P<T,dftfe::utils::MemorySpace::HOST>> d_mpiCommP2PPtr;

    std::shared_ptr<dftfe::utils::mpi::MPIPatternP2P<memorySpace>> d_mpiP2PPtrMemSpace;
    std::unique_ptr<dftfe::utils::mpi::MPICommunicatorP2P<T,
                                                          memorySpace>> d_mpiCommPtrMemSpace;

//    dftfe::utils::MemoryStorage<size_type, memorySpace>
//      d_mapPointToShapeFuncIndexMemSpace;

//    dftfe::utils::MemoryStorage<size_type, memorySpace>
//      d_mapPointToCellIndexMemSpace;

    dftfe::utils::MemoryStorage<global_size_type, memorySpace>
      d_mapPointToProcLocalMemSpace;

//    dftfe::utils::MemoryStorage<dataTypes::number,
//                                memorySpace>
//      d_shapeValuesMemSpace;

    dftfe::utils::MemoryStorage<dataTypes::number,
                                memorySpace>
      d_cellLevelParentNodalMemSpace;

    size_type d_numPointsLocal;
    size_type d_numCells;
    std::vector<unsigned int> d_numDofsPerElement;
    std::vector<unsigned int> d_cumulativeDofs;
    size_type totalDofsInCells;

    std::vector<size_type> d_numPointsInCell;

    std::vector<std::vector<size_type>> d_mapCellLocalToProcLocal;


    size_type d_numLocalPtsSze;

    size_type d_pointsFoundInProc;

    std::vector<global_size_type> d_ghostGlobalIds;
    std::pair<global_size_type,global_size_type> d_localRange;

    dftfe::utils::MemoryStorage<dataTypes::number,
                                memorySpace>
      d_tempOutputMemSpace;

    std::vector<std::shared_ptr<InterpolateFromCellToLocalPoints<memorySpace>>> d_interpolateLocalObj;
  }; // end of class InterpolateCellWiseDataToPoints
} // end of namespace dftfe


#endif // DFTFE_INTERPOLATECELLWISEDATATOPOINTS_H
