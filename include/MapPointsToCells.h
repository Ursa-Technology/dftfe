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
 * @author Bikash Kanungo, Vishal Subramanian
 */

#ifndef dftfeMapPointsToCells_h
#define dftfeMapPointsToCells_h

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/range/adaptors.hpp>
#include "RTreeBox.h"
#include "RTreePoint.h"

#include <TypeConfig.h>
#include <Cell.h>

namespace dftfe
{
  namespace utils
  {
    template <size_type dim, size_type M>
    class MapPointsToCells
    {
    public:

      MapPointsToCells(const MPI_Comm & mpiComm);

      void init(std::vector<std::shared_ptr<const Cell<dim>>> srcCells,
           const std::vector<std::vector<double>> & targetPts,
           std::vector<std::vector<double>> &mapCellsToRealCoordinates,
           std::vector<std::vector<size_type>> &mapCellLocalToProcLocal,
           std::pair<global_size_type,global_size_type> &locallyOwnedRange,
           std::vector<global_size_type> & ghostGlobalIds,
           const double paramCoordsTol);


    private :
      const MPI_Comm d_mpiComm;
      int d_numMPIRank;
      int d_thisRank;

      //          std::map<size_type, std::vector<size_type>> d_mapCellsToLocalPoints;
      //          std::map<size_type, std::vector<std::pair<size_type,size_type>>> d_mapCellsToNonLocalPoints;

    }; // end of class MapPointsToCells
  }// end of namespace utils
}// end of namespace dftfe

#include "../src/TransferBetweenMeshes/MapPointsToCells.t.cc"
#endif // dftfeMapPointsToCells_h
