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

#ifndef DFTFE_TRANSFERDATABETWEENMESHESBASE_H
#define DFTFE_TRANSFERDATABETWEENMESHESBASE_H

#include "headers.h"
#include "linearAlgebraOperationsInternal.h"
#include "linearAlgebraOperations.h"
#include "vectorUtilities.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class TransferDataBetweenMeshesBase
  {
  public:
    template <typename T>
    virtual void
    interpolateMesh1DataToMesh2QuadPoints(
      const dftfe::linearAlgebra::MultiVector<T,
                                              memorySpace> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
      dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST> &outputQuadData,
      bool resizeOutputVec) = 0;

    template <typename T>
    virtual void
    interpolateMesh2DataToMesh1QuadPoints(
      const dftfe::linearAlgebra::MultiVector<T,
                                              memorySpace> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
      dftfe::linearAlgebra::MultiVector<T,
                                        memorySpace> &                 outputQuadData,
      bool resizeOutputVec) = 0;

    template <typename T>
    virtual void
    interpolateMesh1DataToMesh2QuadPoints(
      const distributedCPUVec<T> &inputVec,
      const unsigned int               numberOfVectors,
      const std::vector<dealii::types::global_dof_index>
                                                                   &                  fullFlattenedArrayCellLocalProcIndexIdMapParent,
      dftfe::utils::MemoryStorage<T,
                                  dftfe::utils::MemorySpace::HOST> &outputQuadData,
      bool resizeOutputVec) =0;

    template <typename T>
    virtual void
    interpolateMesh2DataToMesh1QuadPoints(
      const distributedCPUVec<T> &inputVec,
      const unsigned int               numberOfVectors,
      dftfe::utils::MemoryStorage<T,
                                  dftfe::utils::MemorySpace::HOST> &            outputQuadData,
      bool resizeOutputVec) = 0;

  };
}

#endif // DFTFE_TRANSFERDATABETWEENMESHESBASE_H

