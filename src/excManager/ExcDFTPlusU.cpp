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
// @author Vishal Subramanian
//

#include "excDensityGGAClass.h"
#include "excDensityLDAClass.h"
#include "excDensityLLMGGAClass.h"
#include "ExcDFTPlusU.h"
#include "Exceptions.h"
#include <dftfeDataTypes.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  ExcDFTPlusU<memorySpace>::ExcDFTPlusU(
    xc_func_type *          funcXPtr,
    xc_func_type *          funcCPtr,
    unsigned int            numSpins,
    const densityFamilyType densityFamilyType)
    : ExcSSDFunctionalBaseClass<memorySpace>(densityFamilyType)
  {
    if (densityFamilyType == densityFamilyType::LDA)
      {
        d_excDensityObjPtr =
          new excDensityLDAClass<memorySpace>(funcXPtr, funcCPtr);
      }
    else if (densityFamilyType == densityFamilyType::GGA)
      {
        d_excDensityObjPtr =
          new excDensityGGAClass<memorySpace>(funcXPtr, funcCPtr);
      }
    else if (densityFamilyType == densityFamilyType::LLMGGA)
      {
        d_excDensityObjPtr =
          new excDensityLLMGGAClass<memorySpace>(funcXPtr, funcCPtr);
      }
    else
      {
        std::string errMsg = "Not implemented";
        dftfe::utils::throwException(false, errMsg);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  ExcDFTPlusU<memorySpace>::~ExcDFTPlusU()
  {
    if (d_excDensityObjPtr != nullptr)
      delete d_excDensityObjPtr;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<memorySpace>::computeOutputXCData(
    AuxDensityMatrix<memorySpace> &auxDensityMatrix,
    const std::vector<double> &    quadPoints,
    const std::vector<double> &    quadWeights,
    std::unordered_map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
    std::unordered_map<xcOutputDataAttributes, std::vector<double>> &cDataOut)
    const
  {
    d_excDensityObjPtr->computeExcVxcFxc(
      auxDensityMatrix, quadPoints, quadWeights, xDataOut, cDataOut);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<memorySpace>::applyWaveFunctionDependentVxc() const
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  ExcDFTPlusU<memorySpace>::updateWaveFunctionDependentVxc() const
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  ExcDFTPlusU<memorySpace>::computeWaveFunctionDependentExcEnergy() const
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template class ExcDFTPlusU<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class ExcDFTPlusU<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
