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

#ifndef DFTFE_EXCSSDFUNCTIONALBASECLASS_H
#define DFTFE_EXCSSDFUNCTIONALBASECLASS_H

#include "AuxDensityMatrix.h"
#include <vector>
#include <fstream>
#include <iostream>
#include "excDensityBaseClass.h"
namespace dftfe
{
  enum class SSDFamilyType
  {
    HYBRID,
    DFTPlusU,
    MGGA
  };

  class ExcSSDFunctionalBaseClass
  {
  public:
    ExcSSDFunctionalBaseClass(bool isSpinPolarized);

    virtual ~ExcSSDFunctionalBaseClass();

    virtual void
    applyWaveFunctionDependentVxc() const = 0;
    virtual void
    updateWaveFunctionDependentVxc() const = 0;
    virtual double
    computeWaveFunctionDependentExcEnergy() const = 0;

    /**
     * x and c denotes exchange and correlation respectively
     */
    virtual void
    computeExcVxcFxc(
      AuxDensityMatrix &         auxDensityMatrix,
      const std::vector<double> &quadPoints,
      const std::vector<double> &quadWeights,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &cDataout)
      const = 0;

    SSDFamilyType
    getSSDFamilyType() const;

  protected:
    SSDFamilyType d_SSDFamilyType;
    bool                   d_isSpinPolarized;
  };
} // namespace dftfe

#endif // DFTFE_EXCSSDFUNCTIONALBASECLASS_H
