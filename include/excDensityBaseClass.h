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


#ifndef DFTFE_EXCDENSITYBASECLASS_H
#define DFTFE_EXCDENSITYBASECLASS_H

#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include "AuxDensityMatrix.h"

namespace dftfe
{
  enum class densityFamilyType
  {
    LDA,
    GGA,
    LLMGGA,
  };

  enum class xcOutputDataAttributes
  {
    e, // energy density per unit volume
    vSpinUp,
    vSpinDown,
    pdeDensitySpinUp,
    pdeDensitySpinDown,
    pdeSigma,
    pdeLaplacianSpinUp,
    pdeLaplacianSpinDown
  };

  template <dftfe::utils::MemorySpace memorySpace>
  class excDensityBaseClass
  {
  public:
    excDensityBaseClass(const densityFamilyType familyType,
                        const std::vector<DensityDescriptorDataAttributes>
                          &densityDescriptorAttributesList);
    densityFamilyType
    getDensityBasedFamilyType() const;

    const std::vector<DensityDescriptorDataAttributes> &
    getDensityDescriptorAttributesList() const;

    /**
     * x and c denotes exchange and correlation respectively
     */
    virtual void
    computeExcVxcFxc(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    quadPoints,
      const std::vector<double> &    quadWeights,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &cDataout)
      const = 0;

  protected:
    virtual void
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcOutputDataAttributes> &outputDataAttributes)
      const = 0;


    densityFamilyType d_densityFamilyType;
    const std::vector<DensityDescriptorDataAttributes>
      d_densityDescriptorAttributesList;
  };

} // namespace dftfe

#include "excDensityBaseClass.t.cc"
#endif // DFTFE_EXCDENSITYBASECLASS_H
