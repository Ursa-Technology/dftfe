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
  enum class ExcFamilyType
  {
    PURE_DENSITY,
    HYBRID,
    DFTPlusU,
    MGGA
  };


  /**
   * @brief This class provides the structure for all
   * Exc functional dependent on Single Slater Determinant
   * such as DFT+U, Hybrid and Tau dependent MGGA.
   * This derived class of this class provides the
   * description to handle non-multiplicative potential
   * arising for the above formulations
   *
   * @author Vishal Subramanian, Sambit Das
   */
  template <dftfe::utils::MemorySpace memorySpace>
  class GKSBaseClass
  {
  public:
    GKSBaseClass(const ExcFamilyType excFamType);

    virtual ~GKSBaseClass();

    virtual void
    applyWaveFunctionDependentVxc() const = 0;
    virtual void
    updateWaveFunctionDependentVxc() const = 0;
    virtual double
    computeWaveFunctionDependentExcEnergy() const = 0;

    /**
     * x and c denotes exchange and correlation respectively.
     * This function computes the rho and tau dependent parts
     * of the Exc and Vxc functionals
     */
    virtual void
    computeOutputXCData(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    quadPoints,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &xDataOut,
      std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
        &cDataout) const = 0;

    ExcFamilyType
    getExcFamilyType() const;

    densityFamilyType
      getDensityFamilyType() const;

  protected:
    ExcFamilyType     d_ExcFamilyType;
    densityFamilyType d_densityFamilyType;
  };
} // namespace dftfe

#include "GKSBaseClass.t.cc"
#endif // DFTFE_EXCSSDFUNCTIONALBASECLASS_H
