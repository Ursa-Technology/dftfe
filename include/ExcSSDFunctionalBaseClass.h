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
namespace dftfe
{
  enum class ExcFamilyType
  {
    LDA,
    GGA,
    LLMGGA,
    HYBRID,
    DFTPlusU,
    MGGA
  };

  enum class densityFamilyType
  {
    LDA,
    GGA,
    LLMGGA,
  };

  enum class xcRemainderOutputDataAttributes
  {
    e, // energy density per unit volume
    vSpinUp,
    vSpinDown,
    pdeDensitySpinUp,
    pdeDensitySpinDown,
    pdeSigma,
    pdeLaplacianSpinUp,
    pdeLaplacianSpinDown,
    pdeTauSpinUp,
    pdeTauSpinDown
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
  class ExcSSDFunctionalBaseClass
  {
  public:
    ExcSSDFunctionalBaseClass(const ExcFamilyType     excFamType,
                              const densityFamilyType densityFamType,
                              const bool              isGradRequired,
                              const bool              isTauRequired,
                              const bool              isIntegrateByParts,
                              const std::vector<DensityDescriptorDataAttributes>
                                &densityDescriptorAttributesList);

    virtual ~ExcSSDFunctionalBaseClass();

    const std::vector<DensityDescriptorDataAttributes> &
    getDensityDescriptorAttributesList() const;

    densityFamilyType
    getDensityBasedFamilyType() const;
    virtual void
    applyWaveFunctionDependentVxc(
      const dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        &                                                                src,
      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &dst,
      const unsigned int inputVecSize,
      const double       factor,
      const unsigned int kPointIndex,
      const unsigned int spinIndex) = 0;
    virtual void
    updateWaveFunctionDependentVxc(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    kPointWeights) = 0;
    virtual double
    computeWaveFunctionDependentExcEnergy(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    kPointWeights) = 0;

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

    bool
    isGradDensityRequired() const;

    bool
    isIntegrateByPartsRequired() const;

    bool
    isTauRequired() const;

    virtual void
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
      const = 0;

  protected:
    const std::vector<DensityDescriptorDataAttributes>
      d_densityDescriptorAttributesList;

    ExcFamilyType     d_ExcFamilyType;
    densityFamilyType d_densityFamilyType;

    bool d_isGradDensityRequired, d_isIntegrateByPartsRequired, d_isTauRequired;
  };
} // namespace dftfe

#include "ExcSSDFunctionalBaseClass.t.cc"
#endif // DFTFE_EXCSSDFUNCTIONALBASECLASS_H
