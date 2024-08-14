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

#ifndef DFTFE_EXE_EXCDFTPLUSU_H
#define DFTFE_EXE_EXCDFTPLUSU_H



#include "ExcSSDFunctionalBaseClass.h"
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class ExcDFTPlusU : public ExcSSDFunctionalBaseClass<memorySpace>
  {
  public:
    ExcDFTPlusU(xc_func_type *          funcXPtr,
                xc_func_type *          funcCPtr,
                unsigned int            numSpins,
                const densityFamilyType densityFamilyType);

    ~ExcDFTPlusU();

    void
    applyWaveFunctionDependentVxc() const override;
    void
    updateWaveFunctionDependentVxc() const override;
    double
    computeWaveFunctionDependentExcEnergy() const override;

    /**
     * x and c denotes exchange and correlation respectively
     */
    void
    computeOutputXCData(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    quadPoints,
      const std::vector<double> &    quadWeights,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &cDataout)
      const override;


  public:
    excDensityBaseClass<memorySpace> *d_excDensityObjPtr;
  };
} // namespace dftfe
#endif // DFTFE_EXE_EXCDFTPLUSU_H
