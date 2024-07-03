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
#ifndef DFTFE_EXCDENSITYGGACLASS_H
#define DFTFE_EXCDENSITYGGACLASS_H

#include <xc.h>
#include <excDensityBaseClass.h>
namespace dftfe
{
  class NNGGA;
  class excDensityGGAClass : public excDensityBaseClass
  {
  public:
    excDensityGGAClass(xc_func_type *funcXPtr, xc_func_type *funcCPtr);


    excDensityGGAClass(xc_func_type *funcXPtr,
                       xc_func_type *funcCPtr,
                       std::string   modelXCInputFile);


    ~excDensityGGAClass();

    void
    computeExcVxcFxc(
      AuxDensityMatrix &         auxDensityMatrix,
      const std::vector<double> &quadPoints,
      const std::vector<double> &quadWeights,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &cDataout)
      const override;

    void
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcOutputDataAttributes> &outputDataAttributes)
      const override;

  private:
    NNGGA *             d_NNGGAPtr;
    xc_func_type *      d_funcXPtr;
    xc_func_type *      d_funcCPtr;
    std::vector<double> d_spacingFDStencil;
    unsigned int        d_vxcDivergenceTermFDStencilSize;
  };
} // namespace dftfe
#endif // DFTFE_EXCDENSITYGGACLASS_H
