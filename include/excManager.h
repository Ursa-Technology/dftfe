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

#ifndef DFTFE_EXCMANAGER_H
#define DFTFE_EXCMANAGER_H

#include <xc.h>
#include <excDensityBaseClass.h>
#include <ExcSSDFunctionalBaseClass.h>
namespace dftfe
{
  enum class XCPrimaryVariable
  {
    DENSITY,
    SSDETERMINANT
  };

  template <dftfe::utils::MemorySpace memorySpace>
  class excManager
  {
  public:
    /**
     * @brief Constructor
     *
     */
    excManager();

    /**
     * @brief  destructor
     */
    ~excManager();

    void
    clear();


    void
    init(std::string XCType,
         bool        isSpinPolarized,
         std::string modelXCInputFile);

    //    densityFamilyType
    //    getDensityBasedFamilyType() const;



    excDensityBaseClass<memorySpace> *
    getExcDensityObj();

    ExcSSDFunctionalBaseClass<memorySpace> *
    getExcSSDFunctionalObj();


    const excDensityBaseClass<memorySpace> *
    getExcDensityObj() const;

    const ExcSSDFunctionalBaseClass<memorySpace> *
    getExcSSDFunctionalObj() const;

    XCPrimaryVariable
    getXCPrimaryVariable() const;


  private:
    /// objects for various exchange-correlations (from libxc package)
    xc_func_type *d_funcXPtr;
    xc_func_type *d_funcCPtr;

    excDensityBaseClass<memorySpace> *      d_excDensityObjPtr;
    ExcSSDFunctionalBaseClass<memorySpace> *d_SSDObjPtr;

    XCPrimaryVariable d_xcPrimVariable;
  };
} // namespace dftfe

#endif // DFTFE_EXCMANAGER_H
