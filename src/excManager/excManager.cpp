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

#include <excManager.h>
#include <excDensityGGAClass.h>
#include <excDensityLDAClass.h>
#include <excDensityLLMGGAClass.h>
#include "ExcDFTPlusU.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  excManager<memorySpace>::excManager()
  {
    d_funcXPtr         = nullptr;
    d_funcCPtr         = nullptr;
    d_excDensityObjPtr = nullptr;
    d_SSDObjPtr        = nullptr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excManager<memorySpace>::~excManager()
  {
    clear();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excManager<memorySpace>::clear()
  {
    if (d_funcXPtr != nullptr)
      {
        xc_func_end(d_funcXPtr);
        delete d_funcXPtr;
      }

    if (d_funcCPtr != nullptr)
      {
        xc_func_end(d_funcCPtr);
        delete d_funcCPtr;
      }

    if (d_excDensityObjPtr != nullptr)
      delete d_excDensityObjPtr;

    if (d_SSDObjPtr != nullptr)
      delete d_SSDObjPtr;

    d_funcXPtr         = nullptr;
    d_funcCPtr         = nullptr;
    d_excDensityObjPtr = nullptr;
    d_SSDObjPtr        = nullptr;
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  excManager<memorySpace>::init(std::string XCType,
                                bool        isSpinPolarized,
                                std::string modelXCInputFile)
  {
    clear();

    d_funcXPtr = new xc_func_type;
    d_funcCPtr = new xc_func_type;


    int exceptParamX = -1, exceptParamC = -1;


    if (XCType == "LDA-PZ")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_LDA_X, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_LDA_C_PZ, XC_POLARIZED);
        d_excDensityObjPtr =
          new excDensityLDAClass<memorySpace>(d_funcXPtr, d_funcCPtr);

        d_SSDObjPtr = nullptr;

        d_xcPrimVariable = XCPrimaryVariable::DENSITY;
      }
    else if (XCType == "LDA-PW")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_LDA_X, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_LDA_C_PW, XC_POLARIZED);
        d_excDensityObjPtr =
          new excDensityLDAClass<memorySpace>(d_funcXPtr, d_funcCPtr);


        d_SSDObjPtr      = nullptr;
        d_xcPrimVariable = XCPrimaryVariable::DENSITY;
      }
    else if (XCType == "LDA-VWN")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_LDA_X, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_LDA_C_VWN, XC_POLARIZED);
        d_excDensityObjPtr =
          new excDensityLDAClass<memorySpace>(d_funcXPtr, d_funcCPtr);

        d_SSDObjPtr      = nullptr;
        d_xcPrimVariable = XCPrimaryVariable::DENSITY;
      }
    else if (XCType == "GGA-PBE")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_GGA_X_PBE, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_GGA_C_PBE, XC_POLARIZED);
        d_excDensityObjPtr =
          new excDensityGGAClass<memorySpace>(d_funcXPtr, d_funcCPtr);

        d_SSDObjPtr      = nullptr;
        d_xcPrimVariable = XCPrimaryVariable::DENSITY;
      }
    else if (XCType == "GGA-RPBE")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_GGA_X_RPBE, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_GGA_C_PBE, XC_POLARIZED);
        d_excDensityObjPtr =
          new excDensityGGAClass<memorySpace>(d_funcXPtr, d_funcCPtr);

        d_SSDObjPtr      = nullptr;
        d_xcPrimVariable = XCPrimaryVariable::DENSITY;
      }
    else if (XCType == "GGA-LBxPBEc")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_GGA_X_LB, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_GGA_C_PBE, XC_POLARIZED);

        d_excDensityObjPtr =
          new excDensityGGAClass<memorySpace>(d_funcXPtr, d_funcCPtr);

        d_SSDObjPtr      = nullptr;
        d_xcPrimVariable = XCPrimaryVariable::DENSITY;
      }
    else if (XCType == "MLXC-NNLDA")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_LDA_X, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_LDA_C_PW, XC_POLARIZED);
        d_excDensityObjPtr =
          new excDensityLDAClass<memorySpace>(d_funcXPtr,
                                              d_funcCPtr,
                                              modelXCInputFile);

        d_SSDObjPtr      = nullptr;
        d_xcPrimVariable = XCPrimaryVariable::DENSITY;
      }
    else if (XCType == "MLXC-NNGGA")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_GGA_X_PBE, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_GGA_C_PBE, XC_POLARIZED);
        d_excDensityObjPtr =
          new excDensityGGAClass<memorySpace>(d_funcXPtr,
                                              d_funcCPtr,
                                              modelXCInputFile);

        d_SSDObjPtr      = nullptr;
        d_xcPrimVariable = XCPrimaryVariable::DENSITY;
      }
    else if (XCType == "MLXC-NNLLMGGA")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_GGA_X_PBE, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_GGA_C_PBE, XC_POLARIZED);
        d_excDensityObjPtr =
          new excDensityLLMGGAClass<memorySpace>(d_funcXPtr,
                                                 d_funcCPtr,
                                                 modelXCInputFile);

        d_SSDObjPtr      = nullptr;
        d_xcPrimVariable = XCPrimaryVariable::DENSITY;
      }
    else if (XCType == "PBE+U")
      {
        exceptParamX = xc_func_init(d_funcXPtr, XC_GGA_X_PBE, XC_POLARIZED);
        exceptParamC = xc_func_init(d_funcCPtr, XC_GGA_C_PBE, XC_POLARIZED);
        unsigned int numSpin = 0;
        if (isSpinPolarized == true)
          {
            numSpin = 1;
          }
        d_SSDObjPtr = new ExcDFTPlusU<memorySpace>(d_funcXPtr,
                                                   d_funcCPtr,
                                                   numSpin,
                                                   densityFamilyType::GGA);

        d_excDensityObjPtr = nullptr;
        d_xcPrimVariable   = XCPrimaryVariable::SSDETERMINANT;
      }
    else
      {
        std::cout << "Error in xc code \n";
        if (exceptParamX != 0 || exceptParamC != 0)
          {
            std::cout << "-------------------------------------" << std::endl;
            std::cout << "Exchange or Correlation Functional not found"
                      << std::endl;
            std::cout << "-------------------------------------" << std::endl;
            exit(-1);
          }
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityBaseClass<memorySpace> *
  excManager<memorySpace>::getExcDensityObj()
  {
    return d_excDensityObjPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  ExcSSDFunctionalBaseClass<memorySpace> *
  excManager<memorySpace>::getExcSSDFunctionalObj()
  {
    return d_SSDObjPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const excDensityBaseClass<memorySpace> *
  excManager<memorySpace>::getExcDensityObj() const
  {
    return d_excDensityObjPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  const ExcSSDFunctionalBaseClass<memorySpace> *
  excManager<memorySpace>::getExcSSDFunctionalObj() const
  {
    return d_SSDObjPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  XCPrimaryVariable
  excManager<memorySpace>::getXCPrimaryVariable() const
  {
    return d_xcPrimVariable;
  }

  template class excManager<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class excManager<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
