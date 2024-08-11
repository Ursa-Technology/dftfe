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
// @author Vishal Subramanian, Sambit Das
//

#include "excDensityGGAClass.h"
#include "NNGGA.h"
#include "Exceptions.h"
#include <dftfeDataTypes.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  excDensityGGAClass<memorySpace>::excDensityGGAClass(xc_func_type *funcXPtr,
                                                      xc_func_type *funcCPtr)
    : excDensityBaseClass<memorySpace>(
        densityFamilyType::GGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
    d_NNGGAPtr = nullptr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityGGAClass<memorySpace>::excDensityGGAClass(
    xc_func_type *funcXPtr,
    xc_func_type *funcCPtr,
    std::string   modelXCInputFile)
    : excDensityBaseClass<memorySpace>(
        densityFamilyType::GGA,
        std::vector<DensityDescriptorDataAttributes>{
          DensityDescriptorDataAttributes::valuesSpinUp,
          DensityDescriptorDataAttributes::valuesSpinDown,
          DensityDescriptorDataAttributes::gradValuesSpinUp,
          DensityDescriptorDataAttributes::gradValuesSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
#ifdef DFTFE_WITH_TORCH
    d_NNGGAPtr = new NNGGA(modelXCInputFile, true);
#endif
  }

  template <dftfe::utils::MemorySpace memorySpace>
  excDensityGGAClass<memorySpace>::~excDensityGGAClass()
  {
    if (d_NNGGAPtr != nullptr)
      delete d_NNGGAPtr;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::checkInputOutputDataAttributesConsistency(
    const std::vector<xcOutputDataAttributes> &outputDataAttributes) const
  {
    const std::vector<xcOutputDataAttributes> allowedOutputDataAttributes = {
      xcOutputDataAttributes::e,
      xcOutputDataAttributes::pdeDensitySpinUp,
      xcOutputDataAttributes::pdeDensitySpinDown,
      xcOutputDataAttributes::pdeSigma};

    for (size_t i = 0; i < outputDataAttributes.size(); i++)
      {
        bool isFound = false;
        for (size_t j = 0; j < allowedOutputDataAttributes.size(); j++)
          {
            if (outputDataAttributes[i] == allowedOutputDataAttributes[j])
              isFound = true;
          }


        std::string errMsg =
          "xcOutputDataAttributes do not matched allowed choices for the family type.";
        dftfe::utils::throwException(isFound, errMsg);
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  excDensityGGAClass<memorySpace>::computeExcVxcFxc(
    AuxDensityMatrix<memorySpace> &auxDensityMatrix,
    const std::vector<double> &    quadPoints,
    const std::vector<double> &    quadWeights,
    std::unordered_map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
    std::unordered_map<xcOutputDataAttributes, std::vector<double>> &cDataOut)
    const
  {
    std::vector<xcOutputDataAttributes> outputDataAttributes;
    for (const auto &element : xDataOut)
      outputDataAttributes.push_back(element.first);

    checkInputOutputDataAttributesConsistency(outputDataAttributes);


    std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
      densityDescriptorData;

    for (size_t i = 0; i < this->d_densityDescriptorAttributesList.size(); i++)
      {
        if (this->d_densityDescriptorAttributesList[i] ==
              DensityDescriptorDataAttributes::valuesSpinUp ||
            this->d_densityDescriptorAttributesList[i] ==
              DensityDescriptorDataAttributes::valuesSpinDown)
          densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
            std::vector<double>(quadWeights.size(), 0);
        else if (this->d_densityDescriptorAttributesList[i] ==
                   DensityDescriptorDataAttributes::gradValuesSpinUp ||
                 this->d_densityDescriptorAttributesList[i] ==
                   DensityDescriptorDataAttributes::gradValuesSpinDown)
          densityDescriptorData[this->d_densityDescriptorAttributesList[i]] =
            std::vector<double>(3 * quadWeights.size(), 0);
      }

    auxDensityMatrix.applyLocalOperations(quadPoints, densityDescriptorData);


    auto &densityValuesSpinUp =
      densityDescriptorData.find(DensityDescriptorDataAttributes::valuesSpinUp)
        ->second;
    auto &densityValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::valuesSpinDown)
        ->second;
    auto &gradValuesSpinUp =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::gradValuesSpinUp)
        ->second;
    auto &gradValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::gradValuesSpinDown)
        ->second;



    std::vector<double> densityValues(2 * quadWeights.size(), 0);
    std::vector<double> sigmaValues(3 * quadWeights.size(), 0);

    std::vector<double> exValues(quadWeights.size(), 0);
    std::vector<double> ecValues(quadWeights.size(), 0);
    std::vector<double> pdexDensityValuesNonNN(2 * quadWeights.size(), 0);
    std::vector<double> pdecDensityValuesNonNN(2 * quadWeights.size(), 0);
    std::vector<double> pdexDensitySpinUpValues(quadWeights.size(), 0);
    std::vector<double> pdexDensitySpinDownValues(quadWeights.size(), 0);
    std::vector<double> pdecDensitySpinUpValues(quadWeights.size(), 0);
    std::vector<double> pdecDensitySpinDownValues(quadWeights.size(), 0);
    std::vector<double> pdexSigmaValues(3 * quadWeights.size(), 0);
    std::vector<double> pdecSigmaValues(3 * quadWeights.size(), 0);

    for (size_t i = 0; i < quadWeights.size(); i++)
      {
        densityValues[2 * i + 0] = densityValuesSpinUp[i];
        densityValues[2 * i + 1] = densityValuesSpinDown[i];
        for (size_t j = 0; j < 3; j++)
          {
            sigmaValues[3 * i + 0] +=
              gradValuesSpinUp[3 * i + j] * gradValuesSpinUp[3 * i + j];
            sigmaValues[3 * i + 1] +=
              gradValuesSpinUp[3 * i + j] * gradValuesSpinDown[3 * i + j];
            sigmaValues[3 * i + 2] +=
              gradValuesSpinDown[3 * i + j] * gradValuesSpinDown[3 * i + j];
          }
      }

    xc_gga_exc_vxc(d_funcXPtr,
                   quadWeights.size(),
                   &densityValues[0],
                   &sigmaValues[0],
                   &exValues[0],
                   &pdexDensityValuesNonNN[0],
                   &pdexSigmaValues[0]);
    xc_gga_exc_vxc(d_funcCPtr,
                   quadWeights.size(),
                   &densityValues[0],
                   &sigmaValues[0],
                   &ecValues[0],
                   &pdecDensityValuesNonNN[0],
                   &pdecSigmaValues[0]);

    for (size_t i = 0; i < quadWeights.size(); i++)
      {
        exValues[i] =
          exValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        ecValues[i] =
          ecValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
        pdexDensitySpinUpValues[i]   = pdexDensityValuesNonNN[2 * i + 0];
        pdexDensitySpinDownValues[i] = pdexDensityValuesNonNN[2 * i + 1];
        pdecDensitySpinUpValues[i]   = pdecDensityValuesNonNN[2 * i + 0];
        pdecDensitySpinDownValues[i] = pdecDensityValuesNonNN[2 * i + 1];
      }

#ifdef DFTFE_WITH_TORCH
    if (d_NNGGAPtr != nullptr)
      {
        std::vector<double> excValuesFromNN(quadWeights.size(), 0);
        const size_t        numDescriptors =
          this->d_densityDescriptorAttributesList.size();
        std::vector<double> pdexcDescriptorValuesFromNN(numDescriptors *
                                                          quadWeights.size(),
                                                        0);
        d_NNGGAPtr->evaluatevxc(&(densityValues[0]),
                                &sigmaValues[0],
                                quadWeights.size(),
                                &excValuesFromNN[0],
                                &pdexcDescriptorValuesFromNN[0]);
        for (size_t i = 0; i < quadWeights.size(); i++)
          {
            exValues[i] += excValuesFromNN[i];
            pdexDensitySpinUpValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 0];
            pdexDensitySpinDownValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 1];
            pdexSigmaValues[3 * i + 0] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 2];
            pdexSigmaValues[3 * i + 1] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 3];
            pdexSigmaValues[3 * i + 2] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 4];
          }
      }
#endif

    for (size_t i = 0; i < outputDataAttributes.size(); i++)
      {
        if (outputDataAttributes[i] == xcOutputDataAttributes::e)
          {
            xDataOut.find(outputDataAttributes[i])->second = exValues;

            cDataOut.find(outputDataAttributes[i])->second = ecValues;
          }
        else if (outputDataAttributes[i] ==
                 xcOutputDataAttributes::pdeDensitySpinUp)
          {
            xDataOut.find(outputDataAttributes[i])->second =
              pdexDensitySpinUpValues;

            cDataOut.find(outputDataAttributes[i])->second =
              pdecDensitySpinUpValues;
          }
        else if (outputDataAttributes[i] ==
                 xcOutputDataAttributes::pdeDensitySpinDown)
          {
            xDataOut.find(outputDataAttributes[i])->second =
              pdexDensitySpinDownValues;

            cDataOut.find(outputDataAttributes[i])->second =
              pdecDensitySpinDownValues;
          }
        else if (outputDataAttributes[i] == xcOutputDataAttributes::pdeSigma)
          {
            xDataOut.find(outputDataAttributes[i])->second = pdexSigmaValues;

            cDataOut.find(outputDataAttributes[i])->second = pdecSigmaValues;
          }
      }
  }

  template class excDensityGGAClass<dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class excDensityGGAClass<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
