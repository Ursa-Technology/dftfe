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
#include "FiniteDifference.h"

namespace dftfe
{
  excDensityGGAClass::excDensityGGAClass(xc_func_type *funcXPtr,
                                         xc_func_type *funcCPtr)
    : excDensityBaseClass(densityFamilyType::GGA,
                          std::vector<DensityDescriptorDataAttributes>{
                            DensityDescriptorDataAttributes::valuesSpinUp,
                            DensityDescriptorDataAttributes::valuesSpinDown,
                            DensityDescriptorDataAttributes::sigma})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
    d_NNGGAPtr = nullptr;
  }


  excDensityGGAClass::excDensityGGAClass(xc_func_type *funcXPtr,
                                         xc_func_type *funcCPtr,
                                         std::string   modelXCInputFile)
    : excDensityBaseClass(densityFamilyType::GGA,
                          std::vector<DensityDescriptorDataAttributes>{
                            DensityDescriptorDataAttributes::valuesSpinUp,
                            DensityDescriptorDataAttributes::valuesSpinDown,
                            DensityDescriptorDataAttributes::sigma})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
#ifdef DFTFE_WITH_TORCH
    d_NNGGAPtr = new NNGGA(modelXCInputFile, true);
#endif
  }

  excDensityGGAClass::~excDensityGGAClass()
  {
    if (d_NNGGAPtr != nullptr)
      delete d_NNGGAPtr;
  }

  void
  excDensityGGAClass::checkInputOutputDataAttributesConsistency(
    const std::vector<xcOutputDataAttributes> &outputDataAttributes)
  {
    const std::vector<xcOutputDataAttributes> allowedOutputDataAttributes =
    { xcOutputDataAttributes::e,
      xcOutputDataAttributes::vSpinUp,
      xcOutputDataAttributes::vSpinDown,
      xcOutputDataAttributes::pdeDensitySpinUp,
      xcOutputDataAttributes::pdeDensitySpinDown,
      xcOutputDataAttributes::pdeSigma }

    for (size_type i = 0; i < outputDataAttributes.size(); i++)
    {
      bool isFound = false;
      for (size_type j = 0; j < allowedOutputDataAttributes.size(); j++)
        {
          if (outputDataAttributes[i] == allowedOutputDataAttributes[j])
            isFound = true;
        }


      std::string errMsg =
        "xcOutputDataAttributes do not matched allowed choices for the family type.";
      throwException(isFound, errMsg);
    }
  }

  void
  excDensityGGAClass::computeExcVxcFxc(
    AuxDensityMatrix &                                     auxDensityMatrix,
    const double *                                         quadPoints,
    const double *                                         quadWeights,
    std::map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
    std::map<xcOutputDataAttributes, std::vector<double>> &cDataout) const;
  {
    std::vector<xcOutputDataAttributes> outputDataAttributes;
    for (const auto &element : xDataOut)
      outputDataAttributes.push_back(element.first);

    checkInputOutputDataAttributesConsistency(outputDataAttributes);


    std::map<DensityDescriptorDataAttributes, std::vector<double>>
      densityDescriptorData;

    for (size_type i = 0; i < d_densityDescriptorAttributesList.size(); i++)
      {
        if (d_densityDescriptorAttributesList[i] =
              DensityDescriptorDataAttributes::valuesSpinUp ||
              d_densityDescriptorAttributesList[i] =
                DensityDescriptorDataAttributes::valuesSpinDown)
          densityDescriptorData[d_densityDescriptorAttributesList[i]] =
            std::vector<double>(quadGrid.getLocalSize(), 0);
        else if (d_densityDescriptorAttributesList[i] =
                   DensityDescriptorDataAttributes::sigma)
          densityDescriptorData[d_densityDescriptorAttributesList[i]] =
            std::vector<double>(3 * quadGrid.getLocalSize(), 0);
      }

    bool isVxcBeingComputed = false;
    if (outputDataAttributes.find(xcOutputDataAttributes::vSpinUp) !=
          outputDataAttributes.end() ||
        outputDataAttributes.find(xcOutputDataAttributes::vSpinDown) !=
          outputDataAttributes.end())
      isVxcBeingComputed = true;



    auxDensityRepContainer.applyLocalOperations(quadGrid,
                                                densityDescriptorData);


    auto &densityValuesSpinUp =
      densityDescriptorData.find(DensityDescriptorDataAttributes::valuesSpinUp)
        ->second;
    auto &densityValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::valuesSpinDown)
        ->second;
    auto &sigmaValues =
      densityDescriptorData.find(DensityDescriptorDataAttributes::sigma)
        ->second;



    std::vector<double> densityValues(2 * quadGrid.getLocalSize(), 0);

    std::vector<double> exValues(quadGrid.getLocalSize(), 0);
    std::vector<double> ecValues(quadGrid.getLocalSize(), 0);
    std::vector<double> pdexDensityValuesNonNN(2 * quadGrid.getLocalSize(), 0);
    std::vector<double> pdecDensityValuesNonNN(2 * quadGrid.getLocalSize(), 0);
    std::vector<double> pdexDensitySpinUpValues(quadGrid.getLocalSize(), 0);
    std::vector<double> pdexDensitySpinDownValues(quadGrid.getLocalSize(), 0);
    std::vector<double> pdecDensitySpinUpValues(quadGrid.getLocalSize(), 0);
    std::vector<double> pdecDensitySpinDownValues(quadGrid.getLocalSize(), 0);
    std::vector<double> pdexSigmaValues(3 * quadGrid.getLocalSize(), 0);
    std::vector<double> pdecSigmaValues(3 * quadGrid.getLocalSize(), 0);

    for (size_type i = 0; i < quadGrid.getLocalSize(); i++)
      {
        densityValues[2 * i + 0] = densityValuesSpinUp[i];
        densityValues[2 * i + 1] = densityValuesSpinDown[i];
      }

    xc_gga_exc_vxc(d_funcXPtr,
                   quadGrid.getLocalSize(),
                   &densityValues[0],
                   &sigmaValues[0],
                   &exValues[0],
                   &pdexDensityValuesNonNN[0],
                   &pdexSigmaValues[0]);
    xc_gga_exc_vxc(d_funcCPtr,
                   quadGrid.getLocalSize(),
                   &densityValues[0],
                   &sigmaValues[0],
                   &ecValues[0],
                   &pdexDensityValuesNonNN[0],
                   &pdecSigmaValues[0]);

    for (size_type i = 0; i < quadGrid.getLocalSize(); i++)
      {
        pdexDensitySpinUpValues[i]   = pdexDensityValuesNonNN[2 * i + 0];
        pdexDensitySpinDownValues[i] = pdexDensityValuesNonNN[2 * i + 1];
        pdecDensitySpinUpValues[i]   = pdecDensityValuesNonNN[2 * i + 0];
        pdecDensitySpinDownValues[i] = pdecDensityValuesNonNN[2 * i + 1];
      }

#ifdef DFTFE_WITH_TORCH
    if (d_NNGGAPtr != nullptr)
      {
        std::vector<double> excValuesFromNN(quadGrid.getLocalSize(), 0);
        const size_type     numDescriptors =
          d_densityDescriptorAttributesList.size();
        std::vector<double> pdexcDescriptorValuesFromNN(
          numDescriptors * quadGrid.getLocalSize(), 0);
        d_NNGGAPtr->evaluatevxc(&(densityValues[0]),
                                &sigmaValues[0],
                                quadGrid.getLocalSize(),
                                &excValuesFromNN[0],
                                &pdexcDescriptorValuesFromNN[0]);
        for (size_type i = 0; i < quadGrid.getLocalSize(); i++)
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

    std::vector<double> vxValuesSpinUp(quadGrid.getLocalSize(), 0);
    std::vector<double> vcValuesSpinUp(quadGrid.getLocalSize(), 0);
    std::vector<double> vxValuesSpinDown(quadGrid.getLocalSize(), 0);
    std::vector<double> vcValuesSpinDown(quadGrid.getLocalSize(), 0);
    if (isVxcBeingComputed)
      {
        std::vector<double> pdexGradDensityidimSpinUpStencil(
          quadGrid.getLocalSize() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecGradDensityidimSpinUpStencil(
          quadGrid.getLocalSize() * d_vxcDivergenceTermFDStencilSize, 0.0);

        std::vector<std::vector<double>> divergenceTermsPdexGradDensitySpinUp(
          3, std::vector<double>(quadGrid.getLocalSize(), 0));
        std::vector<std::vector<double>> divergenceTermsPdecGradDensitySpinUp(
          3, std::vector<double>(quadGrid.getLocalSize(), 0));

        std::vector<double> pdexGradDensityidimSpinDownStencil(
          quadGrid.getLocalSize() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecGradDensityidimSpinDownStencil(
          quadGrid.getLocalSize() * d_vxcDivergenceTermFDStencilSize, 0.0);

        std::vector<std::vector<double>> divergenceTermsPdexGradDensitySpinDown(
          3, std::vector<double>(quadGrid.getLocalSize(), 0));
        std::vector<std::vector<double>> divergenceTermsPdecGradDensitySpinDown(
          3, std::vector<double>(quadGrid.getLocalSize(), 0));

        std::map<DensityDescriptorDataAttributes, std::vector<double>>
          densityDescriptorDataForFD;

        std::vector<double> densityValuesPert(2 * quadGrid.getLocalSize(), 0);

        std::vector<double> exValuesPert(quadGrid.getLocalSize(), 0);
        std::vector<double> ecValuesPert(quadGrid.getLocalSize(), 0);
        std::vector<double> pdexDensityValuesNonNNPert(
          2 * quadGrid.getLocalSize(), 0);
        std::vector<double> pdecDensityValuesNonNNPert(
          2 * quadGrid.getLocalSize(), 0);
        std::vector<double> pdexDensitySpinUpValuesPert(quadGrid.getLocalSize(),
                                                        0);
        std::vector<double> pdexDensitySpinDownValuesPert(
          quadGrid.getLocalSize(), 0);
        std::vector<double> pdecDensitySpinUpValuesPert(quadGrid.getLocalSize(),
                                                        0);
        std::vector<double> pdecDensitySpinDownValuesPert(
          quadGrid.getLocalSize(), 0);
        std::vector<double> pdexSigmaValuesPert(3 * quadGrid.getLocalSize(), 0);
        std::vector<double> pdecSigmaValuesPert(3 * quadGrid.getLocalSize(), 0);

        for (size_type i = 0; i < d_densityDescriptorAttributesList.size(); i++)
          {
            if (d_densityDescriptorAttributesList[i] =
                  DensityDescriptorDataAttributes::valuesSpinUp ||
                  d_densityDescriptorAttributesList[i] =
                    DensityDescriptorDataAttributes::valuesSpinDown)
              densityDescriptorDataForFD[d_densityDescriptorAttributesList[i]] =
                std::vector<double>(quadGrid.getLocalSize(), 0);
            else if (d_densityDescriptorAttributesList[i] =
                       DensityDescriptorDataAttributes::sigma)
              densityDescriptorDataForFD[d_densityDescriptorAttributesList[i]] =
                std::vector<double>(3 * quadGrid.getLocalSize(), 0);
          }

        densityDescriptorDataForFD
          [DensityDescriptorDataAttributes::gradValuesSpinUp] =
            std::vector<double>(3 * quadGrid.getLocalSize(), 0);
        densityDescriptorDataForFD
          [DensityDescriptorDataAttributes::gradValuesSpinDown] =
            std::vector<double>(3 * quadGrid.getLocalSize(), 0);

        const std::vector<double> stencil =
          utils::FiniteDifference::getStencilGridOneVariableCentral(
            d_vxcDivergenceTermFDStencilSize, d_spacingFDStencil);

        for (size_type idim = 0; idim < 3; idim++)
          {
            for (size_type istencil = 0;
                 istencil < d_vxcDivergenceTermFDStencilSize;
                 istencil++)
              {
                auxDensityRepContainer.applyLocalOperations(
                  quadGrid,
                  idim,
                  stencil[istencil],
                  densityDescriptorDataForFD);

                auto &densityValuesSpinUpPert =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::gradValuesSpinUp)
                    ->second;
                auto &densityValuesSpinDownPert =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::valuesSpinDown)
                    ->second;
                auto &gradValuesSpinUpPert =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::gradValuesSpinUp)
                    ->second;
                auto &gradValuesSpinDownPert =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::valuesSpinDown)
                    ->second;
                auto &sigmaValuesPert =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::sigma)
                    ->second;

                for (size_type i = 0; i < quadGrid.getLocalSize(); i++)
                  {
                    densityValuesPert[2 * i + 0] = densityValuesSpinUp[i];
                    densityValuesPert[2 * i + 1] = densityValuesSpinDown[i];
                  }

                xc_gga_exc_vxc(d_funcXPtr,
                               quadGrid.getLocalSize(),
                               &densityValuesPert[0],
                               &sigmaValuesPert[0],
                               &exValuesPert[0],
                               &pdexDensityValuesNonNNPert[0],
                               &pdexSigmaValuesPert[0]);
                xc_gga_exc_vxc(d_funcCPtr,
                               quadGrid.getLocalSize(),
                               &densityValuesPert[0],
                               &sigmaValuesPert[0],
                               &ecValuesPert[0],
                               &pdexDensityValuesNonNNPert[0],
                               &pdecSigmaValuesPert[0]);

#ifdef DFTFE_WITH_TORCH
                if (d_NNGGAPtr != nullptr)
                  {
                    std::vector<double> excValuesFromNNPert(
                      quadGrid.getLocalSize(), 0);
                    const size_type numDescriptors =
                      d_densityDescriptorAttributesList.size();
                    std::vector<double> pdexcDescriptorValuesFromNN(
                      numDescriptors * quadGrid.getLocalSize(), 0);
                    d_NNGGAPtr->evaluatevxc(
                      &(densityValuesPert[0]),
                      &sigmaValuesPert[0],
                      quadGrid.getLocalSize(),
                      &excValuesFromNNPert[0],
                      &pdexcDescriptorValuesFromNNPert[0]);
                    for (size_type i = 0; i < quadGrid.getLocalSize(); i++)
                      {
                        pdexSigmaValuesPert[3 * i + 0] +=
                          pdexcDescriptorValuesFromNN[numDescriptors * i + 2];
                        pdexSigmaValuesPert[3 * i + 1] +=
                          pdexcDescriptorValuesFromNN[numDescriptors * i + 3];
                        pdexSigmaValuesPert[3 * i + 2] +=
                          pdexcDescriptorValuesFromNN[numDescriptors * i + 4];
                      }
                  }
#endif
                for (size_type igrid = 0; igrid < quadGrid.getLocalSize();
                     igrid++)
                  {
                    pdexGradDensityidimSpinUpStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        (2.0 * pdexSigmaValuesPert[3 * igrid] +
                         pdexSigmaValuesPert[3 * igrid + 1]) *
                        gradValuesSpinUpPert[idim * quadGrid.getLocalSize() +
                                             igrid];

                    pdecGradDensityidimSpinUpStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        (2.0 * pdecSigmaValuesPert[3 * igrid] +
                         pdecSigmaValuesPert[3 * igrid + 1]) *
                        gradValuesSpinUpPert[idim * quadGrid.getLocalSize() +
                                             igrid];


                    pdexGradDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        (pdexSigmaValuesPert[3 * igrid + 1] +
                         2.0 * pdexSigmaValuesPert[3 * igrid + 2]) *
                        gradValuesSpinDownPert[idim * quadGrid.getLocalSize() +
                                               igrid];


                    pdecGradDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        (pdecSigmaValuesPert[3 * igrid + 1] +
                         2.0 * pdecSigmaValuesPert[3 * igrid + 2]) *
                        gradValuesSpinDownPert[idim * quadGrid.getLocalSize() +
                                               igrid];
                  }
              } // stencil grid filling loop

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadGrid.getLocalSize(),
              &(pdexGradDensityidimSpinUpStencil[0]),
              &(divergenceTermsPdexGradDensitySpinUp[idim][0]));

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadGrid.getLocalSize(),
              &(pdecGradDensityidimSpinUpStencil[0]),
              &(divergenceTermsPdecGradDensitySpinUp[idim][0]));

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadGrid.getLocalSize(),
              &(pdexGradDensityidimSpinDownStencil[0]),
              &(divergenceTermsPdexGradDensitySpinUp[idim][0]));

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadGrid.getLocalSize(),
              &(pdecGradDensityidimSpinDownStencil[0]),
              &(divergenceTermsPdecGradDensitySpinUp[idim][0]));

          } // dim loop


        for (size_type igrid = 0; igrid < quadGrid.getLocalSize(); igrid++)
          {
            vxValuesSpinUp[igrid] =
              pdexDensitySpinUpValues[igrid] -
              (divergenceTermsPdexGradDensitySpinUp[0][igrid] +
               divergenceTermsPdexGradDensitySpinUp[1][igrid] +
               divergenceTermsPdexGradDensitySpinUp[2][igrid]);
            vcValuesSpinUp[igrid] =
              pdecDensitySpinUpValues[igrid] -
              (divergenceTermsPdecGradDensitySpinUp[0][igrid] +
               divergenceTermsPdecGradDensitySpinUp[1][igrid] +
               divergenceTermsPdecGradDensitySpinUp[2][igrid]);

            vxValuesSpinDown[igrid] =
              pdexDensitySpinDownValues[igrid] -
              (divergenceTermsPdexGradDensitySpinDown[0][igrid] +
               divergenceTermsPdexGradDensitySpinDown[1][igrid] +
               divergenceTermsPdexGradDensitySpinDown[2][igrid]);
            vcValuesSpinDown[igrid] =
              pdecDensitySpinDownValues[igrid] -
              (divergenceTermsPdecGradDensitySpinDown[0][igrid] +
               divergenceTermsPdecGradDensitySpinDown[1][igrid] +
               divergenceTermsPdecGradDensitySpinDown[2][igrid]);
          }

      } // is vxc to be computed check

    for (size_type i = 0; i < outputDataAttributes.size(); i++)
      {
        if (outputDataAttributes[i] == xcOutputDataAttributes::e)
          {
            xDataOut.find(outputDataAttributes[i])->second = exValues;

            cDataOut.find(outputDataAttributes[i])->second = ecValues;
          }
        else if (outputDataAttributes[i] == xcOutputDataAttributes::vSpinUp)
          {
            xDataOut.find(outputDataAttributes[i])->second = vxSpinUpValues;

            cDataOut.find(outputDataAttributes[i])->second = vcSpinUpValues;
          }
        else if (outputDataAttributes[i] == xcOutputDataAttributes::vSpinDown)
          {
            xDataOut.find(outputDataAttributes[i])->second = vxSpinDownValues;

            cDataOut.find(outputDataAttributes[i])->second = vcSpinDownValues;
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
} // namespace dftfe
