#include <excDensityLLMGGAClass.h>
#include <NNLLMGGA.h>
#include "Exceptions.h"
#include "FiniteDifference.h"

namespace dftfe
{
  excDensityLLMGGAClass::excDensityLLMGGAClass(xc_func_type *funcXPtr,
                                               xc_func_type *funcCPtr)
    : excDensityBaseClass(densityFamilyType::LLMGGA,
                          std::vector<DensityDescriptorDataAttributes>{
                            DensityDescriptorDataAttributes::valuesSpinUp,
                            DensityDescriptorDataAttributes::valuesSpinDown,
                            DensityDescriptorDataAttributes::gradValueSpinUp,
                            DensityDescriptorDataAttributes::gradValueSpinDown,
                            DensityDescriptorDataAttributes::laplacianSpinUp,
                            DensityDescriptorDataAttributes::laplacianSpinDown})
  {
    d_funcXPtr    = funcXPtr;
    d_funcCPtr    = funcCPtr;
    d_NNLLMGGAPtr = nullptr;
  }

  excDensityLLMGGAClass::excDensityLLMGGAClass(xc_func_type *funcXPtr,
                                               xc_func_type *funcCPtr,
                                               std::string   modelXCInputFile)
    : excDensityBaseClass(densityFamilyType::LLMGGA,
                          std::vector<DensityDescriptorDataAttributes>{
                            DensityDescriptorDataAttributes::valuesSpinUp,
                            DensityDescriptorDataAttributes::valuesSpinDown,
                            DensityDescriptorDataAttributes::gradValueSpinUp,
                            DensityDescriptorDataAttributes::gradValueSpinDown,
                            DensityDescriptorDataAttributes::laplacianSpinUp,
                            DensityDescriptorDataAttributes::laplacianSpinDown})
  {
    d_funcXPtr = funcXPtr;
    d_funcCPtr = funcCPtr;
#ifdef DFTFE_WITH_TORCH
    d_NNLLMGGAPtr = new NNLLMGGA(modelXCInputFile, true);
#endif
  }

  excDensityLLMGGAClass::~excDensityLLMGGAClass()
  {
    if (d_NNLLMGGAPtr != nullptr)
      delete d_NNLLMGGAPtr;
  }

  void
  excDensityLLMGGAClass::checkInputOutputDataAttributesConsistency(
    const std::vector<xcOutputDataAttributes> &outputDataAttributes)
  {
    const std::vector<xcOutputDataAttributes> allowedOutputDataAttributes =
    { xcOutputDataAttributes::e,
      xcOutputDataAttributes::vSpinUp,
      xcOutputDataAttributes::vSpinDown,
      xcOutputDataAttributes::pdeDensitySpinUp,
      xcOutputDataAttributes::pdeDensitySpinDown,
      xcOutputDataAttributes::pdeSigma,
      xcOutputDataAttributes::pdeLaplacianSpinUp,
      xcOutputDataAttributes::pdeLaplacianSpinDown }

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
    AuxDensityMatrix &         auxDensityMatrix,
    const std::vector<double> &quadPoints,
    const std::vector<double> &quadWeights,
    std::unordered_map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
    std::unordered_map<xcOutputDataAttributes, std::vector<double>> &cDataout)
    const
  {
    std::vector<xcOutputDataAttributes> outputDataAttributes;
    for (const auto &element : xDataOut)
      outputDataAttributes.push_back(element.first);

    checkInputOutputDataAttributesConsistency(outputDataAttributes);

    std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
      densityDescriptorData;

    // d_densityDescriptorAttributesList not defined
    for (size_type i = 0; i < d_densityDescriptorAttributesList.size(); i++)
      {
        if (d_densityDescriptorAttributesList[i] =
              DensityDescriptorDataAttributes::valuesSpinUp ||
              d_densityDescriptorAttributesList[i] =
                DensityDescriptorDataAttributes::valuesSpinDown)
          densityDescriptorData[d_densityDescriptorAttributesList[i]] =
            std::vector<double>(quadWeights.size(), 0);
        else if (d_densityDescriptorAttributesList[i] =
                   DensityDescriptorDataAttributes::sigma)
          densityDescriptorData[d_densityDescriptorAttributesList[i]] =
            std::vector<double>(3 * quadWeights.size(), 0);
      }

    bool isVxcBeingComputed = false;
    if (outputDataAttributes.find(xcOutputDataAttributes::vSpinUp) !=
          outputDataAttributes.end() ||
        outputDataAttributes.find(xcOutputDataAttributes::vSpinDown) !=
          outputDataAttributes.end())
      isVxcBeingComputed = true;

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
        .find(DensityDescriptorDataAttributes::gradValueSpinUp)
        ->second;
    auto &gradValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::gradValueSpinDown)
        ->second;
    auto &laplacianValuesSpinUp =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::laplacianSpinUp)
        ->second;
    auto &laplacianValuesSpinDown =
      densityDescriptorData
        .find(DensityDescriptorDataAttributes::laplacianSpinDown)
        ->second;


    std::vector<double> densityValues(2 * quadWeights.size(), 0);
    std::vector<double> sigmaValues(3 * quadWeights.size(), 0);
    std::vector<double> laplacianValues(2 * quadWeights.size(), 0);

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


    for (size_type i = 0; i < quadWeights.size(); i++)
      {
        densityValues[2 * i + 0] = densityValuesSpinUp[i];
        densityValues[2 * i + 1] = densityValuesSpinDown[i];

        for (size_type j = 0; j < 3; j++)
          {
            sigmaValues[3 * i + 0] +=
              gradValuesSpinUp[3 * i + j] * gradValuesSpinUp[3 * i + j];
            sigmaValues[3 * i + 1] +=
              gradValuesSpinUp[3 * i + j] * gradValuesSpinDown[3 * i + j];
            sigmaValues[3 * i + 2] +=
              gradValuesSpinDown[3 * i + j] * gradValuesSpinDown[3 * i + j];
          }

        laplacianValues[2 * i + 0] = laplacianValuesSpinUp[i];
        laplacianValues[2 * i + 1] = laplacianValuesSpinDown[i];
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

    for (size_type i = 0; i < quadWeights.size(); i++)
      {
        pdexDensitySpinUpValues[i]   = pdexDensityValuesNonNN[2 * i + 0];
        pdexDensitySpinDownValues[i] = pdexDensityValuesNonNN[2 * i + 1];
        pdecDensitySpinUpValues[i]   = pdecDensityValuesNonNN[2 * i + 0];
        pdecDensitySpinDownValues[i] = pdecDensityValuesNonNN[2 * i + 1];
      }


#ifdef DFTFE_WITH_TORCH
    if (d_NNLLMGGAPtr != nullptr)
      {
        std::vector<double> excValuesFromNN(quadWeights.size(), 0);
        const size_type     numDescriptors =
          d_densityDescriptorAttributesList.size();
        std::vector<double> pdexcDescriptorValuesFromNN(numDescriptors *
                                                          quadWeights.size(),
                                                        0);

        d_NNLLMGGAPtr->evaluatevxc(&(densityValues[0]),
                                   &sigmaValues[0],
                                   &laplacianValues[0],
                                   quadWeights.size(),
                                   &excValuesFromNN[0],
                                   &pdexcDescriptorValuesFromNN[0]);

        for (size_type i = 0; i < quadWeights.size(); i++)
          {
            exValues[i] += excValuesFromNN[i];
            pdexDensitySpinUpValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 0];
            pdexDensitySpinDownValues[i] +=
              pdexcDescriptorValuesFromNN[numDescriptors * i + 1];
          }
      }
#endif

    std::vector<double> vxValuesSpinUp(quadWeights.size(), 0);
    std::vector<double> vcValuesSpinUp(quadWeights.size(), 0);
    std::vector<double> vxValuesSpinDown(quadWeights.size(), 0);
    std::vector<double> vcValuesSpinDown(quadWeights.size(), 0);

    if (isVxcBeingComputed)
      {
        std::vector<double> pdexGradDensityidimSpinTotalStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecGradDensityidimSpinTotalStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);

        std::vector<double> pdexLapDensityidimSpinTotalStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecLapDensityidimSpinTotalStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);

        std::vector<std::vector<double>> laplacianTermsPdexLapDensitySpinTotal(
          3, std::vector<double>(quadWeights.size(), 0));
        std::vector<std::vector<double>> laplacianTermsPdecLapDensitySpinTotal(
          3, std::vector<double>(quadWeights.size(), 0));


        std::vector<double> pdexGradDensityidimSpinUpStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecGradDensityidimSpinUpStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<std::vector<double>> divergenceTermsPdexGradDensitySpinUp(
          3, std::vector<double>(quadWeights.size(), 0));
        std::vector<std::vector<double>> divergenceTermsPdecGradDensitySpinUp(
          3, std::vector<double>(quadWeights.size(), 0));
        std::vector<double> pdexLapDensityidimSpinUpStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecLapDensityidimSpinUpStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<std::vector<double>> laplacianTermsPdexLapDensitySpinUp(
          3, std::vector<double>(quadWeights.size(), 0));
        std::vector<std::vector<double>> laplacianTermsPdecLapDensitySpinUp(
          3, std::vector<double>(quadWeights.size(), 0));

        std::vector<double> pdexGradDensityidimSpinDownStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecGradDensityidimSpinDownStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<std::vector<double>> divergenceTermsPdexGradDensitySpinDown(
          3, std::vector<double>(quadWeights.size(), 0));
        std::vector<std::vector<double>> divergenceTermsPdecGradDensitySpinDown(
          3, std::vector<double>(quadWeights.size(), 0));
        std::vector<double> pdexLapDensityidimSpinDownStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<double> pdecLapDensityidimSpinDownStencil(
          quadWeights.size() * d_vxcDivergenceTermFDStencilSize, 0.0);
        std::vector<std::vector<double>> laplacianTermsPdexLapDensitySpinDown(
          3, std::vector<double>(quadWeights.size(), 0));
        std::vector<std::vector<double>> laplacianTermsPdecLapDensitySpinDown(
          3, std::vector<double>(quadWeights.size(), 0));



        const std::vector<double> stencil1D =
          utils::FiniteDifference::getStencilGridOneVariableCentral(
            d_vxcDivergenceTermFDStencilSize, d_spacingFDStencil);

        std::map<DensityDescriptorDataAttributes, std::vector<double>>
          densityDescriptorDataForFD;

        std::vector<double> densityValuesFD(2 * quadWeights.size(), 0);
        std::vector<double> sigmaValuesFD(3 * quadWeights.size(), 0);
        std::vector<double> laplacianValuesFD(2 * quadWeights.size(), 0);

        std::vector<double> exValuesFD(quadWeights.size(), 0);
        std::vector<double> ecValuesFD(quadWeights.size(), 0);
        std::vector<double> pdexDensityValuesNonNNFD(2 * quadWeights.size(),
                                                     0); // not used
        std::vector<double> pdecDensityValuesNonNNFD(2 * quadWeights.size(),
                                                     0); // not used
        std::vector<double> pdexSigmaValuesFD(3 * quadWeights.size(), 0);
        std::vector<double> pdecSigmaValuesFD(3 * quadWeights.size(), 0);
        std::vector<double> pdexLaplacianValuesFD(2 * quadWeights.size(), 0);
        std::vector<double> pdecLaplacianValuesFD(2 * quadWeights.size(),
                                                  0); // not used



        for (size_type idim = 0; idim < 3; idim++)
          {
            for (size_type istencil = 0;
                 istencil < d_vxcDivergenceTermFDStencilSize;
                 istencil++)
              {
                std::vector<double> quadShiftedFD = quadGrid;
                for (size_type igrid = 0; igrid < quadWeights.size(); igrid++)
                  {
                    // create FD grid
                    quadShiftedFD[3 * igrid + idim] =
                      quadGrid[3 * igrid + idim] +
                      (-std::floor(d_vxcDivergenceTermFDStencilSize / 2) *
                         hSizeFD[igrid] +
                       igrid * hSizeFD[igrid]);
                  }

                auxDensityMatrix.applyLocalOperations(
                  quadShiftedFD, densityDescriptorDataForFD);

                auto &densityValuesSpinUpFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::valuesSpinUp)
                    ->second;
                auto &densityValuesSpinDownFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::valuesSpinDown)
                    ->second;
                auto &gradValuesSpinUpFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::gradValueSpinUp)
                    ->second;
                auto &gradValuesSpinDownFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::gradValueSpinDown)
                    ->second;
                auto &laplacianValuesSpinUpFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::laplacianSpinUp)
                    ->second;
                auto &laplacianValuesSpinDownFD =
                  densityDescriptorDataForFD
                    .find(DensityDescriptorDataAttributes::laplacianSpinDown)
                    ->second;

                for (size_type i = 0; i < quadWeights.size(); i++)
                  {
                    densityValuesFD[2 * i + 0] = densityValuesSpinUpFD[i];
                    densityValuesFD[2 * i + 1] = densityValuesSpinDownFD[i];

                    sigmaValuesFD[3 * i + 0] = 0;
                    sigmaValuesFD[3 * i + 1] = 0;
                    sigmaValuesFD[3 * i + 2] = 0;

                    for (size_type j = 0; j < 3; j++)
                      {
                        sigmaValuesFD[3 * i + 0] +=
                          gradValuesSpinUpFD[3 * i + j] *
                          gradValuesSpinUpFD[3 * i + j];
                        sigmaValuesFD[3 * i + 1] +=
                          gradValuesSpinUpFD[3 * i + j] *
                          gradValuesSpinDownFD[3 * i + j];
                        sigmaValuesFD[3 * i + 2] +=
                          gradValuesSpinDownFD[3 * i + j] *
                          gradValuesSpinDownFD[3 * i + j];
                      }

                    laplacianValuesFD[2 * i + 0] = laplacianValuesSpinUp[i];
                    laplacianValuesFD[2 * i + 1] = laplacianValuesSpinDown[i];
                  }

                xc_gga_exc_vxc(d_funcXPtr,
                               quadWeights.size(),
                               &densityValuesFD[0],
                               &sigmaValuesFD[0],
                               &exValuesFD[0],
                               &pdexDensityValuesNonNNFD[0],
                               &pdexSigmaValuesFD[0]);
                xc_gga_exc_vxc(d_funcCPtr,
                               quadWeights.size(),
                               &densityValuesFD[0],
                               &sigmaValuesFD[0],
                               &ecValuesFD[0],
                               &pdecDensityValuesNonNNFD[0],
                               &pdecSigmaValuesFD[0]);

#ifdef DFTFE_WITH_TORCH
                if (d_NNLLMGGAPtr != nullptr)
                  {
                    std::vector<double> excValuesFromNNFD(quadWeights.size(),
                                                          0);
                    const size_type     numDescriptors =
                      d_densityDescriptorAttributesList.size();
                    std::vector<double> pdexcDescriptorValuesFromNNFD(
                      numDescriptors * quadWeights.size(), 0);


                    d_NNLLMGGAPtr->evaluatevxc(
                      &(densityValuesFD[0]),
                      &sigmaValuesFD[0],
                      &laplacianValuesFD[0],
                      quadWeights.size(),
                      &excValuesFromNNFD[0],
                      &pdexcDescriptorValuesFromNNFD[0]);

                    for (size_type i = 0; i < quadWeights.size(); i++)
                      {
                        pdexSigmaValuesFD[3 * i + 0] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 2];
                        pdexSigmaValuesFD[3 * i + 1] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 3];
                        pdexSigmaValuesFD[3 * i + 2] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 4];
                        pdexLaplacianValuesFD[2 * i + 0] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 5];
                        pdexLaplacianValuesFD[2 * i + 1] +=
                          pdexcDescriptorValuesFromNNFD[numDescriptors * i + 6];
                      }
                  }
#endif

                for (size_type igrid = 0; igrid < quadWeights.size(); igrid++)
                  {
                    pdexGradDensityidimSpinUpStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        (2.0 * pdexSigmaValuesFD[3 * igrid] +
                         pdexSigmaValuesFD[3 * igrid + 1] +
                         2.0 * pdexSigmaValuesFD[3 * igrid + 2]) *
                        (gradValuesSpinUpFD[idim * quadWeights.size() + igrid] +
                         gradValuesSpinDownFD[idim * quadWeights.size() +
                                              igrid]);

                    pdecGradDensityidimSpinUpStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        (2.0 * pdecSigmaValuesFD[3 * igrid] +
                         pdecSigmaValuesFD[3 * igrid + 1] +
                         2.0 * pdecSigmaValuesFD[3 * igrid + 2]) *
                        (gradValuesSpinUpFD[idim * quadWeights.size() + igrid] +
                         gradValuesSpinDownFD[idim * quadWeights.size() +
                                              igrid]);

                    pdexLapDensityidimSpinUpStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        pdexLaplacianValuesFD[2 * igrid];

                    pdecLapDensityidimSpinTotalStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        pdecLaplacianValuesFD[2 * igrid];

                    pdexGradDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        (2.0 * pdexSigmaValuesFD[3 * igrid] +
                         pdexSigmaValuesFD[3 * igrid + 1] +
                         2.0 * pdexSigmaValuesFD[3 * igrid + 2]) *
                        (gradValuesSpinUpFD[idim * quadWeights.size() + igrid] +
                         gradValuesSpinDownFD[idim * quadWeights.size() +
                                              igrid]);

                    pdecGradDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        (2.0 * pdecSigmaValuesFD[3 * igrid] +
                         pdecSigmaValuesFD[3 * igrid + 1] +
                         2.0 * pdecSigmaValuesFD[3 * igrid + 2]) *
                        (gradValuesSpinUpFD[idim * quadWeights.size() + igrid] +
                         gradValuesSpinDownFD[idim * quadWeights.size() +
                                              igrid]);

                    pdexLapDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        pdexLaplacianValuesFD[2 * igrid + 1];

                    pdecLapDensityidimSpinDownStencil
                      [igrid * d_vxcDivergenceTermFDStencilSize + istencil] =
                        pdecLaplacianValuesFD[2 * igrid + 1];
                  }
              } // stencil grid filling loop

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadWeights.size(),
              &(pdexGradDensityidimSpinUpStencil[0]),
              &(divergenceTermsPdexGradDensitySpinUp[idim][0]));

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadWeights.size(),
              &(pdecGradDensityidimSpinUpStencil[0]),
              &(divergenceTermsPdecGradDensitySpinUp[idim][0]));

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadWeights.size(),
              &(pdexGradDensityidimSpinDownStencil[0]),
              &(divergenceTermsPdexGradDensitySpinDown[idim][0]));

            utils::FiniteDifference::firstOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadWeights.size(),
              &(pdecGradDensityidimSpinDownStencil[0]),
              &(divergenceTermsPdecGradDensitySpinDown[idim][0]));


            utils::FiniteDifference::secondOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadWeights.size(),
              &(pdexLapDensityidimSpinUpStencil[0]),
              &(laplacianTermsPdexLapDensitySpinUp[idim][0]));

            /*
            utils::FiniteDifference::secondOrderDerivativeOneVariableCentral(
                    d_vxcDivergenceTermFDStencilSize,
                    d_spacingFDStencil,
                    quadWeights.size(),
                    &(pdecLapDensityidimSpinTotalStencil[0]),
                    &(laplacianTermsPdecLapDensitySpinTotal[idim][0]));
            */

            utils::FiniteDifference::secondOrderDerivativeOneVariableCentral(
              d_vxcDivergenceTermFDStencilSize,
              d_spacingFDStencil,
              quadWeights.size(),
              &(pdexLapDensityidimSpinDownStencil[0]),
              &(laplacianTermsPdexLapDensitySpinDown[idim][0]));

            /*
            utils::FiniteDifference::secondOrderDerivativeOneVariableCentral(
                    d_vxcDivergenceTermFDStencilSize,
                    d_spacingFDStencil,
                    quadWeights.size(),
                    &(pdecLapDensityidimSpinDownStencil[0]),
                    &(laplacianTermsPdecLapDensitySpinDown[idim][0]));
            */

          } // dim loop

        for (size_type igrid = 0; igrid < quadWeights.size(); igrid++)
          {
            vxValuesSpinUp[igrid] =
              pdexDensitySpinUpValues[igrid] -
              (divergenceTermsPdexGradDensitySpinTotal[0][igrid] +
               divergenceTermsPdexGradDensitySpinTotal[1][igrid] +
               divergenceTermsPdexGradDensitySpinTotal[2][igrid]) +
              (laplacianTermsPdexLapDensitySpinTotal[0][igrid] +
               laplacianTermsPdexLapDensitySpinTotal[1][igrid] +
               laplacianTermsPdexLapDensitySpinTotal[2][igrid]);

            vcValuesSpinUp[igrid] =
              pdecDensitySpinUpValues[igrid] -
              (divergenceTermsPdecGradDensitySpinTotal[0][igrid] +
               divergenceTermsPdecGradDensitySpinTotal[1][igrid] +
               divergenceTermsPdecGradDensitySpinTotal[2][igrid]) +
              (laplacianTermsPdecLapDensitySpinTotal[0][igrid] +
               laplacianTermsPdecLapDensitySpinTotal[1][igrid] +
               laplacianTermsPdecLapDensitySpinTotal[2][igrid]);


            vxValuesSpinDown[igrid] =
              pdexDensitySpinDownValues[igrid] -
              (divergenceTermsPdexGradDensitySpinTotal[0][igrid] +
               divergenceTermsPdexGradDensitySpinTotal[1][igrid] +
               divergenceTermsPdexGradDensitySpinTotal[2][igrid]) +
              (laplacianTermsPdexLapDensitySpinTotal[0][igrid] +
               laplacianTermsPdexLapDensitySpinTotal[1][igrid] +
               laplacianTermsPdexLapDensitySpinTotal[2][igrid]);

            vcValuesSpinDown[igrid] =
              pdecDensitySpinDownValues[igrid] -
              (divergenceTermsPdecGradDensitySpinTotal[0][igrid] +
               divergenceTermsPdecGradDensitySpinTotal[1][igrid] +
               divergenceTermsPdecGradDensitySpinTotal[2][igrid]) +
              (laplacianTermsPdecLapDensitySpinTotal[0][igrid] +
               laplacianTermsPdecLapDensitySpinTotal[1][igrid] +
               laplacianTermsPdecLapDensitySpinTotal[2][igrid]);
          }
      } // VxcCompute check
  }
} // namespace dftfe
