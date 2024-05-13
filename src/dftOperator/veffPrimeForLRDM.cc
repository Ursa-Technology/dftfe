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
// @author Sambit Das, Nikhil Kodali
//
#include <KohnShamHamiltonianOperator.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  KohnShamHamiltonianOperator<memorySpace>::computeVEffPrime(
      AuxDensityMatrix auxDensityRepresentation,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoPrimeValues,
      const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &gradRhoPrimeValues,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &                phiPrimeValues,
      const unsigned int spinIndex)
  {
    /*
    const bool isGGA =
      d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA;
    const unsigned int spinPolarizedFactor = 1 + d_dftParamsPtr->spinPolarized;
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureID);
    const unsigned int totalLocallyOwnedCells =
      d_basisOperationsPtrHost->nCells();
    const unsigned int numberQuadraturePoints =
      d_basisOperationsPtrHost->nQuadsPerCell();
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffJxWHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_invJacderExcWithSigmaTimesGradRhoJxWHost;
#else
    auto &d_VeffJxWHost = d_VeffJxW;
    auto &d_invJacderExcWithSigmaTimesGradRhoJxWHost =
      d_invJacderExcWithSigmaTimesGradRhoJxW;
#endif
    d_VeffJxWHost.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_invJacderExcWithSigmaTimesGradRhoJxWHost.resize(
      isGGA ? totalLocallyOwnedCells * numberQuadraturePoints * 3 : 0, 0.0);
    auto dot3 = [](const double *a, const double *b) {
      double sum = 0.0;
      for (unsigned int i = 0; i < 3; i++)
        {
          sum += a[i] * b[i];
        }
      return sum;
    };

    if (spinPolarizedFactor == 1)
      {
        std::vector<double> densityValue(numberQuadraturePoints);
        std::vector<double> gradDensityValue(
          isGGA ? 3 * numberQuadraturePoints : 0);
        std::vector<double> densityPrimeValue(numberQuadraturePoints);
        std::vector<double> gradDensityPrimeValue(
          isGGA ? 3 * numberQuadraturePoints : 0);

        std::vector<double> sigmaValue(isGGA ? numberQuadraturePoints : 0);
        std::vector<double> derExchEnergyWithSigmaVal(
          isGGA ? numberQuadraturePoints : 0);
        std::vector<double> derCorrEnergyWithSigmaVal(
          isGGA ? numberQuadraturePoints : 0);
        std::vector<double> der2ExchEnergyWithSigmaVal(
          isGGA ? numberQuadraturePoints : 0);
        std::vector<double> der2CorrEnergyWithSigmaVal(
          isGGA ? numberQuadraturePoints : 0);
        std::vector<double> der2ExchEnergyWithDensitySigmaVal(
          isGGA ? numberQuadraturePoints : 0);
        std::vector<double> der2CorrEnergyWithDensitySigmaVal(
          isGGA ? numberQuadraturePoints : 0);
        std::vector<double> derExchEnergyWithDensityVal(
          isGGA ? numberQuadraturePoints : 0);
        std::vector<double> derCorrEnergyWithDensityVal(
          isGGA ? numberQuadraturePoints : 0);
        std::vector<double> der2ExchEnergyWithDensityVal(
          numberQuadraturePoints);
        std::vector<double> der2CorrEnergyWithDensityVal(
          numberQuadraturePoints);

        //
        // loop over cell block
        //
        for (unsigned int iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
          {
            std::memcpy(densityValue.data(),
                        rhoValues[0].data() + iCell * numberQuadraturePoints,
                        numberQuadraturePoints * sizeof(double));
            if (isGGA)
              std::memcpy(gradDensityValue.data(),
                          gradRhoValues[0].data() +
                            iCell * numberQuadraturePoints * 3,
                          3 * numberQuadraturePoints * sizeof(double));

            std::memcpy(densityPrimeValue.data(),
                        rhoPrimeValues[0].data() +
                          iCell * numberQuadraturePoints,
                        numberQuadraturePoints * sizeof(double));
            if (isGGA)
              std::memcpy(gradDensityPrimeValue.data(),
                          gradRhoPrimeValues[0].data() +
                            iCell * numberQuadraturePoints * 3,
                          3 * numberQuadraturePoints * sizeof(double));

            if (d_dftParamsPtr->nonLinearCoreCorrection)
              {
                std::transform(densityValue.data(),
                               densityValue.data() + numberQuadraturePoints,
                               rhoCoreValues
                                 .find(d_basisOperationsPtrHost->cellID(iCell))
                                 ->second.data(),
                               densityValue.data(),
                               std::plus<>{});
                if (isGGA)
                  std::transform(
                    gradDensityValue.data(),
                    gradDensityValue.data() + 3 * numberQuadraturePoints,
                    gradRhoCoreValues
                      .find(d_basisOperationsPtrHost->cellID(iCell))
                      ->second.data(),
                    gradDensityValue.data(),
                    std::plus<>{});
              }

            if (isGGA)
              for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                   ++iQuad)
                sigmaValue[iQuad] = dot3(gradDensityValue.data() + 3 * iQuad,
                                         gradDensityValue.data() + 3 * iQuad);

            std::map<rhoDataAttributes, const std::vector<double> *> rhoData;

            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerExchangeEnergy;
            std::map<VeffOutputDataAttributes, std::vector<double> *>
              outputDerCorrEnergy;
            std::map<fxcOutputDataAttributes, std::vector<double> *>
              outputDer2ExchangeEnergy;
            std::map<fxcOutputDataAttributes, std::vector<double> *>
              outputDer2CorrEnergy;


            rhoData[rhoDataAttributes::values] = &densityValue;
            if (isGGA)
              rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;

            if (isGGA)
              {
                outputDerExchangeEnergy
                  [VeffOutputDataAttributes::derEnergyWithDensity] =
                    &derExchEnergyWithDensityVal;
                outputDerExchangeEnergy
                  [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                    &derExchEnergyWithSigmaVal;

                outputDerCorrEnergy
                  [VeffOutputDataAttributes::derEnergyWithDensity] =
                    &derCorrEnergyWithDensityVal;
                outputDerCorrEnergy
                  [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                    &derCorrEnergyWithSigmaVal;
              }
            outputDer2ExchangeEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensity] =
                &der2ExchEnergyWithDensityVal;
            if (isGGA)
              {
                outputDer2ExchangeEnergy
                  [fxcOutputDataAttributes::der2EnergyWithDensitySigma] =
                    &der2ExchEnergyWithDensitySigmaVal;
                outputDer2ExchangeEnergy
                  [fxcOutputDataAttributes::der2EnergyWithSigma] =
                    &der2ExchEnergyWithSigmaVal;
              }
            outputDer2CorrEnergy
              [fxcOutputDataAttributes::der2EnergyWithDensity] =
                &der2CorrEnergyWithDensityVal;
            if (isGGA)
              {
                outputDer2CorrEnergy
                  [fxcOutputDataAttributes::der2EnergyWithDensitySigma] =
                    &der2CorrEnergyWithDensitySigmaVal;
                outputDer2CorrEnergy
                  [fxcOutputDataAttributes::der2EnergyWithSigma] =
                    &der2CorrEnergyWithSigmaVal;
              }

            if (isGGA)
              d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
                numberQuadraturePoints,
                rhoData,
                outputDerExchangeEnergy,
                outputDerCorrEnergy);

            d_excManagerPtr->getExcDensityObj()->computeDensityBasedFxc(
              numberQuadraturePoints,
              rhoData,
              outputDer2ExchangeEnergy,
              outputDer2CorrEnergy);


            auto cellJxWPtr = d_basisOperationsPtrHost->JxWBasisData().data() +
                              iCell * numberQuadraturePoints;

            for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                 ++iQuad)
              {
                double sigmaDensityMixedDerTerm = 0.0;
                if (isGGA)
                  {
                    const double gradRhoDotGradRhoPrime =
                      dot3(gradDensityValue.data() + 3 * iQuad,
                           gradDensityPrimeValue.data() + 3 * iQuad);

                    // 2.0*del2{exc}/del{sigma}{rho}*\dot{gradrho^{\prime},gradrho}
                    sigmaDensityMixedDerTerm =
                      2.0 *
                      (der2ExchEnergyWithDensitySigmaVal[iQuad] +
                       der2CorrEnergyWithDensitySigmaVal[iQuad]) *
                      gradRhoDotGradRhoPrime;
                  }

                d_VeffJxWHost[numberQuadraturePoints * iCell + iQuad] =
                  (phiPrimeValues[iCell * numberQuadraturePoints + iQuad] +
                   (der2ExchEnergyWithDensityVal[iQuad] +
                    der2CorrEnergyWithDensityVal[iQuad]) *
                     densityPrimeValue[iQuad] +
                   sigmaDensityMixedDerTerm) *
                  cellJxWPtr[iQuad];
              }


            if (isGGA)
              for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                   ++iQuad)
                {
                  const double jxw = cellJxWPtr[iQuad];
                  const double gradRhoDotGradRhoPrime =
                    dot3(gradDensityValue.data() + 3 * iQuad,
                         gradDensityPrimeValue.data() + 3 * iQuad);

                  std::vector<double> term(3, 0.0);
                  term[0] = derExchEnergyWithSigmaVal[iQuad] +
                            derCorrEnergyWithSigmaVal[iQuad];
                  term[1] = der2ExchEnergyWithSigmaVal[iQuad] +
                            der2CorrEnergyWithSigmaVal[iQuad];
                  term[2] = der2ExchEnergyWithDensitySigmaVal[iQuad] +
                            der2CorrEnergyWithDensitySigmaVal[iQuad];

                  std::vector<double> tempvec(3, 0.0);
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    tempvec[iDim] =
                      (term[0] * gradDensityPrimeValue[3 * iQuad + iDim] +
                       2.0 * term[1] * gradRhoDotGradRhoPrime *
                         gradDensityValue[3 * iQuad + iDim] +
                       term[2] * densityPrimeValue[iQuad] *
                         gradDensityValue[3 * iQuad + iDim]) *
                      jxw;
                  if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                           iCell * numberQuadraturePoints * 9 + iQuad * 9 :
                           iCell * 9);
                      for (unsigned jDim = 0; jDim < 3; ++jDim)
                        for (unsigned iDim = 0; iDim < 3; ++iDim)
                          d_invJacderExcWithSigmaTimesGradRhoJxWHost
                            [iCell * numberQuadraturePoints * 3 + iQuad * 3 +
                             iDim] += inverseJacobiansQuadPtr[3 * jDim + iDim] *
                                      tempvec[jDim];
                    }
                  else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
                    {
                      const double *inverseJacobiansQuadPtr =
                        d_basisOperationsPtrHost->inverseJacobiansBasisData()
                          .data() +
                        iCell * 3;
                      for (unsigned iDim = 0; iDim < 3; ++iDim)
                        d_invJacderExcWithSigmaTimesGradRhoJxWHost
                          [iCell * numberQuadraturePoints * 3 + iQuad * 3 +
                           iDim] =
                            inverseJacobiansQuadPtr[iDim] * tempvec[iDim];
                    }
                }
          } // cell loop
      }
    else
      {
        std::vector<double> densityValue(2 * numberQuadraturePoints);
        std::vector<double> gradDensityValue(
          isGGA ? 6 * numberQuadraturePoints : 0);
        std::vector<double> densityPrimeValue(2 * numberQuadraturePoints);
        std::vector<double> gradDensityPrimeValue(
          isGGA ? 6 * numberQuadraturePoints : 0);

        std::vector<double> derExchEnergyWithDensityVal(2 *
                                                        numberQuadraturePoints);
        std::vector<double> derCorrEnergyWithDensityVal(2 *
                                                        numberQuadraturePoints);
        std::vector<double> derExchEnergyWithSigma(
          isGGA ? 3 * numberQuadraturePoints : 0);
        std::vector<double> derCorrEnergyWithSigma(
          isGGA ? 3 * numberQuadraturePoints : 0);
        std::vector<double> sigmaValue(isGGA ? 3 * numberQuadraturePoints : 0);

        const double lambda = 1e-2;
        for (unsigned int iCellQuad = 0;
             iCellQuad < totalLocallyOwnedCells * numberQuadraturePoints;
             ++iCellQuad)
          d_VeffJxWHost[iCellQuad] =
            phiPrimeValues[iCellQuad] *
            d_basisOperationsPtrHost->JxWBasisData()[iCellQuad];
        std::transform(phiPrimeValues.begin(),
                       phiPrimeValues.end(),
                       d_basisOperationsPtrHost->JxWBasisData().begin(),
                       d_VeffJxWHost.begin(),
                       std::multiplies<>{});

        auto computeXCPerturbedDensity = [&](double densityPerturbCoeff,
                                             double veffCoeff) {
          //
          // loop over cell block
          //
          for (unsigned int iCell = 0; iCell < totalLocallyOwnedCells; ++iCell)
            {
              const double *cellRhoValues =
                rhoValues[0].data() + iCell * numberQuadraturePoints;
              const double *cellMagValues =
                rhoValues[1].data() + iCell * numberQuadraturePoints;
              for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                   ++iQuad)
                {
                  const double rhoByTwo       = cellRhoValues[iQuad] / 2.0;
                  const double magByTwo       = cellMagValues[iQuad] / 2.0;
                  densityValue[2 * iQuad]     = rhoByTwo + magByTwo;
                  densityValue[2 * iQuad + 1] = rhoByTwo - magByTwo;
                }
              if (isGGA)
                {
                  const double *cellGradRhoValues =
                    gradRhoValues[0].data() +
                    3 * iCell * numberQuadraturePoints;
                  const double *cellGradMagValues =
                    gradRhoValues[1].data() +
                    3 * iCell * numberQuadraturePoints;
                  for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      {
                        const double gradRhoByTwo =
                          cellGradRhoValues[3 * iQuad + iDim] / 2.0;
                        const double gradMagByTwo =
                          cellGradMagValues[3 * iQuad + iDim] / 2.0;
                        gradDensityValue[6 * iQuad + iDim] =
                          gradRhoByTwo + gradMagByTwo;
                        gradDensityValue[6 * iQuad + 3 + iDim] =
                          gradRhoByTwo - gradMagByTwo;
                      }
                }


              if (d_dftParamsPtr->nonLinearCoreCorrection)
                {
                  const std::vector<double> &temp2 =
                    rhoCoreValues.find(d_basisOperationsPtrHost->cellID(iCell))
                      ->second;
                  for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    {
                      densityValue[2 * iQuad] += temp2[iQuad] / 2.0;
                      densityValue[2 * iQuad + 1] += temp2[iQuad] / 2.0;
                    }
                  if (isGGA)
                    {
                      const std::vector<double> &temp3 =
                        gradRhoCoreValues
                          .find(d_basisOperationsPtrHost->cellID(iCell))
                          ->second;
                      for (unsigned int iQuad = 0;
                           iQuad < numberQuadraturePoints;
                           ++iQuad)
                        for (unsigned int iDim = 0; iDim < 3; ++iDim)
                          {
                            gradDensityValue[6 * iQuad + iDim] +=
                              temp3[3 * iQuad + iDim] / 2.0;
                            gradDensityValue[6 * iQuad + iDim + 3] +=
                              temp3[3 * iQuad + iDim] / 2.0;
                          }
                    }
                }


              const double *cellRhoPrimeValues =
                rhoPrimeValues[0].data() + iCell * numberQuadraturePoints;
              const double *cellMagPrimeValues =
                rhoPrimeValues[1].data() + iCell * numberQuadraturePoints;
              for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                   ++iQuad)
                {
                  const double rhoByTwo = cellRhoPrimeValues[iQuad] / 2.0;
                  const double magByTwo = cellMagPrimeValues[iQuad] / 2.0;
                  densityPrimeValue[2 * iQuad]     = rhoByTwo + magByTwo;
                  densityPrimeValue[2 * iQuad + 1] = rhoByTwo - magByTwo;
                }
              if (isGGA)
                {
                  const double *cellGradRhoPrimeValues =
                    gradRhoPrimeValues[0].data() +
                    3 * iCell * numberQuadraturePoints;
                  const double *cellGradMagPrimeValues =
                    gradRhoPrimeValues[1].data() +
                    3 * iCell * numberQuadraturePoints;
                  for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                       ++iQuad)
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      {
                        const double gradRhoByTwo =
                          cellGradRhoPrimeValues[3 * iQuad + iDim] / 2.0;
                        const double gradMagByTwo =
                          cellGradMagPrimeValues[3 * iQuad + iDim] / 2.0;
                        gradDensityPrimeValue[6 * iQuad + iDim] =
                          gradRhoByTwo + gradMagByTwo;
                        gradDensityPrimeValue[6 * iQuad + 3 + iDim] =
                          gradRhoByTwo - gradMagByTwo;
                      }
                }
              std::transform(densityValue.begin(),
                             densityValue.end(),
                             densityPrimeValue.begin(),
                             densityValue.begin(),
                             [&densityPerturbCoeff](auto &a, auto &b) {
                               return a + densityPerturbCoeff * b;
                             });
              if (isGGA)
                std::transform(gradDensityValue.begin(),
                               gradDensityValue.end(),
                               gradDensityPrimeValue.begin(),
                               gradDensityValue.begin(),
                               [&densityPerturbCoeff](auto &a, auto &b) {
                                 return a + densityPerturbCoeff * b;
                               });

              if (isGGA)
                for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                     ++iQuad)
                  {
                    sigmaValue[3 * iQuad] =
                      dot3(gradDensityValue.data() + 6 * iQuad,
                           gradDensityValue.data() + 6 * iQuad);
                    sigmaValue[3 * iQuad + 1] =
                      dot3(gradDensityValue.data() + 6 * iQuad,
                           gradDensityValue.data() + 6 * iQuad + 3);
                    sigmaValue[3 * iQuad + 2] =
                      dot3(gradDensityValue.data() + 6 * iQuad + 3,
                           gradDensityValue.data() + 6 * iQuad + 3);
                  }

              std::map<rhoDataAttributes, const std::vector<double> *> rhoData;


              std::map<VeffOutputDataAttributes, std::vector<double> *>
                outputDerExchangeEnergy;
              std::map<VeffOutputDataAttributes, std::vector<double> *>
                outputDerCorrEnergy;


              rhoData[rhoDataAttributes::values] = &densityValue;

              outputDerExchangeEnergy
                [VeffOutputDataAttributes::derEnergyWithDensity] =
                  &derExchEnergyWithDensityVal;

              outputDerCorrEnergy
                [VeffOutputDataAttributes::derEnergyWithDensity] =
                  &derCorrEnergyWithDensityVal;
              if (isGGA)
                {
                  rhoData[rhoDataAttributes::sigmaGradValue] = &sigmaValue;
                  outputDerExchangeEnergy
                    [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                      &derExchEnergyWithSigma;
                  outputDerCorrEnergy
                    [VeffOutputDataAttributes::derEnergyWithSigmaGradDensity] =
                      &derCorrEnergyWithSigma;
                }
              d_excManagerPtr->getExcDensityObj()->computeDensityBasedVxc(
                numberQuadraturePoints,
                rhoData,
                outputDerExchangeEnergy,
                outputDerCorrEnergy);

              auto cellJxWPtr =
                d_basisOperationsPtrHost->JxWBasisData().data() +
                iCell * numberQuadraturePoints;

              for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                   ++iQuad)
                {
                  d_VeffJxWHost[numberQuadraturePoints * iCell + iQuad] +=
                    veffCoeff *
                    (derExchEnergyWithDensityVal[2 * iQuad + spinIndex] +
                     derCorrEnergyWithDensityVal[2 * iQuad + spinIndex]) *
                    cellJxWPtr[iQuad];
                }
              if (isGGA)
                if (d_basisOperationsPtrHost->cellsTypeFlag() != 2)
                  {
                    for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                         ++iQuad)
                      {
                        const double *inverseJacobiansQuadPtr =
                          d_basisOperationsPtrHost->inverseJacobiansBasisData()
                            .data() +
                          (d_basisOperationsPtrHost->cellsTypeFlag() == 0 ?
                             iCell * numberQuadraturePoints * 9 + iQuad * 9 :
                             iCell * 9);
                        const double *gradDensityQuadPtr =
                          gradDensityValue.data() + 6 * iQuad + 3 * spinIndex;
                        const double *gradDensityOtherQuadPtr =
                          gradDensityValue.data() + 6 * iQuad +
                          3 * (1 - spinIndex);
                        const double term =
                          (derExchEnergyWithSigma[3 * iQuad + 2 * spinIndex] +
                           derCorrEnergyWithSigma[3 * iQuad + 2 * spinIndex]) *
                          cellJxWPtr[iQuad];
                        const double termoff =
                          (derExchEnergyWithSigma[3 * iQuad + 1] +
                           derCorrEnergyWithSigma[3 * iQuad + 1]) *
                          cellJxWPtr[iQuad];
                        for (unsigned jDim = 0; jDim < 3; ++jDim)
                          for (unsigned iDim = 0; iDim < 3; ++iDim)
                            d_invJacderExcWithSigmaTimesGradRhoJxWHost
                              [iCell * numberQuadraturePoints * 3 + iQuad * 3 +
                               iDim] +=
                              inverseJacobiansQuadPtr[3 * jDim + iDim] *
                              (2.0 * gradDensityQuadPtr[jDim] * term +
                               gradDensityOtherQuadPtr[jDim] * termoff) *
                              veffCoeff;
                      }
                  }
                else if (d_basisOperationsPtrHost->cellsTypeFlag() == 2)
                  {
                    for (unsigned int iQuad = 0; iQuad < numberQuadraturePoints;
                         ++iQuad)
                      {
                        const double *inverseJacobiansQuadPtr =
                          d_basisOperationsPtrHost->inverseJacobiansBasisData()
                            .data() +
                          iCell * 3;
                        const double *gradDensityQuadPtr =
                          gradDensityValue.data() + 6 * iQuad + 3 * spinIndex;
                        const double *gradDensityOtherQuadPtr =
                          gradDensityValue.data() + 6 * iQuad +
                          3 * (1 - spinIndex);
                        const double term =
                          (derExchEnergyWithSigma[3 * iQuad + 2 * spinIndex] +
                           derCorrEnergyWithSigma[3 * iQuad + 2 * spinIndex]) *
                          cellJxWPtr[iQuad];
                        const double termoff =
                          (derExchEnergyWithSigma[3 * iQuad + 1] +
                           derCorrEnergyWithSigma[3 * iQuad + 1]) *
                          cellJxWPtr[iQuad];
                        for (unsigned iDim = 0; iDim < 3; ++iDim)
                          d_invJacderExcWithSigmaTimesGradRhoJxWHost
                            [iCell * numberQuadraturePoints * 3 + iQuad * 3 +
                             iDim] +=
                            veffCoeff * inverseJacobiansQuadPtr[iDim] *
                            (2.0 * gradDensityQuadPtr[iDim] * term +
                             gradDensityOtherQuadPtr[iDim] * termoff);
                      }
                  }
            } // cell loop
        };
        computeXCPerturbedDensity(2.0 * lambda, -1.0 / 12.0 / lambda);
        computeXCPerturbedDensity(lambda, 2.0 / 3.0 / lambda);
        computeXCPerturbedDensity(-2.0 * lambda, 1.0 / 12.0 / lambda);
        computeXCPerturbedDensity(-lambda, -2.0 / 3.0 / lambda);
      }


#if defined(DFTFE_WITH_DEVICE)
    d_VeffJxW.resize(d_VeffJxWHost.size());
    d_VeffJxW.copyFrom(d_VeffJxWHost);
    d_invJacderExcWithSigmaTimesGradRhoJxW.resize(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost.size());
    d_invJacderExcWithSigmaTimesGradRhoJxW.copyFrom(
      d_invJacderExcWithSigmaTimesGradRhoJxWHost);
#endif
    */
  }
  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
  template class KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
