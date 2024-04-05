#include <excDensityLLMGGAClass.h>
#include <NNLLMGGA.h>
#include <BaseXCHandler.h>
#include <cmath>

namespace dftfe
{
    excDensityLLMGGAClass::excDensityLLMGGAClass(xc_func_type *funcXPtr,
                                                 xc_func_type *funcCPtr,
                                                 bool          isSpinPolarized,
                                                 bool          scaleExchange,
                                                 bool          computeCorrelation,
                                                 double        scaleExchangeFactor)
            : excDensityBaseClass(isSpinPolarized)
    {
        d_familyType = densityFamilyType::GGA;
        d_funcXPtr   = funcXPtr;
        d_funcCPtr   = funcCPtr;
#ifdef DFTFE_WITH_TORCH
        d_NNLDAPtr = nullptr;
#endif
    }

    excDensityLLMGGAClass::excDensityLLMGGAClass(xc_func_type *funcXPtr,
                                                 xc_func_type *funcCPtr,
                                                 bool          isSpinPolarized,
                                                 std::string   modelXCInputFile,
                                                 bool          scaleExchange,
                                                 bool          computeCorrelation,
                                                 double        scaleExchangeFactor)
            : excDensityBaseClass(isSpinPolarized)
    {
        d_familyType = densityFamilyType::GGA;
        d_funcXPtr   = funcXPtr;
        d_funcCPtr   = funcCPtr;
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
    computeDensityBased_exc_vxc_FD(
            const AuxDensityMatrixSlater &       auxDM,
            const  int                           nPoints,
            const  int                           stencilOrder1D,
            std::vector<double> &                exc,
            std::vector<double> &                vxc)
    {
        std::vector<double> rho(nPoints * 2, 0.0);
        std::vector<double> sigma(nPoints * 3, 0.0);
        std::vector<double> drho(nPoints * 6, 0.0);
        std::vector<double> ddrho(nPoints * 18, 0.0);

        std::vector<double> drhoTotal(nPoints * 3, 0.0);
        std::vector<double> ddrhoTotal(nPoints * 9, 0.0);

        std::vector<double> modDRhoTotal(nPoints, 0.0);
        std::vector<double> rhoTotalLap(nPoints, 0.0);

        std::vector<double> v_vec(3, 0.0);
        std::vector<double> w_vec(nPoints, 0.0);

        std::vector<double> exc_Base(nPoints, 0.0);
        std::vector<double> vxc_Base(nPoints * 2, 0.0);



        auto options = torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);
        torch::Tensor rhoInpTensor = torch::zeros({nPoints, 4}, options);


        for(int iPoint = 0; iPoint < nPoints; iPoint++){
            auto valuesSpinUp = auxDM.getLocalDescriptors(DensityDescriptorDataAttributes::valuesSpinUp,
                                                          std::make_pair(iPoint, iPoint + 1));
            rho.insert(rho.end(), valuesSpinUp.begin(), valuesSpinUp.end());

            auto valuesSpinDown = auxDM.getLocalDescriptors(DensityDescriptorDataAttributes::valuesSpinDown,
                                                            std::make_pair(iPoint, iPoint + 1));
            rho.insert(rho.end(), valuesSpinDown.begin(), valuesSpinDown.end());

            auto gradValuesSpinUp = auxDM.getLocalDescriptors(DensityDescriptorDataAttributes::gradValueSpinUp,
                                                              std::make_pair(iPoint * 3, iPoint * 3 + 3));
            drho.insert(drho.end(), gradValuesSpinUp.begin(), gradValuesSpinUp.end());

            auto gradValuesSpinDown = auxDM.getLocalDescriptors(DensityDescriptorDataAttributes::gradValueSpinDown,
                                                                std::make_pair(iPoint * 3, iPoint * 3 + 3));
            drho.insert(drho.end(), gradValuesSpinDown.begin(), gradValuesSpinDown.end());

            auto sigmaValues = auxDM.getLocalDescriptors(DensityDescriptorDataAttributes::sigma,
                                                         std::make_pair(iPoint * 3, iPoint * 3 + 3));
            sigma.insert(sigma.end(), sigmaValues.begin(), sigmaValues.end());

            auto hessianSpinUp = auxDM.getLocalDescriptors(DensityDescriptorDataAttributes::hessianSpinUp,
                                                             std::make_pair(iPoint * 9, iPoint * 9 + 9));
            ddrho.insert(ddrho.end(), hessianSpinUp.begin(), hessianSpinUp.end());

            auto hessianSpinDown = auxDM.getLocalDescriptors(DensityDescriptorDataAttributes::hessianSpinDown,
                                                               std::make_pair(iPoint * 9, iPoint * 9 + 9));
            ddrho.insert(ddrho.end(), hessianSpinDown.begin(), hessianSpinDown.end());

            double sumofSquaresdrhoTotal = 0.0;
            for (int i = 0; i < 3; i++) {
                drhoTotal[iPoint * 3 + i] = gradValuesSpinUp[i] + gradValuesSpinDown[i];
                sumofSquaresdrhoTotal += drhoTotal[iPoint * 3 + i] * drhoTotal[iPoint * 3 + i];
            }
            modDRhoTotal[iPoint] = std::sqrt(sumofSquaresdrhoTotal);

            for (int i = 0; i < 9; i++)
                ddrhoTotal[iPoint * 9 + i] = hessianSpinUp[i] + hessianSpinDown[i];
            rhoTotalLap[iPoint]  = ddrhoTotal[iPoint * 9] + ddrhoTotal[iPoint * 9 + 4] +
                                   ddrhoTotal[iPoint * 9 + 8];

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    v_vec[i] += ddrhoTotal[iPoint * 9 + i * 3 + j] * drhoTotal[iPoint * 3 + j];
                }
            }

            for (int i = 0; i < 3; i++) {
                w_vec[iPoint] += (v_vec[i] * drhoTotal[iPoint * 3 + i]);
            }
            w_vec[iPoint] = w_vec[iPoint]/modDRhoTotal[iPoint];

            rhoInpTensor[iPoint][0] = valuesSpinUp[iPoint];
            rhoInpTensor[iPoint][1] = valuesSpinDown[iPoint];
            rhoInpTensor[iPoint][2] = modDRhoTotal[iPoint];
            rhoInpTensor[iPoint][3] = rhoTotalLap[iPoint];
        }

        auto FDSpacing = auxDM.getLocalDescriptors(DensityDescriptorDataAttributes::FDSpacing,
                                                    std::make_pair(0, nPoints));

        int StencilDataSize = 3 * 2 * stencilOrder1D * 4;
        auto rhoStencilData = auxDM.getLocalDescriptors(DensityDescriptorDataAttributes::rhoStencilData,
                                                        std::make_pair(0, nPoints * StencilDataSize));



        // Base model
        BaseXCHandler GGAHandler("GGA_X_PBE", "GGA_C_PBE", "polarized");
        GGAHandler.getexcAndvxc(rho, drho, ddrho, sigma, exc_Base, vxc_Base, nPoints);

        // NN model
    #ifdef DFTFE_WITH_TORCH
        if (d_NNLLMGGAPtr != nullptr)
        {
        std::vector<double> exc_NN(nPoints, 0.0);
        std::vector<double> vxc_NN(nPoints * 2, 0.0);
        d_NNLLMGGAPtr->evaluate_exc_vxc(rho, modDRhoTotal, rhoTotalLap, w_vec, drhoTotal,
                                        rhoStencilData, FDSpacing, stencilOrder1D, numPoints,
                                        exc_NN, vxc_NN);
        for (unsigned int i = 0; i < nPoints; i++) {
          exc[i]         = exc_Base[i] + exc_NN[i];
          vxc[2 * i]     = vxc_Base[2 * i] + vxc_NN[2 * i];
          vxc[2 * i + 1] = vxc_Base[2 * i + 1] + vxc_NN[2 * i + 1];
          }
        }
    #endif
    }
} // namespace dftfe