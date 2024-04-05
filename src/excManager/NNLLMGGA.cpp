#ifdef DFTFE_WITH_TORCH
#  include <torch/script.h>
#  include <NNLLMGGA.h>
#  include <iostream>
#  include <vector>
#  include <algorithm>
#  include <iterator>
#  include <Exceptions.h>
#  include <stdexcept>
namespace dftfe
{
  namespace
  {
    struct CastToFloat
    {
      float
      operator()(double value) const
      {
        return static_cast<float>(value);
      }
    };

    struct CastToDouble
    {
      double
      operator()(float value) const
      {
        return static_cast<double>(value);
      }
    };

    void
    exc_vxc_SpinPolarized(
      const double *                       rho,
      const unsigned int                   numPoints,
      double *                             exc,
      double *                             vxc,
      torch::jit::script::Module *         model,
      const excDensityPositivityCheckTypes densityPositivityCheckType,
      const double                         rhoTol)

    void
    exc_vxc_SpinPolarized(const std::vector<double>& rho,
                          const std::vector<double>& modDRhoTotal,
                          const std::vector<double>& rhoTotalLap,
                          const std::vector<double>& w_vec,
                          const std::vector<double>& drhoTotal,
                          const std::vector<double>& rhoStencilData,
                          const std::vector<double>& FDSpacing,
                          const int stencilOrder1D,
                          const int numPoints,
                          std::vector<double>& exc,
                          std::vector<double>& vxc,
                          torch::jit::script::Module *         model,
                          const excDensityPositivityCheckTypes densityPositivityCheckType,
                          const double                         rhoTol)
    {
      std::vector<double> rhoModified(2 * numPoints, 0.0);
      if (densityPositivityCheckType ==
          excDensityPositivityCheckTypes::EXCEPTION_POSITIVE){
        for (unsigned int i = 0; i < 2 * numPoints; ++i)
          {
            std::string errMsg =
              "Negative electron-density encountered during xc evaluations";
            dftfe::utils::throwException(rho[i] > 0, errMsg);
          }
      }
      else if (densityPositivityCheckType ==
               excDensityPositivityCheckTypes::MAKE_POSITIVE){
        for (unsigned int i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] =
              std::max(rho[i], 0.0); // d_rhoTol will be added subsequently
          }
      }
      else {
        for (unsigned int i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] = rho[i];
          }
      }

      auto options = torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);
      torch::Tensor rhoInpTensor = torch::zeros({nPoints, 4}, options);

      for(int iPoint = 0; iPoint < nPoints; iPoint++){
          rhoInpTensor[iPoint][0] = rhoModified[2 * iPoint];
          rhoInpTensor[iPoint][1] = rhoModified[2 * iPoint + 1];
          rhoInpTensor[iPoint][2] = modDRhoTotal[iPoint];
          rhoInpTensor[iPoint][3] = rhoTotalLap[iPoint];
      }

      auto w_tensor = torch::from_blob(w_vec.data(), {nPoints}, torch::kDouble).clone();

      auto drhoTotalTensor = torch::from_blob(drhoTotal.data(), {nPoints * 3}, torch::kDouble).clone();
      drhoTotalTensor = drhoTotalTensor.view({nPoints, 3});


      int StencilDataSize = 3 * 2 * stencilOrder1D * 4;
      auto rhoStencilDataTensor = torch::from_blob(rhoStencilData.data(),
                                                   {nPoints * StencilDataSize},
                                                    torch::kDouble).clone();
      rhoStencilDataTensor = rhoStencilDataTensor.view({nPoints, StencilDataSize});
      rhoStencilDataTensor.set_requires_grad(true);


      auto FDspacingTensor = torch::from_blob(FDspacing.data(),
                                              {nPoints}, torch::kDouble).clone();
      FDspacingTensor = FDspacingTensor.view({-1, 1});


      auto excTensor = model->forward({rhoInp}).toTensor();
      auto vxcTensor = model->getVxc({rhoInp, drhoTotalTensor, w_tensor,
                                        rhoStencilDataTensor, FDspacingTensor});

      for (unsigned int i = 0; i < numPoints; ++i)
        {
          exc[i] = static_cast<double>(excTensor[i][0].item<double>());
          for (unsigned int j = 0; j < 2; ++j)
            vxc[2 * i + j] = static_cast<double>(vxcTensor[i][j].item<double>());
        }
    }
  } // namespace

  NNLLMGGA::NNLLMGGA(std::string                          modelFileName,
                     const bool                           isSpinPolarized /*=false*/,
                     const excDensityPositivityCheckTypes densityPositivityCheckType,
                     const double                         rhoTol)
    : d_modelFileName(modelFileName)
    , d_isSpinPolarized(isSpinPolarized)
    , d_densityPositivityCheckType(densityPositivityCheckType)
    , d_rhoTol(rhoTol)
  {
    d_model  = new torch::jit::script::Module;
    *d_model = torch::jit::load(d_modelFileName);
    // Explicitly load model onto CPU, you can use kGPU if you are on Linux
    // and have libtorch version with CUDA support (and a GPU)
    d_model->to(torch::kCPU);
  }

  NNLLMGGA::~NNLLMGGA()
  {
    delete d_model;
  }

  void
  evaluate_exc_vxc(const std::vector<double>& rho,
                   const std::vector<double>& modDRhoTotal,
                   const std::vector<double>& rhoTotalLap,
                   const std::vector<double>& w_vec,
                   const std::vector<double>& drhoTotal,
                   const std::vector<double>& rhoStencilData,
                   const std::vector<double>& FDSpacing,
                   const int stencilOrder1D,
                   const int numPoints,
                   std::vector<double>& exc,
                   std::vector<double>& vxc)
  {
    if (!d_isSpinPolarized)
       throw std::runtime_error("Spin unpolarized NN is yet to be implemented.");
    else
        exc_vxc_SpinPolarized(rho, modDRhoTotal, rhoTotalLap, w_vec, drhoTotal,
                              rhoStencilData, FDSpacing, stencilOrder1D, numPoints, exc, vxc,
                              d_model, d_densityPositivityCheckType, d_rhoTol);

  }

} // namespace dftfe
#endif