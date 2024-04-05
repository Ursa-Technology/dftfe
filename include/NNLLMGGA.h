#ifndef NNLLMGGA_H
#define NNLLMGGA_H
#ifdef DFTFE_WITH_TORCH
#  include <string>
#  include <torch/torch.h>
#  include <excDensityPositivityCheckTypes.h>
namespace dftfe
{
  class NNLLMGGA
  {
  public:
    NNLLMGGA(std::string                          modelFileName,
            const bool                           isSpinPolarized = false,
            const excDensityPositivityCheckTypes densityPositivityCheckType =
                    excDensityPositivityCheckTypes::MAKE_POSITIVE,
            const double rhoTol = 1.0e-8);
    ~NNLLMGGA();

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
                    std::vector<double>& vxc);


  private:
    std::string                          d_modelFileName;
    torch::jit::script::Module *         d_model;
    const bool                           d_isSpinPolarized;
    const double                         d_rhoTol;
    const excDensityPositivityCheckTypes d_densityPositivityCheckType;
  };
} // namespace dftfe
#endif
#endif // NNLLMGGA_H