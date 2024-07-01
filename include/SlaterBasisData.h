//
// Created by Arghadwip Paul on 2/15/24.
//

#ifndef DFTFE_SLATER_SLATERBASISDATA_H
#define DFTFE_SLATER_SLATERBASISDATA_H

#ifdef DFTFE_WITH_TORCH

#  include <vector>
#  include <memory>
#  include <torch/torch.h>
#  include "SlaterBasisSet.h"
#  include "SphericalHarmonicFunc.h"

namespace dftfe
{
  class SlaterBasisData
  {
  public:
    // Public Member functions declarations
    void
    evalBasisData(const std::vector<double> &quadpts,
                  const SlaterBasisSet &     sbs,
                  int                        maxDerOrder);

    double
    getBasisValues(const int index);

    std::vector<double>
    getBasisValuesAll();

    double
    getBasisGradValues(const int index);

    double
    getBasisHessianValues(const int index);


  private:
    // Member variables
    std::vector<double> basisValues;
    std::vector<double> basisGradValues;
    std::vector<double> basisHessValues;

    void
    evalBasisValues(const std::vector<double> &quadpts,
                    const SlaterBasisSet &     sbs);

    void
    evalBasisGradValues(const std::vector<double> &quadpts,
                        const SlaterBasisSet &     sbs);

    void
    evalBasisHessianValues(const std::vector<double> &quadpts,
                           const SlaterBasisSet &     sbs);

    torch::Tensor
    evalSlaterFunc(const torch::Tensor &  x_s,
                   const torch::Tensor &  y_s,
                   const torch::Tensor &  z_s,
                   const SlaterPrimitive *sp);
  };
} // namespace dftfe

#endif
#endif // DFTFE_SLATER_SLATERBASISDATA_H
