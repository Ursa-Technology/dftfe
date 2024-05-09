//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_SLATER_SPHERICALHARMONICFUNC_H
#define DFTFE_SLATER_SPHERICALHARMONICFUNC_H

#ifdef DFTFE_WITH_TORCH

#  include <torch/torch.h>

namespace dftfe
{
  double
  Dm(const int m);
  double
  Clm(const int l, const int absm);
  torch::Tensor
  Rn(const int n, const double alpha, const torch::Tensor &r);
  torch::Tensor
  Qm(const int m, const torch::Tensor &phi);
  torch::Tensor
  associatedLegendre(const int l, const int absm, const torch::Tensor &x);

} // namespace dftfe

#endif
#endif // DFTFE_SLATER_SPHERICALHARMONICFUNC_H
