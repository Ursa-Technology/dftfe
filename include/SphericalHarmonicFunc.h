//
// Created by Arghadwip Paul.
//

#ifndef SLATER2_SPHERICALHARMONICFUNC_H
#define SLATER2_SPHERICALHARMONICFUNC_H

#include <torch/torch.h>

double Dm(const int m);
double Clm(const int l, const int absm);
torch::Tensor Rn(const int n, const double alpha, const torch::Tensor& r);
torch::Tensor Qm(const int m,  const torch::Tensor& phi);
torch::Tensor associatedLegendre(const int l, const int absm, const torch::Tensor& x);

#endif //SLATER2_SPHERICALHARMONICFUNC_H
