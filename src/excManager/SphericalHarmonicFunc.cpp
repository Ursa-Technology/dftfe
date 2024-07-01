//
// Created by Arghadwip Paul.
//

#ifdef DFTFE_WITH_TORCH

#  include <cmath>
#  include <vector>
#  include <cassert>
#  include <torch/torch.h>
#  include "SphericalHarmonicFunc.h"

namespace dftfe
{
  double
  Dm(const int m)
  {
    if (m == 0)
      return 1.0 / sqrt(2 * M_PI);
    else
      return 1.0 / sqrt(M_PI);
  }

  double
  Clm(const int l, const int absm)
  {
    assert(absm <= l);
    double a = (2.0 * l + 1.0) * tgamma(l - absm + 1);
    double b = 2.0 * tgamma(l + absm + 1);
    return sqrt(a / b);
  }

  torch::Tensor
  Rn(const int n, const double alpha, const torch::Tensor &r)
  {
    if (n == 1)
      return torch::exp(-alpha * r); // torch
    else
      return (torch::pow(r, n - 1) * torch::exp(-alpha * r)); // torch
  }

  torch::Tensor
  Qm(const int m, const torch::Tensor &phi)
  {
    if (m > 0)
      {
        return torch::cos(m * phi); // torch
      }
    else if (m == 0)
      {
        return torch::ones_like(phi); // torch
      }
    else
      {
        return torch::sin(abs(m) * phi); // torch
      }
  }

  torch::Tensor
  associatedLegendre(const int l, const int absm, const torch::Tensor &x)
  {
    assert(absm >= 0 && absm <= l);

    // Check if any value is outside the [-1, 1] range
    if ((x < -1.0).any().item<bool>() || (x > 1.0).any().item<bool>())
      {
        throw std::runtime_error(
          "The argument to associated legendre must be in [-1,1]");
      }

    auto cxM   = torch::ones_like(x);
    auto somx2 = torch::sqrt(1.0 - x * x);

    double fact = 1.0;
    for (int i = 0; i < absm; ++i)
      {
        cxM = -cxM * fact * somx2;
        fact += 2.0;
      }

    auto cx = cxM;

    if (absm != l)
      {
        auto cxMPlus1 = x * (2 * absm + 1) * cxM;
        cx            = cxMPlus1;

        auto cxPrev     = cxMPlus1;
        auto cxPrevPrev = cxM;

        for (int i = absm + 2; i <= l; ++i)
          {
            cx = ((2 * i - 1) * x * cxPrev + (-i - absm + 1) * cxPrevPrev) /
                 (i - absm);
            cxPrevPrev = cxPrev;
            cxPrev     = cx;
          }
      }

    return (pow(-1.0, absm) * cx);
  }
} // namespace dftfe
#endif
