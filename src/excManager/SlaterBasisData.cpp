//
// Created by Arghadwip Paul.
//
#ifdef DFTFE_WITH_TORCH
#  include <iostream>
#  include <torch/torch.h>
#  include <typeinfo>
#  include "SlaterBasisData.h"

namespace dftfe
{
  void
  SlaterBasisData::evalBasisData(const std::vector<double> &quadpts,
                                 const SlaterBasisSet &     sbs,
                                 int                        maxDerOrder)
  {
    int nQuad  = quadpts.size() / 3;
    int nBasis = sbs.getSlaterBasisSize();

    switch (maxDerOrder)
      {
        case 0:
          basisValues = std::vector<double>(nQuad * nBasis, 0.0);
          this->evalBasisValues(quadpts, sbs);
          break;

        case 1:
          basisValues     = std::vector<double>(nQuad * nBasis, 0.0);
          basisGradValues = std::vector<double>(nQuad * nBasis * 3, 0.0);
          this->evalBasisGradValues(quadpts, sbs);
          break;

        case 2:
          basisValues     = std::vector<double>(nQuad * nBasis, 0.0);
          basisGradValues = std::vector<double>(nQuad * nBasis * 3, 0.0);
          basisHessValues = std::vector<double>(nQuad * nBasis * 9, 0.0);
          this->evalBasisHessianValues(quadpts, sbs);
          break;

        default:
          throw std::runtime_error("\n\n maxDerOrder should be 0, 1 or 2 \n\n");
      }
  }


  // getBasisValues implementation
  void
  SlaterBasisData::evalBasisValues(const std::vector<double> &quadpts,
                                   const SlaterBasisSet &     sbs)
  {
    int  nBasis    = sbs.getSlaterBasisSize();
    auto quadpts_t = torch::tensor(quadpts, torch::dtype(torch::kDouble));
    const std::vector<SlaterBasisInfo> &basisInfo = sbs.getSlaterBasisInfo();

    // Slice the tensor to create x_t, y_t, z_t
    auto x_t = quadpts_t
                 .index({torch::indexing::Slice(torch::indexing::None,
                                                torch::indexing::None,
                                                3)})
                 .clone();
    auto y_t =
      quadpts_t.index({torch::indexing::Slice(1, torch::indexing::None, 3)})
        .clone();
    auto z_t =
      quadpts_t.index({torch::indexing::Slice(2, torch::indexing::None, 3)})
        .clone();

    int basis_ctr = 0;
    for (const auto &info : basisInfo)
      {
        auto center          = info.center;
        auto slaterPrimitive = info.sp;

        auto x_shifted = x_t - center[0];
        auto y_shifted = y_t - center[1];
        auto z_shifted = z_t - center[2];

        auto SV      = this->evalSlaterFunc(x_shifted,
                                       y_shifted,
                                       z_shifted,
                                       slaterPrimitive);
        int  nQuad_t = SV.size(0);
        for (int i = 0; i < nQuad_t; ++i)
          {
            int vecIndex          = basis_ctr + i * nBasis;
            basisValues[vecIndex] = SV[i].item<double>();
          }
        basis_ctr += 1;
      }
  }

  // getBasisGradValues implementation
  void
  SlaterBasisData::evalBasisGradValues(const std::vector<double> &quadpts,
                                       const SlaterBasisSet &     sbs)
  {
    int  nBasis    = sbs.getSlaterBasisSize();
    auto quadpts_t = torch::tensor(quadpts, torch::dtype(torch::kDouble));
    const std::vector<SlaterBasisInfo> &basisInfo = sbs.getSlaterBasisInfo();
    // Slice the tensor to create x_t, y_t, z_t
    auto x_t = quadpts_t
                 .index({torch::indexing::Slice(torch::indexing::None,
                                                torch::indexing::None,
                                                3)})
                 .clone()
                 .set_requires_grad(true);
    auto y_t =
      quadpts_t.index({torch::indexing::Slice(1, torch::indexing::None, 3)})
        .clone()
        .set_requires_grad(true);
    auto z_t =
      quadpts_t.index({torch::indexing::Slice(2, torch::indexing::None, 3)})
        .clone()
        .set_requires_grad(true);

    int basis_ctr = 0;
    for (const auto &info : basisInfo)
      {
        auto center          = info.center;
        auto slaterPrimitive = info.sp;

        auto x_shifted = x_t - center[0];
        auto y_shifted = y_t - center[1];
        auto z_shifted = z_t - center[2];

        auto SV      = this->evalSlaterFunc(x_shifted,
                                       y_shifted,
                                       z_shifted,
                                       slaterPrimitive);
        int  nQuad_t = SV.size(0);
        SV.backward(torch::ones_like(SV));
        for (int i = 0; i < nQuad_t; ++i)
          {
            int vecIndex_1                  = basis_ctr + i * nBasis;
            int vecIndex_2                  = (basis_ctr + i * nBasis) * 3;
            basisValues[vecIndex_1]         = SV[i].item<double>();
            basisGradValues[vecIndex_2]     = x_t.grad()[i].item<double>();
            basisGradValues[vecIndex_2 + 1] = y_t.grad()[i].item<double>();
            basisGradValues[vecIndex_2 + 2] = z_t.grad()[i].item<double>();
          }
        x_t.grad().zero_();
        y_t.grad().zero_();
        z_t.grad().zero_();

        basis_ctr += 1;
      }
  }

  // getBasisHessianValues implementation
  void
  SlaterBasisData::evalBasisHessianValues(const std::vector<double> &quadpts,
                                          const SlaterBasisSet &     sbs)
  {
    int  nBasis    = sbs.getSlaterBasisSize();
    auto quadpts_t = torch::tensor(quadpts, torch::dtype(torch::kDouble));
    const std::vector<SlaterBasisInfo> &basisInfo = sbs.getSlaterBasisInfo();

    // Slice the tensor to create x_t, y_t, z_t
    auto x_t = quadpts_t
                 .index({torch::indexing::Slice(torch::indexing::None,
                                                torch::indexing::None,
                                                3)})
                 .clone()
                 .set_requires_grad(true);
    auto y_t =
      quadpts_t.index({torch::indexing::Slice(1, torch::indexing::None, 3)})
        .clone()
        .set_requires_grad(true);

    auto z_t =
      quadpts_t.index({torch::indexing::Slice(2, torch::indexing::None, 3)})
        .clone()
        .set_requires_grad(true);

    int basis_ctr = 0;
    for (const auto &info : basisInfo)
      {
        auto center          = info.center;
        auto slaterPrimitive = info.sp;

        auto x_shifted = x_t - center[0];
        auto y_shifted = y_t - center[1];
        auto z_shifted = z_t - center[2];

        auto SF = this->evalSlaterFunc(x_shifted,
                                       y_shifted,
                                       z_shifted,
                                       slaterPrimitive);
        auto SF_prime =
          torch::autograd::grad({SF},
                                {x_t, y_t, z_t},
                                /*grad_outputs=*/{torch::ones_like(SF)},
                                /*retain_graph=*/c10::optional<bool>(true),
                                /*create_graph=*/true,
                                /*allow_unused=*/false);

        auto SFx_xyz = torch::autograd::grad({SF_prime[0]},
                                             {x_t, y_t, z_t},
                                             {torch::ones_like(SF_prime[0])},
                                             c10::optional<bool>(true),
                                             false,
                                             false);

        auto SFy_xyz = torch::autograd::grad({SF_prime[1]},
                                             {x_t, y_t, z_t},
                                             {torch::ones_like(SF_prime[1])},
                                             c10::optional<bool>(true),
                                             false,
                                             false);

        auto SFz_xyz = torch::autograd::grad({SF_prime[2]},
                                             {x_t, y_t, z_t},
                                             {torch::ones_like(SF_prime[2])},
                                             c10::optional<bool>(false),
                                             false,
                                             false);


        int nQuad_t = SF.size(0);
        for (int i = 0; i < nQuad_t; ++i)
          {
            int vecIndex_der0 = basis_ctr + i * nBasis;
            int vecIndex_der1 = (basis_ctr + i * nBasis) * 3;
            int vecIndex_der2 = (basis_ctr + i * nBasis) * 9;

            basisValues[vecIndex_der0] = SF[i].item<double>();

            basisGradValues[vecIndex_der1]     = SF_prime[0][i].item<double>();
            basisGradValues[vecIndex_der1 + 1] = SF_prime[1][i].item<double>();
            basisGradValues[vecIndex_der1 + 2] = SF_prime[2][i].item<double>();

            basisHessValues[vecIndex_der2]     = SFx_xyz[0][i].item<double>();
            basisHessValues[vecIndex_der2 + 1] = SFx_xyz[1][i].item<double>();
            basisHessValues[vecIndex_der2 + 2] = SFx_xyz[2][i].item<double>();
            basisHessValues[vecIndex_der2 + 3] = SFy_xyz[0][i].item<double>();
            basisHessValues[vecIndex_der2 + 4] = SFy_xyz[1][i].item<double>();
            basisHessValues[vecIndex_der2 + 5] = SFy_xyz[2][i].item<double>();
            basisHessValues[vecIndex_der2 + 6] = SFz_xyz[0][i].item<double>();
            basisHessValues[vecIndex_der2 + 7] = SFz_xyz[1][i].item<double>();
            basisHessValues[vecIndex_der2 + 8] = SFz_xyz[2][i].item<double>();
          }
        basis_ctr += 1;
      }
  }

  // evaluate Slater Function
  torch::Tensor
  SlaterBasisData::evalSlaterFunc(const torch::Tensor &  x_s,
                                  const torch::Tensor &  y_s,
                                  const torch::Tensor &  z_s,
                                  const SlaterPrimitive *sp)
  {
    int    n     = sp->n;
    int    l     = sp->l;
    int    m     = sp->m;
    double alpha = sp->alpha;

    double t1  = std::pow(2.0 * alpha, n + 0.5);
    double t2  = std::sqrt(std::tgamma(2 * n + 1));
    double nrm = t1 / t2;

    auto r     = torch::sqrt(x_s * x_s + y_s * y_s + z_s * z_s);
    auto theta = torch::acos(z_s / r);
    auto phi   = torch::atan2(y_s, x_s);

    int  absm     = abs(m);
    auto cosTheta = torch::cos(theta);
    auto C        = Clm(l, absm) * Dm(m);
    auto R        = Rn(n, alpha, r);
    auto P        = associatedLegendre(l, absm, cosTheta);
    auto Q        = Qm(m, phi);

    auto SF = nrm * C * R * P * Q;

    return SF;
  }

  double
  SlaterBasisData::getBasisValues(const int index)
  {
    if (index < 0 || index >= basisValues.size())
      {
        throw std::out_of_range(
          "Index is outside the range of the basisValues vector.");
      }

    return basisValues[index];
  }

  double
  SlaterBasisData::getBasisGradValues(const int index)
  {
    if (index < 0 || index >= basisGradValues.size())
      {
        throw std::out_of_range(
          "Index is outside the range of the basisValues vector.");
      }

    return basisGradValues[index];
  }

  double
  SlaterBasisData::getBasisHessianValues(const int index)
  {
    if (index < 0 || index >= basisHessValues.size())
      {
        throw std::out_of_range(
          "Index is outside the range of the basisValues vector.");
      }

    return basisHessValues[index];
  }

  std::vector<double>
  SlaterBasisData::getBasisValuesAll()
  {
    return basisValues;
  }
} // namespace dftfe
#endif
