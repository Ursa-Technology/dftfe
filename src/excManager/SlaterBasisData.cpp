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
  SlaterBasisData::evalBasisData(const std::vector<Atom> &  atoms,
                                 const std::vector<double> &quadpts,
                                 const SlaterBasisSet &     sbs,
                                 int                        nQuad,
                                 int                        nBasis,
                                 int                        maxDerOrder)
  {
    if (maxDerOrder == 0)
      {
        basisValues = std::vector<double>(nQuad * nBasis, 0.0);
        this->evalBasisValues(atoms, quadpts, sbs, nBasis);
      }
    else if (maxDerOrder == 1)
      {
        basisValues     = std::vector<double>(nQuad * nBasis, 0.0);
        basisGradValues = std::vector<double>(nQuad * nBasis * 3, 0.0);
        this->evalBasisGradValues(atoms, quadpts, sbs, nBasis);
      }
    else if (maxDerOrder == 2)
      {
        basisValues     = std::vector<double>(nQuad * nBasis, 0.0);
        basisGradValues = std::vector<double>(nQuad * nBasis * 3, 0.0);
        basisHessValues = std::vector<double>(nQuad * nBasis * 9, 0.0);
        this->evalBasisHessianValues(atoms, quadpts, sbs, nBasis);
      }
    else
      {
        throw std::runtime_error("\n\n maxDerOrder should be 0, 1 or 2 \n\n");
      }
  }


  // getBasisValues implementation
  void
  SlaterBasisData::evalBasisValues(const std::vector<Atom> &  atoms,
                                   const std::vector<double> &quadpts,
                                   const SlaterBasisSet &     sbs,
                                   int                        nBasis)
  {
    auto quadpts_t = torch::tensor(quadpts, torch::dtype(torch::kDouble));
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
    for (const auto &atom : atoms)
      {
        auto center    = atom.coord;
        auto cBasisSet = sbs.getBasisSet(atom.basisfile);

        auto x_shifted = x_t - center[0];
        auto y_shifted = y_t - center[1];
        auto z_shifted = z_t - center[2];

        for (const auto &basis : cBasisSet)
          {
            auto SV =
              this->evalSlaterFunc(x_shifted, y_shifted, z_shifted, basis);
            int nQuad_t = SV.size(0);
            for (int i = 0; i < nQuad_t; ++i)
              {
                int vecIndex          = basis_ctr + i * nBasis;
                basisValues[vecIndex] = SV[i].item<double>();
              }
            basis_ctr += 1;
          }
      }
  }

  // getBasisGradValues implementation
  void
  SlaterBasisData::evalBasisGradValues(const std::vector<Atom> &  atoms,
                                       const std::vector<double> &quadpts,
                                       const SlaterBasisSet &     sbs,
                                       int                        nBasis)
  {
    auto quadpts_t = torch::tensor(quadpts, torch::dtype(torch::kDouble));
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
    for (const auto &atom : atoms)
      {
        auto center    = atom.coord;
        auto cBasisSet = sbs.getBasisSet(atom.basisfile);

        auto x_shifted = x_t - center[0];
        auto y_shifted = y_t - center[1];
        auto z_shifted = z_t - center[2];

        for (const auto &basis : cBasisSet)
          {
            auto SV =
              this->evalSlaterFunc(x_shifted, y_shifted, z_shifted, basis);
            int nQuad_t = SV.size(0);
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
  }

  // getBasisHessianValues implementation
  void
  SlaterBasisData::evalBasisHessianValues(const std::vector<Atom> &  atoms,
                                          const std::vector<double> &quadpts,
                                          const SlaterBasisSet &     sbs,
                                          int                        nBasis)
  {
    auto quadpts_t = torch::tensor(quadpts, torch::dtype(torch::kDouble));

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
    for (const auto &atom : atoms)
      {
        auto center = atom.coord; // std::vector<double> center = {0., 0., 0.};
        auto cBasisSet = sbs.getBasisSet(atom.basisfile);

        auto x_shifted = x_t - center[0];
        auto y_shifted = y_t - center[1];
        auto z_shifted = z_t - center[2];


        for (const auto &basis : cBasisSet)
          {
            auto SF =
              this->evalSlaterFunc(x_shifted, y_shifted, z_shifted, basis);
            auto SF_prime =
              torch::autograd::grad({SF},
                                    {x_t, y_t, z_t},
                                    /*grad_outputs=*/{torch::ones_like(SF)},
                                    /*retain_graph=*/c10::optional<bool>(true),
                                    /*create_graph=*/true,
                                    /*allow_unused=*/false);

            auto SFx_xyz =
              torch::autograd::grad({SF_prime[0]},
                                    {x_t, y_t, z_t},
                                    {torch::ones_like(SF_prime[0])},
                                    c10::optional<bool>(true),
                                    false,
                                    false);

            auto SFy_xyz =
              torch::autograd::grad({SF_prime[1]},
                                    {x_t, y_t, z_t},
                                    {torch::ones_like(SF_prime[1])},
                                    c10::optional<bool>(true),
                                    false,
                                    false);

            auto SFz_xyz =
              torch::autograd::grad({SF_prime[2]},
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

                basisGradValues[vecIndex_der1] = SF_prime[0][i].item<double>();
                basisGradValues[vecIndex_der1 + 1] =
                  SF_prime[1][i].item<double>();
                basisGradValues[vecIndex_der1 + 2] =
                  SF_prime[2][i].item<double>();

                basisHessValues[vecIndex_der2] = SFx_xyz[0][i].item<double>();
                basisHessValues[vecIndex_der2 + 1] =
                  SFx_xyz[1][i].item<double>();
                basisHessValues[vecIndex_der2 + 2] =
                  SFx_xyz[2][i].item<double>();
                basisHessValues[vecIndex_der2 + 3] =
                  SFy_xyz[0][i].item<double>();
                basisHessValues[vecIndex_der2 + 4] =
                  SFy_xyz[1][i].item<double>();
                basisHessValues[vecIndex_der2 + 5] =
                  SFy_xyz[2][i].item<double>();
                basisHessValues[vecIndex_der2 + 6] =
                  SFz_xyz[0][i].item<double>();
                basisHessValues[vecIndex_der2 + 7] =
                  SFz_xyz[1][i].item<double>();
                basisHessValues[vecIndex_der2 + 8] =
                  SFz_xyz[2][i].item<double>();
              }
            basis_ctr += 1;
          }
      }
  }

  //
  void
  SlaterBasisData::evalSlaterOverlapMatrix(const std::vector<double> &quadWt,
                                           int                        nQuad,
                                           int                        nBasis)
  {
    std::vector<double> SMatrix(nBasis * nBasis, 0.0);

    for (int i = 0; i < nBasis; ++i)
      {
        for (int j = i; j < nBasis; ++j)
          {
            double sum = 0.0;
            for (int iQuad = 0; iQuad < nQuad; iQuad++)
              {
                sum += basisValues[iQuad * nBasis + i] *
                       basisValues[iQuad * nBasis + j] * quadWt[iQuad];
              }
            SMatrix[i * nBasis + j] = sum;
            if (i != j)
              {
                SMatrix[j * nBasis + i] = sum;
              }
          }
      }
  }

  //
  void
  SlaterBasisData::evalSlaterOverlapMatrixInv(const std::vector<double> &quadWt,
                                              int                        nQuad,
                                              int                        nBasis)
  {
    std::vector<double> SMatrix(nBasis * nBasis, 0.0);
    SMatrixInv = std::vector<double>(nBasis * nBasis, 0.0);

    for (int i = 0; i < nBasis; ++i)
      {
        for (int j = i; j < nBasis; ++j)
          {
            double sum = 0.0;
            for (int iQuad = 0; iQuad < nQuad; iQuad++)
              {
                sum += basisValues[iQuad * nBasis + i] *
                       basisValues[iQuad * nBasis + j] * quadWt[iQuad];
              }
            SMatrix[i * nBasis + j] = sum;
            if (i != j)
              {
                SMatrix[j * nBasis + i] = sum;
              }
          }
      }

    torch::Tensor S_tensor =
      torch::from_blob(SMatrix.data(), {nBasis, nBasis}, torch::kDouble);
    auto SInv_tensor = torch::inverse(S_tensor);

    for (int i = 0; i < nBasis; ++i)
      {
        for (int j = 0; j < nBasis; ++j)
          {
            SMatrixInv[i * nBasis + j] = SInv_tensor[i][j].item<double>();
          }
      }
  }

  // evaluate Slater Function
  torch::Tensor
  SlaterBasisData::evalSlaterFunc(const torch::Tensor &  x_s,
                                  const torch::Tensor &  y_s,
                                  const torch::Tensor &  z_s,
                                  const SlaterPrimitive &basis)
  {
    int n, l, m;
    basis.nlm(n, l, m); // Retrieve the n, l, m values
    double alpha = basis.alpha();
    double nrm   = basis.normConst();

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

  void
  SlaterBasisData::printBasisInfo(int nQuad, int nBasis)
  {
    std::cout << "\n\nPrinting Basis Info : "
              << "\n\n";
    std::cout << "\n\n" << basisValues.size() << std::endl;
    int ctr = 0;
    /*
    for (const auto& val : basisValues) {
        std::cout << ctr << " " << val << "\n";
        ctr +=1;
    }*/

    /*
    std::cout << "Printing Basis Derivative Info : " << "\n\n";
    int d_ctr = 0;
    int derComp = 1; // 0 for der w.r.t. x, 1 for y, 2 for z

    for (const auto& val : basisGradValues) {
        if ( (d_ctr % (nBasis * 3)) == 0)
            std::cout << "\n ----- New Point ----- \n";
        // std::cout << (d_ctr % 3) << " " << val << "\n";
        if (d_ctr % 3 == derComp)
            std::cout << (d_ctr % 3) << " " << val << "\n";
        d_ctr +=1;
    }

    std::cout << "\n\nPrinting Basis Double Derivative Info : " << "\n\n";
    int dd_ctr = 0;
    int dderComp_1 = 2;
    int dderComp_2 = 2;

    for (const auto& val : basisHessValues) {
        if ( (dd_ctr % (nBasis * 9)) == 0)
            std::cout << "\n ----- New Point ----- \n";
        // std::cout << (dd_ctr % 9) << " " << val << "\n";
        if (dd_ctr % 9 == (3 * dderComp_1 + dderComp_2))
            std::cout << (dd_ctr % 9) << " " << val << "\n";
        dd_ctr +=1;
    } */
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
  SlaterBasisData::getSlaterOverlapMatrixInv()
  {
    return SMatrixInv;
  }


  std::vector<double>
  SlaterBasisData::getBasisValuesAll()
  {
    return basisValues;
  }
} // namespace dftfe
#endif
