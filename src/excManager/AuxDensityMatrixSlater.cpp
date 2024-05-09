//
// Created by Arghadwip Paul.
//

#include "AuxDensityMatrixSlater.h"
#include <stdexcept>
#include <lapacke.h>

namespace dftfe
{
  AuxDensityMatrixSlater::AuxDensityMatrixSlater(
    const std::map<int, std::string> &atom_coords,
    const std::vector<double> &       quadpts,
    const std::vector<double> &       quadWt,
    const std::string                 atomBasisNameFile,
    const int                         nQuad,
    const int                         nSpin,
    const int                         maxDerOrder)
    : AuxDensityMatrix()
    , // call the base class constructor
    d_quadpts(quadpts)
    , d_quadWt(quadWt)
    ,
    , d_nQuad(nQuad)
    , d_nSpin(nSpin)
    , d_maxDerOrder(maxDerOrder)
  {
    // ------------------ Read AtomicCoords_Slater --------------
    d_atoms = Atom::readCoordFile(atomBasisNameFile);

    // ------------------ SlaterBasisSets -------------
    d_sbs.constructBasisSet(d_atoms);
    d_nBasis = d_sbs.getTotalBasisSize(d_atoms);
    std::cout << "nBasis : " << d_nBasis << std::endl;

    d_DM.assign(nSpin * d_nBasis * d_nBasis, 0.0);

    d_sbd.evalBasisData(
      d_atoms, d_quadpts, d_sbs, d_nQuad, d_nBasis, d_maxDerOrder);
  }

  AuxDensityMatrixSlater::~AuxDensityMatrixSlater()
  {
    // Destructor implementation
  }

  void
  AuxDensityMatrixSlater::setDMzero()
  {
    // setDMzero implementation
    for (auto &element : d_DM)
      {
        element = 0.0;
      }
  }

  namespace
  {
    void
    fillDensityAttributeData(std::vector<double> &            attributeData,
                             const std::vector<double> &      values,
                             const std::pair<size_t, size_t> &indexRange)
    {
      size_t startIndex = indexRange.first;
      size_t endIndex   = indexRange.second;

      if (startIndex > endIndex || endIndex >= attributeData.size() ||
          endIndex >= values.size())
        {
          throw std::invalid_argument("Invalid index range for densityData");
        }

      for (size_t i = startIndex; i <= endIndex; ++i)
        {
          attributeData[i] = values[i];
        }
    }
  } // namespace

  void
  AuxDensityMatrixSlater::applyLocalOperations(
    const std::vector<double> &                                     Points,
    std::map<DensityDescriptorDataAttributes, std::vector<double>> &densityData)
  {
    int                       DMSpinOffset = d_nBasis * d_nBasis;
    std::pair<size_t, size_t> indexRange;

    for (int iQuad = 0; iQuad < Qpts.size(); iQuad++)
      {
        std::vector<double> rhoUp(1, 0.0);
        std::vector<double> rhoDown(1, 0.0);
        std::vector<double> rhoTotal(1, 0.0);
        std::vector<double> gradrhoUp(3, 0.0);
        std::vector<double> gradrhoDown(3, 0.0);
        std::vector<double> HessianrhoUp(9, 0.0);
        std::vector<double> HessianrhoDown(9, 0.0);
        std::vector<double> LaplacianrhoUp(1, 0.0);
        std::vector<double> LaplacianrhoDown(1, 0.0);

        for (int i = 0; i < d_nBasis; i++)
          {
            for (int j = 0; j < d_nBasis; j++)
              {
                for (int iSpin = 0; iSpin < d_nSpin; iSpin++)
                  {
                    if (iSpin == 0)
                      {
                        rhoUp[0] += d_DM[i * d_nBasis + j] *
                                    d_sbd.getBasisValues(iQuad * d_nBasis + i) *
                                    d_sbd.getBasisValues(iQuad * d_nBasis + j);

                        for (int derIndex = 0; derIndex < 3; derIndex++)
                          {
                            gradrhoUp[derIndex] +=
                              d_DM[i * d_nBasis + j] *
                              (d_sbd.getBasisGradValues(iQuad * d_nBasis +
                                                        3 * i + derIndex) *
                                 d_sbd.getBasisValues(iQuad * d_nBasis + j) +
                               d_sbd.getBasisGradValues(iQuad * d_nBasis +
                                                        3 * j + derIndex) *
                                 d_sbd.getBasisValues(iQuad * d_nBasis + i));
                          }

                        for (int derIndex1 = 0; derIndex1 < 3; derIndex1++)
                          {
                            for (int derIndex2 = 0; derIndex2 < 3; derIndex2++)
                              {
                                HessianrhoUp[derIndex1 * 3 + derIndex2] +=
                                  d_DM[i * d_nBasis + j] *
                                  (d_sbd.getBasisHessianValues(
                                     iQuad * d_nBasis + 9 * i + 3 * derIndex1 +
                                     derIndex2) *
                                     d_sbd.getBasisValues(iQuad * d_nBasis +
                                                          j) +
                                   d_sbd.getBasisGradValues(iQuad * d_nBasis +
                                                            3 * i + derIndex1) *
                                     d_sbd.getBasisGradValues(
                                       iQuad * d_nBasis + 3 * j + derIndex2) +
                                   d_sbd.getBasisGradValues(iQuad * d_nBasis +
                                                            3 * i + derIndex2) *
                                     d_sbd.getBasisGradValues(
                                       iQuad * d_nBasis + 3 * j + derIndex1) +
                                   d_sbd.getBasisValues(iQuad * d_nBasis + i) *
                                     d_sbd.getBasisHessianValues(
                                       iQuad * d_nBasis + 9 * j +
                                       3 * derIndex1 + derIndex2));
                              }
                          }
                      }

                    if (iSpin == 1)
                      {
                        rhoDown[0] +=
                          d_DM[DMSpinOffset + i * d_nBasis + j] *
                          d_sbd.getBasisValues(iQuad * d_nBasis + i) *
                          d_sbd.getBasisValues(iQuad * d_nBasis + j);

                        for (int derIndex = 0; derIndex < 3; derIndex++)
                          {
                            gradrhoDown[derIndex] +=
                              d_DM[DMSpinOffset + i * d_nBasis + j] *
                              (d_sbd.getBasisGradValues(iQuad * d_nBasis +
                                                        3 * i + derIndex) *
                                 d_sbd.getBasisValues(iQuad * d_nBasis + j) +
                               d_sbd.getBasisGradValues(iQuad * d_nBasis +
                                                        3 * j + derIndex) *
                                 d_sbd.getBasisValues(iQuad * d_nBasis + i));
                          }

                        for (int derIndex1 = 0; derIndex1 < 3; derIndex1++)
                          {
                            for (int derIndex2 = 0; derIndex2 < 3; derIndex2++)
                              {
                                HessianrhoDown[derIndex1 * 3 + derIndex2] +=
                                  d_DM[DMSpinOffset + i * d_nBasis + j] *
                                  (d_sbd.getBasisHessianValues(
                                     iQuad * d_nBasis + 9 * i + 3 * derIndex1 +
                                     derIndex2) *
                                     d_sbd.getBasisValues(iQuad * d_nBasis +
                                                          j) +
                                   d_sbd.getBasisGradValues(iQuad * d_nBasis +
                                                            3 * i + derIndex1) *
                                     d_sbd.getBasisGradValues(
                                       iQuad * d_nBasis + 3 * j + derIndex2) +
                                   d_sbd.getBasisGradValues(iQuad * d_nBasis +
                                                            3 * i + derIndex2) *
                                     d_sbd.getBasisGradValues(
                                       iQuad * d_nBasis + 3 * j + derIndex1) +
                                   d_sbd.getBasisValues(iQuad * d_nBasis + i) *
                                     d_sbd.getBasisHessianValues(
                                       iQuad * d_nBasis + 9 * j +
                                       3 * derIndex1 + derIndex2));
                              }
                          }
                      }
                  }
              }
          }

        rhoTotal[0]       = rhoUp[0] + rhoDown[0];
        LaplacianrhoUp[0] = HessianrhoUp[0] + HessianrhoUp[4] + HessianrhoUp[8];
        LaplacianrhoDown[0] =
          HessianrhoDown[0] + HessianrhoDown[4] + HessianrhoDown[8];


        indexRange = std::make_pair(iQuad, iQuad);
        if (densityData.find(DensityDescriptorDataAttributes::valuesTotal) ==
            densityData.end())
          {
            this->fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes::valuesTotal],
              rhoTotal,
              indexRange);
          }

        if (densityData.find(DensityDescriptorDataAttributes::valuesSpinUp) ==
            densityData.end())
          {
            this->fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::valuesSpinUp],
              rhoUp,
              indexRange);
          }

        if (densityData.find(DensityDescriptorDataAttributes::valuesSpinDown) ==
            densityData.end())
          {
            this->fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::valuesSpinDown],
              rhoDown,
              indexRange);
          }

        indexRange = std::make_pair(iQuad * 3, iQuad * 3 + 2);

        if (densityData.find(
              DensityDescriptorDataAttributes::gradValueSpinUp) ==
            densityData.end())
          {
            this->fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::gradValueSpinUp],
              gradrhoUp,
              indexRange);
          }

        if (densityData.find(
              DensityDescriptorDataAttributes::gradValueSpinDown) ==
            densityData.end())
          {
            this->fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::gradValueSpinDown],
              gradrhoDown,
              indexRange);
          }

        indexRange = std::make_pair(iQuad * 9, iQuad * 9 + 8);
        if (densityData.find(DensityDescriptorDataAttributes::hessianSpinUp) ==
            densityData.end())
          {
            this->fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::hessianSpinUp],
              HessianrhoUp,
              indexRange);
          }

        // Check for hessianSpinDown attribute
        if (densityData.find(
              DensityDescriptorDataAttributes::hessianSpinDown) ==
            densityData.end())
          {
            this->fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::hessianSpinDown],
              HessianrhoDown,
              indexRange);
          }


        indexRange = std::make_pair(iQuad, iQuad);
        if (densityData.find(
              DensityDescriptorDataAttributes::laplacianSpinUp) ==
            densityData.end())
          {
            this->fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::laplacianSpinUp],
              LaplacianrhoUp,
              indexRange);
          }

        if (densityData.find(
              DensityDescriptorDataAttributes::laplacianSpinDown) ==
            densityData.end())
          {
            this->fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::laplacianSpinDown],
              LaplacianrhoDown,
              indexRange);
          }
      }
  }

  void
  AuxDensityMatrixSlater::projectDensityMatrix(
    const std::vector<double> &Qpts,
    const std::vector<double> &QWt,
    const int                  nQ,
    const std::vector<double> &psiFunc,
    const std::vector<double> &fValues,
    const std::pair<int, int>  nPsi,
    double                     alpha,
    double                     beta)
  {
    /*
     * Qpts - x1, y1, z1, x2, y2, z2, ...
     * QWt  - quadWt1, quadWt2, ....
     * nQ   - number of quad points
     * psiFunc - FE eigenfunction, \psi_{sigma, quadpt, basisIndex}
     * fValues - FE eigenValues, f_{sigma, basisIndex}
     * nPsi   - pair of <num of eigenfunctions up, num of eigenfunctions down>
     * alpha, beta - DM = alpha * DM + beta * DM_dash (BLAS format)
     */


    // Check if QPts are same as quadpts, then update quadpts, qwt, nQuad
    /*
    if(Qpts.size() != d_quadWt.size()){
        d_quadpts = Qpts;
        d_quadWt = QWt;
        d_nQuad  = nQ;
        d_sbd.evalBasisData(d_atoms, d_quadpts, d_sbs, d_nQuad, d_nBasis,
    d_maxDerOrder); d_sbd.evalSlaterOverlapMatrixInv(d_quadWt, d_nQuad,
    d_nBasis);
    }
    else if( (std::equal(Qpts.begin(), Qpts.end(), d_quadWt.begin())) == 0){
        d_quadpts = Qpts;
        d_quadWt = QWt;
        d_nQuad  = nQ;
        d_sbd.evalBasisData(d_atoms, d_quadpts, d_sbs, d_nQuad, d_nBasis,
    d_maxDerOrder); d_sbd.evalSlaterOverlapMatrixInv(d_quadWt, d_nQuad,
    d_nBasis);
    } */

    int nPsi1   = nPsi.first;
    int nPsi2   = nPsi.second;
    int nPsiTot = nPsi1 + nPsi2;

    auto SlaterOverlapInv =
      this->evalSlaterOverlapMatrixInv(quadWt, nQuad, nBasis);

    std::vector<double> SlaterBasisVal = d_sbd.getBasisValuesAll();

    // stores AA = Slater_Phi^T * FE_EigVec = (Ns * Nq) * (Nq * N_eig) = (Ns *
    // N_eig) Slater_Phi^T is quadrature weighted

    std::vector<double> AA(d_nBasis * nPsiTot, 0.0);

    std::vector<double> AA1(d_nBasis * nPsi1, 0.0);
    std::vector<double> AA2(d_nBasis * nPsi2, 0.0);

    std::vector<double> SlaterBasisValWeighted = SlaterBasisVal;
    for (int i = 0; i < d_nQuad; ++i)
      {
        for (int j = 0; j < d_nBasis; ++j)
          {
            SlaterBasisValWeighted[i * d_nBasis + j] *= d_quadWt[i];
          }
      }

    // Perform matrix multiplication: C[:, :Ne1] = A^T * B1
    cblas_dgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                d_nBasis,
                nPsi1,
                d_nQuad,
                1.0,
                SlaterBasisValWeighted.data(),
                d_nBasis,
                psiFunc.data(),
                nPsi1,
                0.0,
                AA1.data(),
                nPsi1);

    // Perform matrix multiplication: C[:, Ne1:] = A^T * B2
    cblas_dgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                d_nBasis,
                nPsi2,
                d_nQuad,
                1.0,
                SlaterBasisValWeighted.data(),
                d_nBasis,
                psiFunc.data() + d_nQuad * nPsi1,
                nPsi2,
                0.0,
                AA2.data(),
                nPsi2);

    // stores BB = S^(-1) * AA   // (Ns * N_eig)
    std::vector<double> BB1(d_nBasis * nPsi1, 0.0);
    std::vector<double> BB2(d_nBasis * nPsi2, 0.0);

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                d_nBasis,
                nPsi1,
                d_nBasis,
                1.0,
                SlaterOverlapInv.data(),
                d_nBasis,
                AA1.data(),
                nPsi1,
                0.0,
                BB1.data(),
                nPsi1);

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                d_nBasis,
                nPsi2,
                d_nBasis,
                1.0,
                SlaterOverlapInv.data(),
                d_nBasis,
                AA2.data(),
                nPsi2,
                0.0,
                BB2.data(),
                nPsi2);

    // BF = BB * F    // (Ns * Neig)
    std::vector<double> BF1 = BB1;
    std::vector<double> BF2 = BB2;

    for (int i = 0; i < d_nBasis; i++)
      {
        for (int j = 0; j < nPsi1; j++)
          {
            BF1[i * nPsi1 + j] *= fValues[j];
          }
        for (int j = 0; j < nPsi2; j++)
          {
            BF2[i * nPsi2 + j] *= fValues[nPsi1 + j];
          }
      }

    std::vector<double> DM_dash(d_nSpin * d_nBasis * d_nBasis, 0.0);

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                d_nBasis,
                d_nBasis,
                nPsi1,
                1.0,
                BF1.data(),
                nPsi1,
                BB1.data(),
                nPsi1,
                0.0,
                DM_dash.data(),
                d_nBasis);

    // Perform matrix multiplication: C[:, Ne1:] = A^T * B2
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                d_nBasis,
                d_nBasis,
                nPsi2,
                1.0,
                BF2.data(),
                nPsi2,
                BB2.data(),
                nPsi2,
                0.0,
                DM_dash.data() + d_nBasis * d_nBasis,
                d_nBasis);

    for (int i = 0; i < d_DM.size(); i++)
      {
        d_DM[i] = alpha * d_DM[i] + beta * DM_dash[i];
      }

    std::cout << "DM1:" << std::endl;
    for (int i = 0; i < d_nBasis; ++i)
      {
        for (int j = 0; j < d_nBasis; ++j)
          {
            std::cout << DM_dash[i * d_nBasis + j] << " ";
          }
        std::cout << std::endl;
      }

    std::cout << "DM2:" << std::endl;
    for (int i = 0; i < d_nBasis; ++i)
      {
        for (int j = 0; j < d_nBasis; ++j)
          {
            std::cout << DM_dash[d_nBasis * d_nBasis + i * d_nBasis + j] << " ";
          }
        std::cout << std::endl;
      }
  }


  std::vector<double>
  AuxDensityMatrixSlater::evalSlaterOverlapMatrix(
    const std::vector<double> &quadWt,
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
                sum += getBasisValues(iQuad * nBasis + i) *
                       getbasisValues(iQuad * nBasis + j) * quadWt[iQuad];
              }
            SMatrix[i * nBasis + j] = sum;
            if (i != j)
              {
                SMatrix[j * nBasis + i] = sum;
              }
          }
      }
    return SMatrix;
  }

  std::vector<double>
  SlaterBasisData::evalSlaterOverlapMatrixInv(const std::vector<double> &quadWt,
                                              int                        nQuad,
                                              int                        nBasis)
  {
    auto SMatrix = this->evalSlaterOverlapMatrix(quadWt, nQuad, nBasis);

    std::vector<double> SMatrixInv(SMatrix);

    // Invert the matrix using Cholesky decomposition
    int info;
    info =
      LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', nBasis, invMatrix.data(), nBasis);
    if (info != 0)
      {
        throw std::runtime_error(
          "Matrix inversion failed. Matrix may not be positive definite.");
      }

    // Fill the upper triangular part of the inverted matrix
    for (int i = 0; i < nBasis; ++i)
      {
        for (int j = i + 1; j < nBasis; ++j)
          {
            SMatrixInv[i * nBasis + j] = SMatrixInv[j * nBasis + i];
          }
      }
    return SMatrixInv;
  }

  void
  AuxDensityMatrixSlater::projectDensity()
  {
    // projectDensity implementation
    std::cout << "Error : No implementation yet" << std::endl;
  }
} // namespace dftfe
