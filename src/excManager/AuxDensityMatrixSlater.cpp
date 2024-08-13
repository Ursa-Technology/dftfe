//
// Created by Arghadwip Paul.
//
#ifdef DFTFE_WITH_TORCH
#  include "AuxDensityMatrixSlater.h"
#  include <stdexcept>
#  include <cmath>
#  include <linearAlgebraOperationsCPU.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::reinitAuxDensityMatrix(
    const std::vector<std::pair<std::string, std::vector<double>>> &atomCoords,
    const std::string &auxBasisFile,
    const int          nSpin,
    const int          maxDerOrder)
  {
    d_sbs.constructBasisSet(atomCoords, auxBasisFile);
    d_nBasis      = d_sbs.getSlaterBasisSize();
    d_nSpin       = nSpin;
    d_maxDerOrder = maxDerOrder;
    d_DM.assign(d_nSpin * d_nBasis * d_nBasis, 0.0);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::evalOverlapMatrixStart(
    const std::vector<double> &quadpts,
    const std::vector<double> &quadWt)
  {
    d_sbd.evalBasisData(quadpts, d_sbs, d_maxDerOrder);
    d_SMatrix = std::vector<double>(d_nBasis * d_nBasis, 0.0);

    for (int i = 0; i < d_nBasis; ++i)
      {
        for (int j = i; j < d_nBasis; ++j)
          {
            double sum = 0.0;
            for (int iQuad = 0; iQuad < quadWt.size(); iQuad++)
              {
                sum += d_sbd.getBasisValues(iQuad * d_nBasis + i) *
                       d_sbd.getBasisValues(iQuad * d_nBasis + j) *
                       quadWt[iQuad];
              }
            d_SMatrix[i * d_nBasis + j] = sum;
            if (i != j)
              {
                d_SMatrix[j * d_nBasis + i] = sum;
              }
          }
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::evalOverlapMatrixEnd(
    const MPI_Comm &mpiComm)
  {
    // MPI All Reduce
    MPI_Allreduce(d_SMatrix.data(),
                  d_SMatrix.data(),
                  d_SMatrix.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpiComm);
    evalOverlapMatrixInv();
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::evalOverlapMatrixInv()
  {
    d_SMatrixInv = d_SMatrix;
    // Invert using Cholesky
    int info;
    // Compute the Cholesky factorization of SMatrix
    dftfe::dpotrf_("L",
                   reinterpret_cast<const unsigned int *>(&d_nBasis),
                   d_SMatrixInv.data(),
                   reinterpret_cast<const unsigned int *>(&d_nBasis),
                   &info);

    if (info != 0)
      {
        throw std::runtime_error("Error in dpotrf."
                                 "Cholesky Factorization failed."
                                 " Matrix may not be positive definite.");
      }

    // Compute the inverse of SMatrix using the Cholesky factorization
    dftfe::dpotri_("L",
                   reinterpret_cast<const unsigned int *>(&d_nBasis),
                   d_SMatrixInv.data(),
                   reinterpret_cast<const unsigned int *>(&d_nBasis),
                   &info);
    if (info != 0)
      {
        throw std::runtime_error("Error in dpotri."
                                 "Inversion using Cholesky Factors failed."
                                 "Matrix may not be positive definite.");
      }

    // Copy the upper triangular part to the lower triangular part
    for (int i = 0; i < d_nBasis; ++i)
      {
        for (int j = i + 1; j < d_nBasis; ++j)
          {
            d_SMatrixInv[j * d_nBasis + i] = d_SMatrixInv[i * d_nBasis + j];
          }
      }

    // Multiply SMatrix and SMatrixInv using dgemm_
    double              alpha = 1.0;
    double              beta  = 0.0;
    std::vector<double> CMatrix(d_nBasis * d_nBasis, 0.0);
    dftfe::dgemm_("N",
                  "N",
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &alpha,
                  d_SMatrix.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  d_SMatrixInv.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &beta,
                  CMatrix.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis));



    double frobenius_norm = 0.0;
    for (int i = 0; i < d_nBasis; ++i)
      {
        for (int j = 0; j < d_nBasis; j++)
          {
            if (i == j)
              {
                frobenius_norm += (CMatrix[i * d_nBasis + j] - 1.0) *
                                  (CMatrix[i * d_nBasis + j] - 1.0);
              }
            else
              {
                frobenius_norm +=
                  CMatrix[i * d_nBasis + j] * CMatrix[i * d_nBasis + j];
              }
          }
      }
    frobenius_norm = std::sqrt(frobenius_norm);
    if (frobenius_norm > 1E-8)
      {
        std::cout << "frobenius norm of inversion : " << frobenius_norm
                  << std::endl;
        throw std::runtime_error("SMatrix Inversion"
                                 "using Cholesky Factors failed"
                                 "to pass check!");
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  std::vector<double> &
  AuxDensityMatrixSlater<memorySpace>::getOverlapMatrixInv()
  {
    return d_SMatrixInv;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::projectDensityMatrixStart(
    std::unordered_map<std::string, std::vector<double>> &projectionInputs,
    int                                                   iSpin)
  {
    // eval <SlaterBasis|WFC> matrix d_SWFC

    d_iSpin                                    = iSpin;
    auto &psiFunc                              = projectionInputs["psiFunc"];
    auto &quadWt                               = projectionInputs["quadWt"];
    d_fValues                                  = projectionInputs["fValues"];
    std::vector<double> SlaterBasisValWeighted = d_sbd.getBasisValuesAll();

    d_nQuad = quadWt.size();
    d_nWFC  = psiFunc.size() / d_nQuad;
    d_SWFC  = std::vector<double>(d_nBasis * d_nWFC, 0.0);


    for (int i = 0; i < d_nQuad; ++i)
      {
        for (int j = 0; j < d_nBasis; ++j)
          {
            SlaterBasisValWeighted[i * d_nBasis + j] *= quadWt[i];
          }
      }

    double alpha = 1.0;
    double beta  = 0.0;
    dftfe::dgemm_("N",
                  "T",
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nWFC),
                  reinterpret_cast<const unsigned int *>(&d_nQuad),
                  &alpha,
                  SlaterBasisValWeighted.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  psiFunc.data(),
                  reinterpret_cast<const unsigned int *>(&d_nWFC),
                  &beta,
                  d_SWFC.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis));
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::projectDensityMatrixEnd(
    const MPI_Comm &mpiComm)
  {
    // MPI All Reduce d_SWFC
    MPI_Allreduce(d_SWFC.data(),
                  d_SWFC.data(),
                  d_SWFC.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpiComm);

    auto SlaterOverlapInv = this->getOverlapMatrixInv();
    // Multiply S^(-1) * <S|W> = BB
    std::vector<double> BB(d_nBasis * d_nWFC, 0.0);
    double              alpha = 1.0;
    double              beta  = 0.0;
    dftfe::dgemm_("N",
                  "N",
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nWFC),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &alpha,
                  SlaterOverlapInv.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  d_SWFC.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &beta,
                  BB.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis));

    // BF = BB * F^(1/2)
    std::vector<double> BF = BB;
    for (int i = 0; i < d_nWFC; i++)
      {
        for (int j = 0; j < d_nBasis; j++)
          {
            BF[i * d_nBasis + j] *= std::sqrt(d_fValues[i]);
          }
      }

    // D = BF * BF^T
    dftfe::dgemm_("N",
                  "T",
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  reinterpret_cast<const unsigned int *>(&d_nWFC),
                  &alpha,
                  BF.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  BF.data(),
                  reinterpret_cast<const unsigned int *>(&d_nBasis),
                  &beta,
                  d_DM.data() + d_iSpin * d_nBasis * d_nBasis,
                  reinterpret_cast<const unsigned int *>(&d_nBasis));
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

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::applyLocalOperations(
    const std::vector<double> &Points,
    std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
      &densityData)
  {
    int                       DMSpinOffset = d_nBasis * d_nBasis;
    std::pair<size_t, size_t> indexRange;

    for (int iQuad = 0; iQuad < Points.size(); iQuad++)
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
            fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes::valuesTotal],
              rhoTotal,
              indexRange);
          }

        if (densityData.find(DensityDescriptorDataAttributes::valuesSpinUp) ==
            densityData.end())
          {
            fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::valuesSpinUp],
              rhoUp,
              indexRange);
          }

        if (densityData.find(DensityDescriptorDataAttributes::valuesSpinDown) ==
            densityData.end())
          {
            fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::valuesSpinDown],
              rhoDown,
              indexRange);
          }

        indexRange = std::make_pair(iQuad * 3, iQuad * 3 + 2);

        if (densityData.find(
              DensityDescriptorDataAttributes::gradValuesSpinUp) ==
            densityData.end())
          {
            fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::gradValuesSpinUp],
              gradrhoUp,
              indexRange);
          }

        if (densityData.find(
              DensityDescriptorDataAttributes::gradValuesSpinDown) ==
            densityData.end())
          {
            fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::gradValuesSpinDown],
              gradrhoDown,
              indexRange);
          }

        indexRange = std::make_pair(iQuad * 9, iQuad * 9 + 8);
        if (densityData.find(DensityDescriptorDataAttributes::hessianSpinUp) ==
            densityData.end())
          {
            fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::hessianSpinUp],
              HessianrhoUp,
              indexRange);
          }

        // Check for hessianSpinDown attribute
        if (densityData.find(
              DensityDescriptorDataAttributes::hessianSpinDown) ==
            densityData.end())
          {
            fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::hessianSpinDown],
              HessianrhoDown,
              indexRange);
          }


        indexRange = std::make_pair(iQuad, iQuad);
        if (densityData.find(
              DensityDescriptorDataAttributes::laplacianSpinUp) ==
            densityData.end())
          {
            fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::laplacianSpinUp],
              LaplacianrhoUp,
              indexRange);
          }

        if (densityData.find(
              DensityDescriptorDataAttributes::laplacianSpinDown) ==
            densityData.end())
          {
            fillDensityAttributeData(
              densityData[DensityDescriptorDataAttributes ::laplacianSpinDown],
              LaplacianrhoDown,
              indexRange);
          }
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::projectDensityStart(
    std::unordered_map<std::string, std::vector<double>> &projectionInputs)
  {
    // projectDensity implementation
    std::cout << "Error : No implementation yet" << std::endl;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::projectDensityEnd(
    const MPI_Comm &mpiComm)
  {
    // projectDensity implementation
    std::cout << "Error : No implementation yet" << std::endl;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::getDensityMatrixComponents_occupancies(
    const std::vector<double> &occupancies) const
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  AuxDensityMatrixSlater<memorySpace>::getDensityMatrixComponents_wavefunctions(
    const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
      &eigenVectors) const
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  template class AuxDensityMatrixSlater<dftfe::utils::MemorySpace::HOST>;
#  ifdef DFTFE_WITH_DEVICE
  template class AuxDensityMatrixSlater<dftfe::utils::MemorySpace::DEVICE>;
#  endif
} // namespace dftfe
#endif
