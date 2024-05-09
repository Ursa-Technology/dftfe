//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_AUXDM_AUXDENSITYMATRIX_H
#define DFTFE_AUXDM_AUXDENSITYMATRIX_H

#include <vector>
#include <utility>

namespace dftfe
{
  enum class DensityDescriptorDataAttributes
  {
    valuesTotal,
    valuesSpinUp,
    valuesSpinDown,
    gradValueSpinUp,
    gradValueSpinDown,
    hessianSpinUp,
    hessianSpinDown,
    laplacianSpinUp,
    laplacianSpinDown
  };

  class AuxDensityMatrix
  {
  public:
    // Constructor
    AuxDensityMatrix();

    // Virtual destructor
    virtual ~AuxDensityMatrix();

    // Pure virtual functions

    virtual void
    applyLocalOperations(const std::vector<double> &    Points,
                         std::map<DensityDescriptorDataAttributes,
                                  std::vector<double>> &densityData) = 0;

    virtual void
    projectDensityMatrix(const std::vector<double> &Qpts,
                         const std::vector<double> &QWt,
                         const int                  nQ,
                         const std::vector<double> &psiFunc,
                         const std::vector<double> &fValues,
                         const std::pair<int, int>  nPsi,
                         double                     alpha,
                         double                     beta) = 0;

    virtual void
    projectDensity() = 0;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYMATRIX_H
