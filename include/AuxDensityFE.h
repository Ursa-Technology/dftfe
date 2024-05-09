//
// Created by Sambit Das.
//

#ifndef DFTFE_AUXDM_AUXDENSITYFE_H
#define DFTFE_AUXDM_AUXDENSITYFE_H

#include <vector>
#include <utility>

namespace dftfe
{
  class AuxDensityFE : public AuxDensity
  {
  public:
    // Constructor
    AuxDensityFE();


    void
    applyLocalOperations(const std::vector<double> &    Points,
                         std::map<DensityDescriptorDataAttributes,
                                  std::vector<double>> &densityData) override;


    void
    projectDensityMatrix(const std::vector<double> &Qpts,
                         const std::vector<double> &QWt,
                         const int                  nQ,
                         const std::vector<double> &psiFunc,
                         const std::vector<double> &fValues,
                         const std::pair<int, int>  nPsi,
                         double                     alpha,
                         double                     beta) override;

    void
    projectDensity(const std::vector<double> &Qpts,
                   const std::vector<double> &QWt,
                   const int                  nQ,
                   const std::vector<double> &densityVals,
                   const std::vector<double> &gradDensityVals) override;

  private:
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYFE_H
