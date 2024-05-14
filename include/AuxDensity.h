//
// Created by Sambit Das.
//

#ifndef DFTFE_AUXDM_AUXDENSITY_H
#define DFTFE_AUXDM_AUXDENSITY_H

#include <vector>
#include <utility>

namespace dftfe
{
  class AuxDensity : public AuxDensityMatrix
  {
  public:
    // Constructor
    AuxDensity();


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
    projectDensity(const std::vector<double> &Qpts,
                   const std::vector<double> &QWt,
                   const int                  nQ,
                   const std::vector<double> &densityVals) = 0;

  private:
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITY_H
