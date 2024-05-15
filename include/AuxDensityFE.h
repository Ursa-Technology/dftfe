//
// Created by Sambit Das.
//

#ifndef DFTFE_AUXDM_AUXDENSITYFE_H
#define DFTFE_AUXDM_AUXDENSITYFE_H

#include <vector>
#include <utility>
#include <AuxDensityMatrix>

namespace dftfe
{
  class AuxDensityFE : public AuxDensityMatrix
  {
  public:
    // Constructor
    AuxDensityFE();

    // CAUTION: Points have to be a subset of d_quadPointsSet
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
                   const std::vector<double> &densityVals) override;


  private:
    std::vector<double> d_densityValsTotal;
    std::vector<double> d_densityValsSpinUp;
    std::vector<double> d_densityValsSpinDown;
    std::vector<double> d_gradDensityValsSpinUp;
    std::vector<double> d_gradDensityValsSpinDown;
    std::vector<double> d_quadPointsSet;
    std::vector<double> d_quadWeightsSet;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYFE_H
