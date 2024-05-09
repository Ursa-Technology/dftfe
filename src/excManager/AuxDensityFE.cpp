//
// Created by Sambit Das.
//

#include "AuxDensityFE.h"

namespace dftfe
{
  // Constructor implementation
  AuxDensityFE::AuxDensityFE()
  {}

  void
  AuxDensityFE::applyLocalOperations(
    const std::vector<double> &                                     Points,
    std::map<DensityDescriptorDataAttributes, std::vector<double>> &densityData)
  {}

  void
  AuxDensityFE::projectDensityMatrix(const std::vector<double> &Qpts,
                                     const std::vector<double> &QWt,
                                     const int                  nQ,
                                     const std::vector<double> &psiFunc,
                                     const std::vector<double> &fValues,
                                     const std::pair<int, int>  nPsi,
                                     double                     alpha,
                                     double                     beta)
  {}

  void
  AuxDensityFE::projectDensity(const std::vector<double> &Qpts,
                               const std::vector<double> &QWt,
                               const int                  nQ,
                               const std::vector<double> &densityVals,
                               const std::vector<double> &gradDensityVals)
  {}

} // namespace dftfe
