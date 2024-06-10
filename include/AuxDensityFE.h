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
    applyLocalOperations(
      const std::vector<double> &Points,
      std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
        &densityData) override;


    void
    projectDensityMatrixStart(
      std::unordered_map<std::string, std::vector<double>> &projectionInputs)
      override;

    void
    projectDensityMatrixEnd(
      std::unordered_map<std::string, std::vector<double>> &projectionInputs,
      int                                                   iSpin) override;

    /**
     * @brief Projects the quadrature density to aux basis (L2 projection).
     *
     * @param projectionInputs is a map from string to inputs needed
     *                          for projection.
     *      projectionInputs["quadpts"],
     *      projectionInputs["quadWt"],
     *      projectionInputs["densityFunc"]
     *      projectionInputs["gradDensityFunc"]
     *
     * densityFunc The density Values at quad points
     *                densityFunc(spin_index, quad_index),
     *                quad_index is fastest.
     *
     * gradDensityFunc The density Values at quad points
     *                gradDensityFunc(spin_index, quad_index,dim_index),
     *                dim_index is fastest.
     *
     */
    void
    projectDensityStart(std::unordered_map<std::string, std::vector<double>>
                          &projectionInputs) override;

    void
    projectDensityEnd() override;


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
