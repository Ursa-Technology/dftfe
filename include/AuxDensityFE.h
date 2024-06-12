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

    // CAUTION: points have to be a contiguous subset of d_quadPointsSet
    void
    applyLocalOperations(
      const std::vector<double> &points,
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
     * This is actually a copy call all the quadrature points must
     * to be passed to this function in one go
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
    std::vector<double> d_densityValsTotalAllQuads;
    std::vector<double> d_densityValsSpinUpAllQuads;
    std::vector<double> d_densityValsSpinDownAllQuads;
    std::vector<double> d_gradDensityValsSpinUpAllQuads;
    std::vector<double> d_gradDensityValsSpinDownAllQuads;
    std::vector<double> d_quadPointsAll;
    std::vector<double> d_quadWeightsAll;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYFE_H
