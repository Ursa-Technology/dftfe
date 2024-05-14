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

    //CAUTION: Points have to be a subset of d_quadPointsSet
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

    /**
     * @brief set values for the quadrature density
     *
     *
     * @param Qpts The quadrature points.
     * @param QWt The quadrature weights.
     * @param nQ The number of quadrature points.
     * @param densityVals density values at quad points with spin index the
     * slowest index followed by the quad index. nspin=2 assumed
     */
    void
    setQuadVals(const std::vector<double> &Qpts,
                const std::vector<double> &QWt,
                const int                  nQ,
                const std::vector<double> &densityVals);

    /**
     * @brief set values for the quadrature density
     *
     *
     * @param Qpts The quadrature points.
     * @param QWt The quadrature weights.
     * @param nQ The number of quadrature points.
     * @param densityVals density values at quad points with spin index the
     * slowest index followed by the quad index. nspin=2 assumed
     * @param gradDensityVals gradient density values at quad points
     * with spin index the slowest index, followed by quad index,
     * and finally the dimension index. nspin=2 assumed
     */
    void
    setQuadVals(const std::vector<double> &Qpts,
                const std::vector<double> &QWt,
                const int                  nQ,
                const std::vector<double> &densityVals,
                const std::vector<double> &gradDensityVals);


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
