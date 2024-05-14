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

    /**
     * @brief Projects the FE density matrix to aux basis (L2 projection).
     *
     * @note This function computes:
     * d_DM := alpha * d_DM + beta * DM_dash
     *
     * @param Qpts The quadrature points.
     * @param QWt The quadrature weights.
     * @param nQ The number of quadrature points.
     * @param psiFunc The SCF wave function or eigen function in FE Basis.
     * @param fValues The SCF eigen values.
     * @param nPsi The number of wave functions as a pair (number of spin-up, number of spin-down).
     * @param alpha The scaling factor for existing density matrix d_DM.
     * @param beta The scaling factor for the newly calculated density matrix DM_dash.
     *
     * @par Example
     * @code
     * std::vector<double> Qpts = {...};  // Quadrature points
     * std::vector<double> QWt = {...};   // Quadrature weights
     * int nQ = Qpts.size();             // Number of quadrature points
     * std::vector<double> psiFunc = {...};  // Wave/Eigen function values
     * std::vector<double> fValues = {...};  // Eigenvalues
     * std::pair<int, int> nPsi = {5, 5};    // Number of spin-up and spin-down
     * wave functions double alpha = 1.0;               // Scaling factor for
     * d_DM double beta = 1.0;                // Scaling factor for DM_dash
     *
     * AuxDensityMatrix auxDensity;
     * auxDensity.projectDensityMatrix(Qpts, QWt, nQ, psiFunc, fValues, nPsi,
     * alpha, beta);
     * @endcode
     */
    virtual void
    projectDensityMatrix(const std::vector<double> &Qpts,
                         const std::vector<double> &QWt,
                         const int                  nQ,
                         const std::vector<double> &psiFunc,
                         const std::vector<double> &fValues,
                         const std::pair<int, int>  nPsi,
                         double                     alpha,
                         double                     beta) = 0;


    /**
     * @brief Projects the quadrature density to aux basis (L2 projection).
     *
     *
     * @param Qpts The quadrature points.
     * @param QWt The quadrature weights.
     * @param nQ The number of quadrature points.
     * @param densityVals density values at quad points with spin index
     * the slowest index followed by the quad index. nspin=2 assumed
     */
    virtual void
    projectDensity(const std::vector<double> &Qpts,
                   const std::vector<double> &QWt,
                   const int                  nQ,
                   const std::vector<double> &densityVals) = 0;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYMATRIX_H
