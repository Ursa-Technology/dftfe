//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_AUXDM_AUXDENSITYMATRIXSLATER_H
#define DFTFE_AUXDM_AUXDENSITYMATRIXSLATER_H


#include "AuxDensityMatrix.h"
#include "SlaterBasisSet.h"
#include "SlaterBasisData.h"
#include "AtomInfo.h"
#include <vector>
#include <utility>
#include <map>
#include <algorithm>
#include <Accelerate/Accelerate.h>

namespace dftfe
{
  class AuxDensityMatrixSlater : public AuxDensityMatrix
  {
  private:
    std::vector<double> d_quadpts;
    std::vector<double> d_quadWt;
    int                 d_nQuad;
    int                 d_nSpin;

    std::vector<Atom> d_atoms;
    SlaterBasisSet    d_sbs;
    SlaterBasisData   d_sbd;
    int               d_nBasis;
    int               d_maxDerOrder;

    std::vector<double> d_DM;

    std::vector<double>
    evalSlaterOverlapMatrix(const std::vector<double> &quadWt,
                            int                        nQuad,
                            int                        nBasis);

    std::vector<double>
    evalSlaterOverlapMatrixInv(const std::vector<double> &quadWt,
                               int                        nQuad,
                               int                        nBasis);


  public:
    // Constructor
    AuxDensityMatrixSlater(const std::map<int, std::string> &atomCoords,
                           const std::vector<double> &       quadpts,
                           const std::vector<double> &       quadWt,
                           const std::string                 atomBasisNameFile,
                           const int                         nQuad,
                           const int                         nSpin,
                           const int                         maxDerOrder);

    // Destructor
    virtual ~AuxDensityMatrixSlater();

    void
    setDMzero();

    void
    applyLocalOperations(const std::vector<double> &    Points,
                         std::map<DensityDescriptorDataAttributes,
                                  std::vector<double>> &densityData) override;

    /**
     * @brief Projects the FE density matrix to Slater Basis (L2 projection).
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
     * AuxDensityMatrixSlater auxDensity;
     * auxDensity.projectDensityMatrix(Qpts, QWt, nQ, psiFunc, fValues, nPsi,
     * alpha, beta);
     * @endcode
     */
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
    projectDensity() override;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYMATRIXSLATER_H
