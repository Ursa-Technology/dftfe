//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_AUXDM_AUXDENSITYMATRIX_H
#define DFTFE_AUXDM_AUXDENSITYMATRIX_H

#include <vector>
#include <utility>
#include <map>
#include <string>
#include <unordered_map>
#include <mpi.h>

namespace dftfe
{
  enum class DensityDescriptorDataAttributes
  {
    valuesTotal,
    valuesSpinUp,
    valuesSpinDown,
    gradValuesSpinUp,
    gradValuesSpinDown,
    hessianSpinUp,
    hessianSpinDown,
    laplacianSpinUp,
    laplacianSpinDown
  };

  class AuxDensityMatrix
  {
  public:
    // Virtual destructor
    virtual ~AuxDensityMatrix();

    // Pure virtual functions

    virtual void
    applyLocalOperations(
      const std::vector<double> &Points,
      std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
        &densityData) = 0;

    virtual void
    evalOverlapMatrixStart(const std::vector<double> &quadpts,
                           const std::vector<double> &quadWt) = 0;

    virtual void
    evalOverlapMatrixEnd(const MPI_Comm &mpiComm) = 0;

    /**
     *
     * @param projectionInputs is a map from string to inputs needed
     *                          for projection.
     *      eg - projectionInputs["quadpts"],
     *          projectionInputs["quadWt"],
     *          projectionInputs["psiFunc"],
     *          projectionInputs["fValues"]
     *
     *      psiFunc The SCF wave function or eigen function in FE Basis.
     *                psiFunc(quad_index, wfc_index),
     *                quad_index is fastest.
     *      fValues are the occupancies.
     *
     * @param iSpin indicates up (iSpin = 0) or down (iSpin = 0) spin.
     *
     */
    virtual void
    projectDensityMatrixStart(
      std::unordered_map<std::string, std::vector<double>> &projectionInputs,
      int                                                   iSpin) = 0;

    virtual void
    projectDensityMatrixEnd(const MPI_Comm &mpiComm) = 0;


    /**
     * @brief Projects the quadrature density to aux basis (L2 projection).
     */
    virtual void
    projectDensityStart(std::unordered_map<std::string, std::vector<double>>
                          &projectionInputs) = 0;

    virtual void
    projectDensityEnd(const MPI_Comm &mpiComm) = 0;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYMATRIX_H
