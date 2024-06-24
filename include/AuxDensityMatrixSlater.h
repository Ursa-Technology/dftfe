//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_AUXDM_AUXDENSITYMATRIXSLATER_H
#define DFTFE_AUXDM_AUXDENSITYMATRIXSLATER_H


#include "AuxDensityMatrix.h"
#include "SlaterBasisSet.h"
#include "SlaterBasisData.h"
#include <vector>
#include <utility>
#include <map>
#include <algorithm>


namespace dftfe
{
  class AuxDensityMatrixSlater : public AuxDensityMatrix
  {
  private:
    int             d_nQuad;
    int             d_nSpin;
    SlaterBasisSet  d_sbs;
    SlaterBasisData d_sbd;

    int d_nBasis;
    int d_maxDerOrder;

    std::vector<double> d_DM;
    std::vector<double> d_SMatrix;
    std::vector<double> d_SMatrixInv;
    std::vector<double> d_SWFC;
    std::vector<double> d_fValues;

    int d_nWFC;
    int d_iSpin;


    void
    evalOverlapMatrixInv();
    std::vector<double> &
    getOverlapMatrixInv();

  public:
    // Constructor
    AuxDensityMatrixSlater();

    // Destructor
    virtual ~AuxDensityMatrixSlater();

    void
    applyLocalOperations(
      const std::vector<double> &Points,
      std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>>
        &densityData) override;

    void
    reinitAuxDensityMatrix(
      const std::vector<std::pair<std::string, std::vector<double>>>
        &                atomCoords,
      const std::string &auxBasisFile,
      const int          nSpin,
      const int          maxDerOrder) override;

    void
    evalOverlapMatrixStart(const std::vector<double> &quadpts,
                           const std::vector<double> &quadWt) override;

    void
    evalOverlapMatrixEnd(const MPI_Comm &mpiComm) override;

    void
    projectDensityMatrixStart(
      std::unordered_map<std::string, std::vector<double>> &projectionInputs,
      int                                                   iSpin) override;

    void
    projectDensityMatrixEnd(const MPI_Comm &mpiComm) override;

    void
    projectDensityStart(std::unordered_map<std::string, std::vector<double>>
                          &projectionInputs) override;

    void
    projectDensityEnd(const MPI_Comm &mpiComm) override;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYMATRIXSLATER_H
