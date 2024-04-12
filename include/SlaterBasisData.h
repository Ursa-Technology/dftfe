//
// Created by Arghadwip Paul on 2/15/24.
//

#ifndef SLATER2_SLATERBASISDATA_H
#define SLATER2_SLATERBASISDATA_H

#include <vector>
#include <memory>
#include "AtomInfo.h"
#include "SlaterBasisSet.h"
#include "SlaterPrimitive.h"
#include "SphericalHarmonicFunc.h"

class SlaterBasisData
{
public:
  // Member functions declarations
  void
  evalBasisData(const std::vector<Atom> &  atoms,
                const std::vector<double> &quadpts,
                const SlaterBasisSet &     sbs,
                int                        nQuad,
                int                        nBasis,
                int                        maxDerOrder);

  void
  evalSlaterOverlapMatrix(const std::vector<double> &quadWt,
                          int                        nQuad,
                          int                        nBasis);

  void
  evalSlaterOverlapMatrixInv(const std::vector<double> &quadWt,
                             int                        nQuad,
                             int                        nBasis);

  void
  printBasisInfo(int nQuad, int nBasis);

  double
  getBasisValues(const int index);

  std::vector<double>
  getBasisValuesAll();

  double
  getBasisGradValues(const int index);

  double
  getBasisHessianValues(const int index);

  std::vector<double>
  getSlaterOverlapMatrixInv();

private:
  // Member variables
  std::vector<double> basisValues;
  std::vector<double> basisGradValues;
  std::vector<double> basisHessValues;
  std::vector<double> SMatrixInv;

  void
  evalBasisValues(const std::vector<Atom> &  atoms,
                  const std::vector<double> &quadpts,
                  const SlaterBasisSet &     sbs,
                  int                        nBasis);

  void
  evalBasisGradValues(const std::vector<Atom> &  atoms,
                      const std::vector<double> &quadpts,
                      const SlaterBasisSet &     sbs,
                      int                        nBasis);

  void
  evalBasisHessianValues(const std::vector<Atom> &  atoms,
                         const std::vector<double> &quadpts,
                         const SlaterBasisSet &     sbs,
                         int                        nBasis);

  torch::Tensor
  evalSlaterFunc(const torch::Tensor &  x_s,
                 const torch::Tensor &  y_s,
                 const torch::Tensor &  z_s,
                 const SlaterPrimitive &basis);
};

#endif // SLATER2_SLATERBASISDATA_H
