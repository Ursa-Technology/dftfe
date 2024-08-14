//
// Created by Sambit Das
//

#ifndef DFTFE_ATOMBASISDATA_H
#define DFTFE_ATOMBASISDATA_H

#  include <vector>
#  include <memory>

#  include "AtomicBasis.h"

namespace dftfe
{
  class AtomBasisData
  {
  public:

    ///quadpoints in Cartesian coordinates
    void
    evalBasisData(const std::vector<double> &quadpts,
                  const AtomicBasis &     atomicBasis,
                  const unsigned int                        maxDerOrder);

    std::vector<double> &
    getBasisValues() const;


    std::vector<double> &
    getBasisGradValues() const;

    std::vector<double> &
    getBasisLaplacianValues() const;


  private:
    // Member variables
    std::vector<double> d_basisValues;
    std::vector<double> d_basisGradValues;
    std::vector<double> d_basisLaplacianValues;

    void
    evalBasisValues(const std::vector<double> &quadpts,
                    const AtomicBasis &     atomicBasis);

    void
    evalBasisGradValues(const std::vector<double> &quadpts,
                        const AtomicBasis &     atomicBasis);

    void
    evalBasisLaplacianValues(const std::vector<double> &quadpts,
                            const AtomicBasis &     atomicBasis);

  };
} // namespace dftfe
#endif // DFTFE_ATOMBASISDATA_H
