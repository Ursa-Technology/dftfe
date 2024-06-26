#ifndef DFTFE_FiniteDifference_H
#define DFTFE_FiniteDifference_H

#include <vector>

namespace dftfe
{
  namespace utils
  {
    class FiniteDifference
    {
    public:
      static std::vector<double>
      getStencilGridOneVariableCentral(const unsigned int totalStencilSize,
                                       const double       h) const;



      // stencil index is the fastest index in stencilDataAllQuadPoints
      // memory for firstOrderDerivative is assumed to be allocated
      static void
      firstOrderDerivativeOneVariableCentral(
        const unsigned int totalStencilSize,
        const double       h,
        const unsigned int numQuadPoints,
        const double *     stencilDataAllQuadPoints,
        double *           firstOrderDerivative) const;


      // stencil index is the fastest index in stencilDataAllQuadPoints
      // memory for secondOrderDerivative is assumed to be allocated
      static void
      secondOrderDerivativeOneVariableCentral(
        const unsigned int totalStencilSize,
        const double       h,
        const unsigned int numQuadPoints,
        const double *     stencilDataAllQuadPoints,
        double *           secondOrderDerivative) const;
    };
  } // namespace utils
} // namespace dftfe

#endif // DFTFE_FiniteDifference_H
