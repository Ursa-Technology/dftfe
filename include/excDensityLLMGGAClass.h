//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_EXCDENSITYLLMGGACLASS_H
#define DFTFE_EXCDENSITYLLMGGACLASS_H

#include <xc.h>
#include <excDensityBaseClass.h>
namespace dftfe
{
  class NNLLMGGA;
  template <dftfe::utils::MemorySpace memorySpace>
  class excDensityLLMGGAClass : public excDensityBaseClass<memorySpace>
  {
  public:
    excDensityLLMGGAClass(xc_func_type *funcXPtr, xc_func_type *funcCPtr);

    excDensityLLMGGAClass(xc_func_type *funcXPtr,
                          xc_func_type *funcCPtr,
                          std::string   modelXCInputFile);

    ~excDensityLLMGGAClass();

    void
    computeExcVxcFxc(
      AuxDensityMatrix<memorySpace> &auxDensityMatrix,
      const std::vector<double> &    quadPoints,
      const std::vector<double> &    quadWeights,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
      std::unordered_map<xcOutputDataAttributes, std::vector<double>> &cDataout)
      const override;

    void
    checkInputOutputDataAttributesConsistency(
      const std::vector<xcOutputDataAttributes> &outputDataAttributes)
      const override;


  private:
    NNLLMGGA *          d_NNLLMGGAPtr;
    xc_func_type *      d_funcXPtr;
    xc_func_type *      d_funcCPtr;
    std::vector<double> d_spacingFDStencil;
    unsigned int        d_vxcDivergenceTermFDStencilSize;
  };
} // namespace dftfe
#endif // DFTFE_EXCDENSITYLLMGGACLASS_H
