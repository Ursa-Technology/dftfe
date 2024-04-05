//
// Created by Arghadwip Paul.
//

#ifndef LLMGGA_EXCDENSITYLLMGGACLASS_H
#define LLMGGA_EXCDENSITYLLMGGACLASS_H


#include <xc.h>
#include <excDensityBaseClass.h>
namespace dftfe
{
    class NNLLMGGA;
    class excDensityLLMGGAClass : public excDensityBaseClass
    {
    public:
        excDensityLLMGGAClass(xc_func_type *funcXPtr,
                              xc_func_type *funcCPtr,
                              bool          isSpinPolarized,
                              bool          scaleExchange,
                              bool          computeCorrelation,
                              double        scaleExchangeFactor);

        excDensityLLMGGAClass(xc_func_type *funcXPtr,
                              xc_func_type *funcCPtr,
                              bool          isSpinPolarized,
                              std::string   modelXCInputFile,
                              bool          scaleExchange,
                              bool          computeCorrelation,
                              double        scaleExchangeFactor);

        ~excDensityLLMGGAClass();

        void
        computeDensityBased_exc_vxc_FD(
                const AuxDensityMatrixSlater &       auxDM,
                const  int                           numPoints,
                const  int                           stencilOrder1D,
                std::vector<double> &                exc,
                std::vector<double> &                vxc);

    private:
        NNLLMGGA *    d_NNLLMGGAPtr;
        xc_func_type *d_funcXPtr;
        xc_func_type *d_funcCPtr;
    };
} // namespace dftfe

#endif //LLMGGA_EXCDENSITYLLMGGACLASS_H
