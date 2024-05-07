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
    class excDensityLLMGGAClass : public excDensityBaseClass
    {
    public:
        excDensityLLMGGAClass(xc_func_type *funcXPtr, xc_func_type *funcCPtr);

        excDensityLLMGGAClass(xc_func_type *funcXPtr,
                              xc_func_type *funcCPtr,
                              std::string   modelXCInputFile);

        ~excDensityLLMGGAClass();

        void
        computeExcVxcFxc(
                AuxDensityMatrixSlater &              auxDensityMatrix,
                const double *                        quadPoints,
                const double *                        quadWeights,
                const unsigned int                    numQuadPoints,
                std::map<xcOutputDataAttributes, std::vector<double>> &xDataOut,
                std::map<xcOutputDataAttributes, std::vector<double>> &cDataout) const override;

    private:
        NNLLMGGA *          d_NNLLMGGAPtr;
        xc_func_type *      d_funcXPtr;
        xc_func_type *      d_funcCPtr;
        std::vector<double> d_spacingFDStencil;
        unsigned int        d_vxcDivergenceTermFDStencilSize;
    };
} // namespace dftfe
#endif //DFTFE_EXCDENSITYLLMGGACLASS_H
