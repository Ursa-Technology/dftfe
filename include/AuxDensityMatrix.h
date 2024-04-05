//
// Created by Arghadwip Paul.
//

#ifndef AUXDM_AUXDENSITYMATRIX_H
#define AUXDM_AUXDENSITYMATRIX_H

#include <vector>
#include <utility>

enum class DensityDescriptorDataAttributes
{
    valuesTotal,
    valuesSpinUp,
    valuesSpinDown,
    gradValueSpinUp,
    gradValueSpinDown,
    sigma, // contracted gradients of the density using libxc format
    hessianSpinUp,
    hessianSpinDown,
    laplacianSpinUp,
    laplacianSpinDown,
    rhoStencilData,
    FDSpacing
};

class AuxDensityMatrix {
public:
    // Constructor
    AuxDensityMatrix();

    // Virtual destructor to ensure proper cleanup of derived classes
    virtual ~AuxDensityMatrix();

    // Pure virtual functions
    virtual void setDMzero() = 0;
    virtual void initializeLocalDescriptors() = 0;
    virtual void setLocalDescriptors(DensityDescriptorDataAttributes attr,
                                     std::pair<int, int> indexRange,
                                     std::vector<double> values) = 0;
    //virtual void getLocalDescriptors() = 0;
    virtual void evalLocalDescriptors() = 0;
    virtual void projectDensityMatrix(const std::vector<double>& Qpts,
                                      const std::vector<double>& QWt,
                                      const int nQ,
                                      const std::vector<double>& psiFunc,
                                      const std::vector<double>& fValues,
                                      const std::pair<int, int> nPsi,
                                      double alpha,
                                      double beta) = 0;
    virtual void projectDensity() = 0;
};

#endif //AUXDM_AUXDENSITYMATRIX_H
