//
// Created by Arghadwip Paul.
//

#ifndef AUXDM_AUXDENSITYMATRIXSLATER_H
#define AUXDM_AUXDENSITYMATRIXSLATER_H

#include "AuxDensityMatrix.h"
#include "SlaterBasisSet.h"
#include "SlaterBasisData.h"
#include "AtomInfo.h"
#include <vector>
#include <utility>
#include <map>
#include <algorithm>
#include <Accelerate/Accelerate.h>

class AuxDensityMatrixSlater : public AuxDensityMatrix {

private:
    std::vector<double> d_quadpts;
    std::vector<double> d_quadWt;
    std::string d_coordFile;
    int d_nQuad;
    int d_nSpin;

    std::vector<Atom> d_atoms;
    SlaterBasisSet d_sbs;
    SlaterBasisData d_sbd;
    int d_nBasis;
    int d_maxDerOrder;

    std::vector<double> d_DM;
    std::unordered_map<DensityDescriptorDataAttributes, std::vector<double>> attributeData;


public:
    // Constructor
    AuxDensityMatrixSlater(const std::vector<double>& quadpts,
                           const std::vector<double>& quadWt,
                           const std::string coordFile,
                           const int nQuad,
                           const int nSpin,
                           const int maxDerOrder
                           );

    // Destructor
    virtual ~AuxDensityMatrixSlater();

    // Implementations of pure virtual functions from the base class
    void setDMzero() override;
    void initializeLocalDescriptors() override;
    void evalLocalDescriptors() override;
    void projectDensity() override;

    void setLocalDescriptors(DensityDescriptorDataAttributes attr,
                             std::pair<int, int> indexRange,
                             std::vector<double> values) override;

    /*
    std::vector<double> getLocalDescriptors(DensityDescriptorDataAttributes attr,
                                            std::pair<int, int> indexRange) const;*/


    void projectDensityMatrix(const std::vector<double>& Qpts,
                              const std::vector<double>& QWt,
                              const int nQ,
                              const std::vector<double>& psiFunc,
                              const std::vector<double>& fValues,
                              const std::pair<int, int> nPsi,
                              double alpha,
                              double beta) override;

};

#endif //AUXDM_AUXDENSITYMATRIXSLATER_H
