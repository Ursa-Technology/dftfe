//
// Created by Arghadwip Paul.
//

#include "AuxDensityMatrixSlater.h"

AuxDensityMatrixSlater::AuxDensityMatrixSlater(
        const std::vector<double>& quadpts,
        const std::vector<double>& quadWt,
        const std::string coordFile,
        const int nQuad,
        const int nSpin,
        const int maxDerOrder
        ):
        AuxDensityMatrix(),  // call the base class constructor
        d_quadpts(quadpts),
        d_quadWt(quadWt),
        d_coordFile(coordFile),
        d_nQuad(nQuad),
        d_nSpin(nSpin),
        d_maxDerOrder(maxDerOrder)
{
    // ------------------ Read AtomicCoords_Slater --------------
    d_atoms = Atom::readCoordFile(coordFile);

    // ------------------ SlaterBasisSets -------------
    d_sbs.constructBasisSet(d_atoms);
    d_nBasis = d_sbs.getTotalBasisSize(d_atoms);
    std::cout << "nBasis : " << d_nBasis << std::endl;

    d_DM.assign(nSpin * d_nBasis * d_nBasis, 0.0);
    this -> initializeLocalDescriptors();

    d_sbd.evalBasisData(d_atoms, d_quadpts, d_sbs, d_nQuad, d_nBasis, d_maxDerOrder);
    d_sbd.evalSlaterOverlapMatrixInv(d_quadWt, d_nQuad, d_nBasis);
}

AuxDensityMatrixSlater::~AuxDensityMatrixSlater() {
    // Destructor implementation
}

void AuxDensityMatrixSlater::setDMzero() {
    // setDMzero implementation
    for (auto& element : d_DM) {
        element = 0.0;
    }
}

void
AuxDensityMatrixSlater::initializeLocalDescriptors(){
    attributeData[DensityDescriptorDataAttributes::valuesTotal] = std::vector<double>(d_nQuad, 0.0);
    attributeData[DensityDescriptorDataAttributes::valuesSpinUp] = std::vector<double>(d_nQuad, 0.0);
    attributeData[DensityDescriptorDataAttributes::valuesSpinDown] = std::vector<double>(d_nQuad, 0.0);
    attributeData[DensityDescriptorDataAttributes::gradValueSpinUp] = std::vector<double>(d_nQuad * 3, 0.0);
    attributeData[DensityDescriptorDataAttributes::gradValueSpinDown] = std::vector<double>(d_nQuad * 3, 0.0);
    attributeData[DensityDescriptorDataAttributes::sigma] = std::vector<double>(d_nQuad * 3, 0.0);
    attributeData[DensityDescriptorDataAttributes::hessianSpinUp] = std::vector<double>(d_nQuad * 9, 0.0);
    attributeData[DensityDescriptorDataAttributes::hessianSpinDown] = std::vector<double>(d_nQuad * 9, 0.0);
    attributeData[DensityDescriptorDataAttributes::laplacianSpinUp] = std::vector<double>(d_nQuad, 0.0);
    attributeData[DensityDescriptorDataAttributes::laplacianSpinDown] = std::vector<double>(d_nQuad, 0.0);
}

void
AuxDensityMatrixSlater::setLocalDescriptors(
        DensityDescriptorDataAttributes attr,
        std::pair<int, int> indexRange,
        std::vector<double> values) {

    if (indexRange.first >= indexRange.second) {
        throw std::invalid_argument("Invalid index range: The start index must be less than the end index.");
    }

    auto it = attributeData.find(attr);
    if (it == attributeData.end()) {
        throw std::invalid_argument("Specified attribute does not exist.");
    }

    size_t expectedRangeSize = indexRange.second - indexRange.first;
    if (values.size() != expectedRangeSize) {
        throw std::invalid_argument("Values vector size does not match the specified index range.");
    }

    if (indexRange.second > it->second.size()) {
        throw std::out_of_range("Index range is out of bounds for the specified attribute.");
    }

    std::copy(values.begin(), values.end(), it->second.begin() + indexRange.first);

}

/*
std::vector<double> AuxDensityMatrixSlater::getLocalDescriptors(
        DensityDescriptorDataAttributes attr,
        std::pair<int, int> indexRange) const {

    if (indexRange.first >= indexRange.second) {
        throw std::invalid_argument("Invalid index range: The start index must be less than the end index.");
    }

    auto it = attributeData.find(attr);
    if (it == attributeData.end()) {
        throw std::invalid_argument("Specified attribute does not exist.");
    }

    if (indexRange.second > it->second.size()) {
        throw std::out_of_range("Index range is out of bounds for the specified attribute.");
    }

    // Extract and return the values vector for the specified range
    std::vector<double> values(it->second.begin() + indexRange.first, it->second.begin() + indexRange.second);

    return values;
}
*/



void
AuxDensityMatrixSlater::evalLocalDescriptors() {
    // evalLocalDescriptors implementation
    std::cout << "eval local descriptors" << std::endl;

    int DMSpinOffset = d_nBasis * d_nBasis;
    std::pair<int, int> indexRange;
    for(int iQuad = 0; iQuad < d_nQuad; iQuad++) {

        std::vector<double> rhoUp(1, 0.0);
        std::vector<double> rhoDown(1, 0.0);
        std::vector<double> rhoTotal(1, 0.0);
        std::vector<double> gradrhoUp(3, 0.0);
        std::vector<double> gradrhoDown(3, 0.0);
        std::vector<double> sigma(3, 0.0);
        std::vector<double> HessianrhoUp(9, 0.0);
        std::vector<double> HessianrhoDown(9, 0.0);
        std::vector<double> LaplacianrhoUp(1, 0.0);
        std::vector<double> LaplacianrhoDown(1, 0.0);

        for (int i = 0; i < d_nBasis; i++) {
            for (int j = 0; j < d_nBasis; j++) {
                for(int iSpin = 0; iSpin < d_nSpin; iSpin++) {
                    if (iSpin == 0) {
                        rhoUp[0] += d_DM[i * d_nBasis + j] * d_sbd.getBasisValues(iQuad * d_nBasis + i) *
                                    d_sbd.getBasisValues(iQuad * d_nBasis + j);

                        for(int derIndex = 0; derIndex < 3; derIndex++) {
                            gradrhoUp[derIndex] += d_DM[i * d_nBasis + j] *
                                                    (d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * i + derIndex) *
                                                     d_sbd.getBasisValues(iQuad * d_nBasis + j) +
                                                     d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * j + derIndex) *
                                                     d_sbd.getBasisValues(iQuad * d_nBasis + i)
                                                     );
                        }

                        for(int derIndex1 = 0; derIndex1 < 3; derIndex1++) {
                            for(int derIndex2 = 0; derIndex2 < 3; derIndex2++){
                                HessianrhoUp[derIndex1 * 3 + derIndex2] += d_DM[i * d_nBasis + j] *
                                        (
                                        d_sbd.getBasisHessianValues(iQuad * d_nBasis + 9 * i + 3 * derIndex1 + derIndex2) *
                                        d_sbd.getBasisValues(iQuad * d_nBasis + j) +
                                        d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * i + derIndex1) *
                                        d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * j + derIndex2) +
                                        d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * i + derIndex2) *
                                        d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * j + derIndex1) +
                                        d_sbd.getBasisValues(iQuad * d_nBasis + i) *
                                        d_sbd.getBasisHessianValues(iQuad * d_nBasis + 9 * j + 3 * derIndex1 + derIndex2)
                                        );
                            }
                        }
                    }

                    if (iSpin == 1) {
                        rhoDown[0] += d_DM[DMSpinOffset + i * d_nBasis + j] * d_sbd.getBasisValues(iQuad * d_nBasis + i) *
                                   d_sbd.getBasisValues(iQuad * d_nBasis + j);

                        for(int derIndex = 0; derIndex < 3; derIndex++) {
                            gradrhoDown[derIndex] += d_DM[DMSpinOffset + i * d_nBasis + j] *
                                                   (d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * i + derIndex) *
                                                    d_sbd.getBasisValues(iQuad * d_nBasis + j) +
                                                    d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * j + derIndex) *
                                                    d_sbd.getBasisValues(iQuad * d_nBasis + i)
                                                    );
                        }

                        for(int derIndex1 = 0; derIndex1 < 3; derIndex1++) {
                            for(int derIndex2 = 0; derIndex2 < 3; derIndex2++){
                                HessianrhoDown[derIndex1 * 3 + derIndex2] += d_DM[DMSpinOffset + i * d_nBasis + j] *
                                        (
                                        d_sbd.getBasisHessianValues(iQuad * d_nBasis + 9 * i + 3 * derIndex1 + derIndex2) *
                                        d_sbd.getBasisValues(iQuad * d_nBasis + j) +
                                        d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * i + derIndex1) *
                                        d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * j + derIndex2) +
                                        d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * i + derIndex2) *
                                        d_sbd.getBasisGradValues(iQuad * d_nBasis + 3 * j + derIndex1) +
                                        d_sbd.getBasisValues(iQuad * d_nBasis + i) *
                                        d_sbd.getBasisHessianValues(iQuad * d_nBasis + 9 * j + 3 * derIndex1 + derIndex2)
                                        );
                            }
                        }
                    }
                }
            }
        }

        rhoTotal[0] = rhoUp[0] + rhoDown[0];
        sigma[0] = gradrhoUp[0] * gradrhoUp[0] + gradrhoUp[1] * gradrhoUp[1] + gradrhoUp[2] * gradrhoUp[2];
        sigma[1] = gradrhoUp[0] * gradrhoDown[0] + gradrhoUp[1] * gradrhoDown[1] + gradrhoUp[2] * gradrhoDown[2];
        sigma[2] = gradrhoDown[0] * gradrhoDown[0] + gradrhoDown[1] * gradrhoDown[1] + gradrhoDown[2] * gradrhoDown[2];
        LaplacianrhoUp[0] = HessianrhoUp[0] + HessianrhoUp[4] + HessianrhoUp[8];
        LaplacianrhoDown[0] = HessianrhoDown[0] + HessianrhoDown[4] + HessianrhoDown[8];

        indexRange = std::make_pair(iQuad, iQuad + 1);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::valuesSpinUp, indexRange, rhoUp);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::valuesSpinDown, indexRange, rhoDown);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::valuesTotal, indexRange, rhoTotal);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::laplacianSpinUp, indexRange, LaplacianrhoUp);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::laplacianSpinDown, indexRange, LaplacianrhoDown);

        indexRange = std::make_pair(iQuad * 3, iQuad * 3 + 3);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::gradValueSpinUp, indexRange, gradrhoUp);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::gradValueSpinDown, indexRange, gradrhoDown);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::sigma, indexRange, sigma);

        indexRange = std::make_pair(iQuad * 9, iQuad * 9 + 9);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::hessianSpinUp, indexRange, HessianrhoUp);
        this -> setLocalDescriptors(DensityDescriptorDataAttributes::hessianSpinDown, indexRange, HessianrhoDown);
    }
}

void
AuxDensityMatrixSlater::projectDensityMatrix(
        const std::vector<double>& Qpts,
        const std::vector<double>& QWt,
        const int nQ,
        const std::vector<double>& psiFunc,
        const std::vector<double>& fValues,
        const std::pair<int, int> nPsi,
        double alpha,
        double beta) {


    // Check if QPts are same as quadpts, then update quadpts, qwt, nQuad
    /*
    if(Qpts.size() != d_quadWt.size()){
        d_quadpts = Qpts;
        d_quadWt = QWt;
        d_nQuad  = nQ;
        d_sbd.evalBasisData(d_atoms, d_quadpts, d_sbs, d_nQuad, d_nBasis, d_maxDerOrder);
        d_sbd.evalSlaterOverlapMatrixInv(d_quadWt, d_nQuad, d_nBasis);
    }
    else if( (std::equal(Qpts.begin(), Qpts.end(), d_quadWt.begin())) == 0){
        d_quadpts = Qpts;
        d_quadWt = QWt;
        d_nQuad  = nQ;
        d_sbd.evalBasisData(d_atoms, d_quadpts, d_sbs, d_nQuad, d_nBasis, d_maxDerOrder);
        d_sbd.evalSlaterOverlapMatrixInv(d_quadWt, d_nQuad, d_nBasis);
    } */

    int nPsi1 = nPsi.first;
    int nPsi2 = nPsi.second;
    int nPsiTot = nPsi1 + nPsi2;

    std::vector<double> SlaterOverlapInv = d_sbd.getSlaterOverlapMatrixInv();
    std::vector<double> SlaterBasisVal   = d_sbd.getBasisValuesAll();

    // stores AA = Slater_Phi^T * FE_EigVec = (Ns * Nq) * (Nq * N_eig) = (Ns * N_eig)
    // Slater_Phi^T is quadrature weighted

    std::vector<double> AA(d_nBasis * nPsiTot, 0.0);

    std::vector<double> AA1(d_nBasis * nPsi1, 0.0);
    std::vector<double> AA2(d_nBasis * nPsi2, 0.0);

    std::vector<double> SlaterBasisValWeighted = SlaterBasisVal;
    for (int i = 0; i < d_nQuad; ++i) {
        for (int j = 0; j < d_nBasis; ++j) {
            SlaterBasisValWeighted[i * d_nBasis + j] *= d_quadWt[i];
        }
    }

    // Perform matrix multiplication: C[:, :Ne1] = A^T * B1
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_nBasis, nPsi1, d_nQuad,
                1.0, SlaterBasisValWeighted.data(), d_nBasis,
                psiFunc.data(), nPsi1,
                0.0, AA1.data(), nPsi1);

    // Perform matrix multiplication: C[:, Ne1:] = A^T * B2
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                d_nBasis, nPsi2, d_nQuad,
                1.0, SlaterBasisValWeighted.data(), d_nBasis,
                psiFunc.data() + d_nQuad * nPsi1, nPsi2,
                0.0, AA2.data(), nPsi2);

    // stores BB = S^(-1) * AA   // (Ns * N_eig)
    std::vector<double> BB1(d_nBasis * nPsi1, 0.0);
    std::vector<double> BB2(d_nBasis * nPsi2, 0.0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                d_nBasis, nPsi1, d_nBasis,
                1.0, SlaterOverlapInv.data(), d_nBasis,
                AA1.data(), nPsi1,
                0.0, BB1.data(), nPsi1);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                d_nBasis, nPsi2, d_nBasis,
                1.0, SlaterOverlapInv.data(), d_nBasis,
                AA2.data(), nPsi2,
                0.0, BB2.data(), nPsi2);

    // BF = BB * F    // (Ns * Neig)
    std::vector<double> BF1 = BB1;
    std::vector<double> BF2 = BB2;

    for(int i = 0; i < d_nBasis; i++){
        for(int j = 0; j < nPsi1; j++){
            BF1[i * nPsi1 + j] *= fValues[j];
        }
        for(int j = 0; j < nPsi2; j++){
            BF2[i * nPsi2 + j] *= fValues[nPsi1 + j];
        }
    }

    std::vector <double> DM_dash(d_nSpin * d_nBasis * d_nBasis, 0.0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                d_nBasis, d_nBasis, nPsi1,
                1.0, BF1.data(), nPsi1,
                BB1.data(), nPsi1,
                0.0, DM_dash.data(), d_nBasis);

    // Perform matrix multiplication: C[:, Ne1:] = A^T * B2
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                d_nBasis, d_nBasis, nPsi2,
                1.0, BF2.data(), nPsi2,
                BB2.data(), nPsi2,
                0.0, DM_dash.data() + d_nBasis * d_nBasis, d_nBasis);

    for (int i = 0; i < d_DM.size(); i++){
        d_DM[i] = alpha * d_DM[i] + beta * DM_dash[i];
    }

    std::cout << "DM1:" << std::endl;
    for (int i = 0; i < d_nBasis; ++i) {
        for (int j = 0; j < d_nBasis; ++j) {
            std::cout << DM_dash[i * d_nBasis + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "DM2:" << std::endl;
    for (int i = 0; i < d_nBasis; ++i) {
        for (int j = 0; j < d_nBasis; ++j) {
            std::cout << DM_dash[d_nBasis * d_nBasis + i * d_nBasis + j] << " ";
        }
        std::cout << std::endl;
    }
}


void AuxDensityMatrixSlater::projectDensity() {
    // projectDensity implementation
    std::cout << "Error : No implementation yet" << std::endl;
}
