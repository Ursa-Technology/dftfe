//
// Created by Arghadwip Paul.
//

#include "BaseXCHandler.h"
#include <stdexcept>
#include <iostream>
#include <utility>
#include <map>

BaseXCHandler::
BaseXCHandler(
        const std::string& XFunctionalName,
        const std::string& CFunctionalName,
        const std::string& spin):
        hasX(!XFunctionalName.empty()),
        hasC(!CFunctionalName.empty()),
        spin(spin){

    if (spin != "polarized" && spin != "unpolarized") {
        throw std::invalid_argument("Invalid spin value = " + spin +
                                    " encountered. Spin value can be either" +
                                    "polarized or unpolarized");
    }

    if (spin == "unpolarized")
        nspin = 1;
    else
        nspin = 2;

    if (hasX && XFunctionalName == "GGA_X_PBE") {
        if (nspin == 1) {
            if (xc_func_init(&func_X, XC_GGA_X_PBE, XC_UNPOLARIZED) != 0)
                std::cerr << "Failed to initialize unpolarized exchange functional." << std::endl;
        }
        else {
            if (xc_func_init(&func_X, XC_GGA_X_PBE, XC_POLARIZED) != 0)
                std::cerr << "Failed to initialize polarized exchange functional." << std::endl;
        }
    }

    if (hasC && CFunctionalName == "GGA_C_PBE"){
        if (nspin == 1) {
            if (xc_func_init(&func_C, XC_GGA_C_PBE, XC_UNPOLARIZED) != 0)
                std::cerr << "Failed to initialize unpolarized correlation functional." << std::endl;
        }
        else {
            if (xc_func_init(&func_C, XC_GGA_C_PBE, XC_POLARIZED) != 0)
                std::cerr << "Failed to initialize polarized correlation functional." << std::endl;
        }

    }
}

void
BaseXCHandler::getexcAndvxc(
        const std::vector<double>& rho,
        const std::vector<double>& drho,
        const std::vector<double>& ddrho,
        const std::vector<double>& sigma,
        std::vector<double>& exc,
        std::vector<double>& vxc,
        int nPoints) {

    if (spin == "unpolarized"){
        throw std::invalid_argument("unpolarized not implemented yet");
    }

    std::vector<double> ex(nPoints, 0.0);
    std::vector<double> vx(nPoints * 2, 0.0);
    std::vector<double> ec(nPoints, 0.0);
    std::vector<double> vc(nPoints * 2, 0.0);

    std::vector<double> vrho(nPoints * 2, 0.0);
    std::vector<double> vsigma(nPoints * 3, 0.0);
    std::vector<double> v2rhosigma(nPoints * 6, 0.0);
    std::vector<double> v2sigma2(nPoints * 6, 0.0);
    std::vector<double> v2rho2(nPoints * 3, 0.0);

    if (hasX) {
        xc_gga_exc_vxc(&func_X, nPoints, rho.data(), sigma.data(),
                       ex.data(), vrho.data(), vsigma.data());
        xc_gga_fxc(&func_X, nPoints, rho.data(), sigma.data(),
                   v2rho2.data(), v2rhosigma.data(), v2sigma2.data());

        vx = getv(vrho, vsigma, v2rhosigma, v2sigma2, drho, ddrho, nPoints);

    }

    if (hasC) {
        xc_gga_exc_vxc(&func_C, nPoints, rho.data(), sigma.data(),
                       ec.data(), vrho.data(), vsigma.data());
        xc_gga_fxc(&func_C, nPoints, rho.data(), sigma.data(),
                   v2rho2.data(), v2rhosigma.data(), v2sigma2.data());

        vc = getv(vrho, vsigma, v2rhosigma, v2sigma2, drho, ddrho, nPoints);
    }

    for(unsigned int i = 0; i < nPoints; i++) {
        exc[i]         = (ex[i] + ec[i]) * (rho[2*i] + rho[2*i+1]);
        vxc[2 * i]     = vx[2 * i] + vc[2 * i];
        vxc[2 * i + 1] = vx[2 * i + 1] + vc[2 * i + 1];
    }

    /*
    // Output results
    std::cout << "Exchange-correlation energy per particle: " << exc[0] << ", " << exc[1] << std::endl;

    for (const auto& value : vxc) {
        std::cout << value << std::endl;
    }*/

}

std::vector<double>
BaseXCHandler::getv(
        const std::vector<double> &vrho,
        const std::vector<double> &vsigma,
        const std::vector<double> &v2rhosigma,
        const std::vector<double> &v2sigma2,
        const std::vector<double> &drho,
        const std::vector<double> &ddrho,
        int nPoints) {

    std::vector<double> v(vrho);

    int nSigma = 1;
    if (nspin == 2)
        nSigma = 3;

    auto gradSigma = getGradSigma(drho, ddrho, nPoints, nSigma);

    std::map<std::pair<int, int>, int> v2sigma2IJMap;

    for (int iSigma = 0; iSigma < nSigma; ++iSigma) {
        for (int jSigma = iSigma; jSigma < nSigma; ++jSigma) {
            int offset = iSigma * nSigma - (iSigma - 1) * iSigma / 2 + (jSigma - iSigma);
            v2sigma2IJMap[std::make_pair(iSigma, jSigma)] = offset;
            v2sigma2IJMap[std::make_pair(jSigma, iSigma)] = offset;
        }
    }


    for(int alpha = 0; alpha < nspin; alpha++) {
        for(int iSigma = 0; iSigma < nSigma; iSigma++) {
            auto g = getg(drho, nPoints, iSigma, alpha);
            for(int iSpin = 0; iSpin < nspin; iSpin++) {
                for (int iPoint = 0; iPoint < nPoints; iPoint++) {
                    v[iPoint * nspin + alpha] -= v2rhosigma[(iPoint * 6) + iSpin * nSigma + iSigma] *
                                                 vecvecDotproduct(drho, g, iPoint * nspin * 3 + iSpin * 3,
                                                                  iPoint * 3);
                }
            }

            for(int jSigma = 0; jSigma < nSigma; jSigma++) {
                auto index = v2sigma2IJMap[{iSigma, jSigma}];
                for (int iPoint = 0; iPoint < nPoints; iPoint++) {
                    v[iPoint * nspin + alpha] -= v2sigma2[iPoint * 6 + index] *
                                                 vecvecDotproduct(gradSigma, g, iPoint * nSigma * 3 + jSigma * 3,
                                                                  iPoint * 3);

                }
            }

            auto divg = getDivg(ddrho, nPoints, iSigma, alpha);
            for (int iPoint = 0; iPoint < nPoints; iPoint++)
                v[iPoint * nspin + alpha] -= (vsigma[iPoint * 3 + iSigma] * divg[iPoint]);
        }
    }
    return v;
}

std::vector<double>
BaseXCHandler::getGradSigma(
        const std::vector<double>& drho,
        const std::vector<double>& ddrho,
        int nPoints,
        int nSigma) {

    std::vector<double> gradSigma(nPoints * nSigma * 3, 0.0);

    for (int iPoint = 0; iPoint < nPoints; iPoint++) {
        for (int iSigma = 0; iSigma < nSigma; iSigma++) {
            int strIndex = iPoint * nSigma * 3 + iSigma * 3;

            if (iSigma == 0) {
                auto ddrho0drho0 = multiplyDDrhoDrho(ddrho, drho, (iPoint * nspin) * 9,
                                                     (iPoint * nspin) * 3);
                gradSigma[strIndex]     = 2.0 * ddrho0drho0[0];
                gradSigma[strIndex + 1] = 2.0 * ddrho0drho0[1];
                gradSigma[strIndex + 2] = 2.0 * ddrho0drho0[2];
            }
            else if (iSigma == 1) {
                auto ddrho0drho1 = multiplyDDrhoDrho(ddrho, drho, (iPoint * nspin) * 9,
                                                      (iPoint * nspin + 1) * 3);
                auto ddrho1drho0 = multiplyDDrhoDrho(ddrho, drho, (iPoint * nspin + 1) * 9,
                                                     (iPoint * nspin) * 3);
                gradSigma[strIndex]     = ddrho0drho1[0] + ddrho1drho0[0];
                gradSigma[strIndex + 1] = ddrho0drho1[1] + ddrho1drho0[1];
                gradSigma[strIndex + 2] = ddrho0drho1[2] + ddrho1drho0[2];

            }
            else if (iSigma == 2) {
                auto ddrho1drho1 = multiplyDDrhoDrho(ddrho, drho, (iPoint * nspin + 1) * 9,
                                                    (iPoint * nspin + 1) * 3);
                gradSigma[strIndex]     = 2.0 * ddrho1drho1[0];
                gradSigma[strIndex + 1] = 2.0 * ddrho1drho1[1];
                gradSigma[strIndex + 2] = 2.0 * ddrho1drho1[2];
            }
            else {
                throw std::invalid_argument("Invalid sigma index/nSigma passed to gradSigma");
            }
        }

    }

    return gradSigma;

}

std::vector<double>
BaseXCHandler::getg(
        const std::vector<double>& drho,
        int nPoints,
        int ii,
        int alpha
        ){

    std::vector<double> g(nPoints * 3, 0.0);
    if(ii == 1) {
        for (int iPoint = 0; iPoint < nPoints; iPoint++){
            int drho_index = ((iPoint * nspin) + (nspin - 1 - alpha)) * 3;
            g[iPoint * 3]     = drho[drho_index ];
            g[iPoint * 3 + 1] = drho[drho_index + 1];
            g[iPoint * 3 + 2] = drho[drho_index + 2];
        }
    }
    else if ((ii == 0 && alpha == 0) || (ii == 2 && alpha == 1))
    {
        for (int iPoint = 0; iPoint < nPoints; iPoint++){
            int drho_index = ((iPoint * nspin) + alpha) * 3;
            g[iPoint * 3]     = 2.0 * drho[drho_index];
            g[iPoint * 3 + 1] = 2.0 * drho[drho_index + 1];
            g[iPoint * 3 + 2] = 2.0 * drho[drho_index + 2];
        }
    }
    return g;
}

std::vector<double>
BaseXCHandler::getDivg(
        const std::vector<double>& ddrho,
        int nPoints,
        int ii,
        int alpha) {

    std::vector<double> divg(nPoints, 0.0);
    if(ii == 1) {
        for (int iPoint = 0; iPoint < nPoints; iPoint++){
            int ddrho_index = ((iPoint * nspin) + (nspin - 1 - alpha)) * 9;
            divg[iPoint]     = ddrho[ddrho_index] +
                               ddrho[ddrho_index + 4] +
                               ddrho[ddrho_index + 8];
        }
    }
    else if ((ii == 0 && alpha == 0) || (ii == 2 && alpha == 1))
    {
        for (int iPoint = 0; iPoint < nPoints; iPoint++){
            int ddrho_index = ((iPoint * nspin) + alpha) * 9;
            divg[iPoint]     = 2.0 * (ddrho[ddrho_index] +
                                      ddrho[ddrho_index + 4] +
                                      ddrho[ddrho_index + 8]);
        }
    }
    return divg;
}

std::vector<double>
BaseXCHandler::multiplyDDrhoDrho(
        const std::vector<double>& ddrho,
        const std::vector<double>& drho,
        int ddrhoStartIndex,
        int drhoStartIndex) {

    std::vector<double> ddrhodrho(3);
    for(int i = 0; i < 3; i++) {
        ddrhodrho[i] =  ddrho[ddrhoStartIndex + 3 * i] * drho[drhoStartIndex]
                       + ddrho[ddrhoStartIndex + 3 * i + 1] * drho[drhoStartIndex + 1]
                       + ddrho[ddrhoStartIndex + 3 * i + 2] * drho[drhoStartIndex + 2];
    }
    return ddrhodrho;
}

double
BaseXCHandler::vecvecDotproduct(
        const std::vector<double>& vec1,
        const std::vector<double>& vec2,
        int vec1StartIndex,
        int vec2StartIndex) {

    double dotprod;

    dotprod =  vec1[vec1StartIndex] * vec2[vec2StartIndex]
              + vec1[vec1StartIndex + 1] * vec2[vec2StartIndex + 1]
              + vec1[vec1StartIndex + 2] * vec2[vec2StartIndex + 2];

    return dotprod;
}



