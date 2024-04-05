//
// Created by Arghadwip Paul.
//

#ifndef LIBXC_CLASS_BASEXCHANDLER_H
#define LIBXC_CLASS_BASEXCHANDLER_H

#include <string>
#include <xc.h>
#include <vector>

class BaseXCHandler {
private:
    bool hasX;
    bool hasC;
    std::string spin;
    int nspin;
    xc_func_type func_X;
    xc_func_type func_C;

    std::vector<double> getGradSigma(const std::vector<double>& drho,
                                     const std::vector<double>& ddrho,
                                     int nPoints,
                                     int nSigma);
    std::vector<double> getg(const std::vector<double>& drho,
                             int nPoints,
                             int ii,
                             int alpha);
    std::vector<double> getDivg(const std::vector<double>& ddrho,
                                int nPoints,
                                int ii,
                                int alpha);

    std::vector<double> getv(const std::vector<double>& vrho,
                             const std::vector<double>& vsigma,
                             const std::vector<double>& v2rhosigma,
                             const std::vector<double>& v2sigma2,
                             const std::vector<double>& drho,
                             const std::vector<double>& ddrho,
                             int nPoints);
    std::vector<double> multiplyDDrhoDrho(const std::vector<double>& ddrho,
                                          const std::vector<double>& drho,
                                          int ddrhoStartIndex,
                                          int drhoStartIndex);
    double  vecvecDotproduct(const std::vector<double>& vec1,
                             const std::vector<double>& vec2,
                             int vec1StartIndex,
                             int vec2StartIndex);

public:
    BaseXCHandler(const std::string& XFunctionalName,
                  const std::string& CFunctionalName,
                  const std::string& spin);

    void getexcAndvxc(const std::vector<double>& rho,
                      const std::vector<double>& drho,
                      const std::vector<double>& ddrho,
                      const std::vector<double>& sigma,
                      std::vector<double>& exc,
                      std::vector<double>& vxc,
                      int nPoints);
};

#endif //LIBXC_CLASS_BASEXCHANDLER_H
