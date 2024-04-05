//
// Created by Arghadwip Paul.
//

#include <cmath>
#include "../include/SlaterPrimitive.h"

SlaterPrimitive::SlaterPrimitive(int n, int l, int m, double a)
        : n(n), l(l), m(m), a(a) {
    double t1 = std::pow(2.0 * a, n + 0.5);
    double t2 = std::sqrt(std::tgamma(2 * n + 1));
    nrm = t1 / t2;
}

double SlaterPrimitive::alpha() const {
    return a;
}

void SlaterPrimitive::nlm(int& out_n, int& out_l, int& out_m) const {
    out_n = n;
    out_l = l;
    out_m = m;
}

double SlaterPrimitive::normConst() const {
    return nrm;
}
