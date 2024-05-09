//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_SLATER_SLATERPRIMITIVE_H
#define DFTFE_SLATER_SLATERPRIMITIVE_H

#include <vector>
#include <string>

namespace dftfe
{
  class SlaterPrimitive
  {
  public:
    SlaterPrimitive(int n, int l, int m, double a);

    double
    alpha() const;

    void
    nlm(int &n, int &l, int &m) const;

    double
    normConst() const;

  private:
    int    n, l, m;
    double a, nrm;
  };
} // namespace dftfe

#endif // SLATER_SLATERPRIMITIVE_H
