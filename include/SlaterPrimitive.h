//
// Created by Arghadwip Paul.
//

#ifndef SLATER2_SLATERPRIMITIVE_H
#define SLATER2_SLATERPRIMITIVE_H

#include <vector>
#include <string>

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

#endif // SLATER2_SLATERPRIMITIVE_H
