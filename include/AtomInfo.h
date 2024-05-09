//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_SLATER_ATOMINFO_H
#define DFTFE_SLATER_ATOMINFO_H

#include <vector>
#include <string>
#include "SlaterPrimitive.h"

// Conversion factor from Angstroms to atomic units
const double ANGS_TO_AU = 1.889726124;

// const std::string DataDirPath = "../Data/";

namespace dftfe
{
  class Atom
  {
  public:
    std::string         name;
    std::vector<double> coord;
    std::string         basisfile;

    static std::vector<Atom>
    readCoordFile(const std::string &coordFile);
  };
} // namespace dftfe

#endif // DFTFE_SLATER_ATOMINFO_H
