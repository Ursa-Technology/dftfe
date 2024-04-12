//
// Created by Arghadwip Paul.
//

#ifndef SLATER2_ATOMINFO_H
#define SLATER2_ATOMINFO_H

#include <vector>
#include <string>
#include "SlaterPrimitive.h"

const double ANGS_TO_AU =
  1.889726124; // Conversion factor from Angstroms to atomic units
const std::string DataDirPath = "../Data/";

class Atom
{
public:
  std::string         name;
  std::vector<double> coord;
  std::string         basisfile;

  static std::vector<Atom>
  readCoordFile(const std::string &coordFile);
};

#endif // SLATER2_ATOMINFO_H
