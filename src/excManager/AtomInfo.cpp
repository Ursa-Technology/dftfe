//
// Created by Arghadwip Paul.
//

/*
 * Reads AtomicCoords_Slater and stores
 * atom name, origin (X,Y,Z), and basisfilename
 * in the class Atom
 */

/* DataDirPath is hardcoded in atom.basisFile here */

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "AtomInfo.h"

namespace dftfe
{
  std::vector<Atom>
  Atom::readCoordFile(const std::string &coordFile)
  {
    std::ifstream file(coordFile);
    if (!file.is_open())
      {
        throw std::runtime_error("Could not open file " + coordFile);
      }

    std::vector<Atom> atoms;
    std::string       line;
    while (std::getline(file, line))
      {
        std::istringstream       iss(line);
        std::vector<std::string> words((std::istream_iterator<std::string>(
                                         iss)),
                                       std::istream_iterator<std::string>());

        if (words.size() != 5)
          {
            throw std::runtime_error("Expects only 5 values in coord file " +
                                     coordFile);
          }

        Atom atom;
        atom.name = words[0];
        for (int i = 1; i <= 3; ++i)
          {
            atom.coord.push_back(ANGS_TO_AU * std::stod(words[i]));
          }
        // atom.basisfile = DataDirPath + words[4];
        atom.basisfile = words[4];
        atoms.push_back(atom);
      }

    file.close();
    return atoms;
  }
} // namespace dftfe
