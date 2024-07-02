//
// Created by Arghadwip Paul.
//

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <iostream>

#include "SlaterBasisSet.h"

namespace dftfe
{
  SlaterBasisSet::SlaterBasisSet()
  {
    // empty constructor
  }

  SlaterBasisSet::~SlaterBasisSet()
  {
    // deallocate SlaterPrimitive pointers stored in the map
    for (auto &pair : d_atomToSlaterPrimitivePtr)
      {
        std::vector<SlaterPrimitive *> &primitives = pair.second;
        for (SlaterPrimitive *sp : primitives)
          {
            if (sp != nullptr)
              {
                delete sp;
              }
          }
        primitives.clear();
      }
  }


  void
  SlaterBasisSet::constructBasisSet(
    const std::vector<std::pair<std::string, std::vector<double>>> &atomCoords,
    const std::string &auxBasisFileName)
  {
    const auto atomToSlaterBasisName =
      readAtomToSlaterBasisName(auxBasisFileName);

    for (const auto &pair : atomToSlaterBasisName)
      {
        const std::string &atomSymbol = pair.first;
        const std::string &basisName  = pair.second;

        this->addSlaterPrimitivesFromBasisFile(atomSymbol, basisName);
      }


    for (const auto &pair : atomCoords)
      {
        const std::string &        atomSymbol = pair.first;
        const std::vector<double> &atomCenter = pair.second;

        for (const SlaterPrimitive *sp : d_atomToSlaterPrimitivePtr[atomSymbol])
          {
            SlaterBasisInfo info = {atomSymbol, atomCenter, sp};
            d_SlaterBasisInfo.push_back(info);
          }
      }
  }

  const std::vector<SlaterBasisInfo> &
  SlaterBasisSet::getSlaterBasisInfo() const
  {
    return d_SlaterBasisInfo;
  }

  int
  SlaterBasisSet::getSlaterBasisSize() const
  {
    return static_cast<int>(d_SlaterBasisInfo.size());
  }

  void
  SlaterBasisSet::addSlaterPrimitivesFromBasisFile(
    const std::string &atomSymbol,
    const std::string &basisName)
  {
    /*
     * Written in a format that ignores the first line
     */
    std::ifstream file("../Data/" + basisName);
    if (file.is_open())
      {
        std::string                          line;
        std::unordered_map<std::string, int> lStringToIntMap = {
          {"S", 0}, {"P", 1}, {"D", 2}, {"F", 3}, {"G", 4}, {"H", 5}};
        // First line - Ignore
        std::getline(file, line);
        std::istringstream iss(line);
        if (iss.fail())
          {
            throw std::runtime_error("Error reading line in file: " +
                                     basisName);
          }
        std::string atomType, extra;
        iss >> atomType;
        if (iss >> extra)
          {
            throw std::runtime_error("Error: More than one entry in line 1 in" +
                                     basisName);
          }
        // Second Line Onwards
        while (std::getline(file, line))
          {
            std::istringstream iss(line);
            if (iss.fail())
              {
                throw std::runtime_error("Error reading line in file: " +
                                         basisName);
              }

            std::string nlString;
            double      alpha;
            iss >> nlString >> alpha;
            if (iss >> extra)
              {
                throw std::runtime_error(
                  "Error: More than two entries in a line in" + basisName);
              }

            char lChar = nlString.back();

            int n = std::stoi(nlString.substr(0, nlString.size() - 1));
            int l;
            try
              {
                l = lStringToIntMap.at(std::string(1, lChar));
              }
            catch (const std::out_of_range &e)
              {
                throw std::runtime_error(
                  std::string(
                    "Character doesn't exist in the lStringToIntMap in "
                    "SlaterBasisSet.cpp: ") +
                  std::string(1, lChar));
              }
            std::vector<int> mList;
            if (l == 1)
              {
                mList = {1, -1, 0}; // Special case for p orbitals
              }
            else
              {
                for (int m = -l; m <= l; ++m)
                  {
                    mList.push_back(m);
                  }
              }
            for (int m : mList)
              {
                SlaterPrimitive *sp = new SlaterPrimitive{n, l, m, alpha};
                d_atomToSlaterPrimitivePtr[atomSymbol].push_back(sp);
              }
          }
      }
    else
      {
        throw std::runtime_error("Unable to open file: " + basisName);
      }
  }


  std::unordered_map<std::string, std::string>
  SlaterBasisSet::readAtomToSlaterBasisName(const std::string &fileName)
  {
    std::unordered_map<std::string, std::string> atomToSlaterBasisName;
    std::ifstream                                file(fileName);
    if (file.is_open())
      {
        std::string line;
        int         lineNumber = 0;
        while (std::getline(file, line))
          {
            lineNumber++;
            std::istringstream iss(line);
            std::string        atomSymbol;
            std::string        slaterBasisName;

            if (iss >> atomSymbol >> slaterBasisName)
              {
                std::string extra;
                if (iss >> extra)
                  {
                    throw std::runtime_error(
                      "Error: More than two entries in line " +
                      std::to_string(lineNumber) + "in" + fileName);
                  }
                atomToSlaterBasisName[atomSymbol] = slaterBasisName;
              }
            else
              {
                throw std::runtime_error("Error: Invalid format in line " +
                                         std::to_string(lineNumber) + "in" +
                                         fileName);
              }
          }
        file.close();
      }
    else
      {
        std::cout << "Unable to open the file." + fileName << std::endl;
      }
    return atomToSlaterBasisName;
  }

} // namespace dftfe
