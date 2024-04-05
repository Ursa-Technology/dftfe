//
// Created by Arghadwip Paul.
//

#ifndef SLATER2_SLATERBASISSET_H
#define SLATER2_SLATERBASISSET_H

#include <vector>
#include <unordered_map>
#include <string>
#include "AtomInfo.h"
#include "SlaterPrimitive.h"

class SlaterBasisSet {
private:
    std::unordered_map<std::string, std::vector<SlaterPrimitive>> basisSets;
    std::unordered_map<std::string, std::vector<SlaterPrimitive>> basisSize;

public:
    // SlaterBasisSet(const std::vector<Atom>& atoms);
    void constructBasisSet(const std::vector<Atom>& atoms);
    static std::vector<SlaterPrimitive> readSlaterBasisFile(const std::string &filename);
    std::vector<SlaterPrimitive> getBasisSet(const std::string& basisFileName) const;
    static void displayBasisSetInfo(const std::vector<SlaterPrimitive> &basisList);
    static int getBasisSize(const std::vector<SlaterPrimitive> &basisList);
    int getTotalBasisSize(const std::vector<Atom>& atoms);
};



#endif //SLATER2_SLATERBASISSET_H
