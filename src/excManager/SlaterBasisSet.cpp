//
// Created by Arghadwip Paul.
//

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <iostream>

#include "../include/SlaterBasisSet.h"

/*
// Implement the constructor
SlaterBasisSet::
    SlaterBasisSet(const std::vector<Atom>& atoms) {

    for (const auto& atom : atoms) {
        // Check if the basisfile has already been processed
        if (basisSets.find(atom.basisfile) == basisSets.end()) {
            basisSets[atom.basisfile] = readSlaterBasisFile(atom.basisfile);
        }
    }
}*/

void
SlaterBasisSet::constructBasisSet(const std::vector<Atom>& atoms)
{
    for (const auto& atom : atoms) {
        // Check if the basisfile has already been processed
        if (basisSets.find(atom.basisfile) == basisSets.end()) {
            basisSets[atom.basisfile] = readSlaterBasisFile(atom.basisfile);
        }
    }
}



std::vector<SlaterPrimitive>
            SlaterBasisSet::
            readSlaterBasisFile(const std::string &filename) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::vector<SlaterPrimitive> basisList;
    int nBasis = 0;
    std::string line;
    std::map<std::string, int> lStringToIntMap = {
            {"S", 0}, {"P", 1}, {"D", 2},
            {"F", 3}, {"G", 4}, {"H", 5}
    };

    // Ignore the first line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string nlString;
        double alpha;
        iss >> nlString >> alpha;

        if (iss.fail()) {
            throw std::runtime_error("Error reading line in file: " + filename);
        }

        char lChar = nlString.back();

        int n = std::stoi(nlString.substr(0, nlString.size() - 1));
        int l = lStringToIntMap[std::string(1, lChar)];

        std::vector<int> mList;
        if (l == 1) {
            mList = {1, -1, 0}; // Special case for p orbitals
        } else {
            for (int m = -l; m <= l; ++m) {
                mList.push_back(m);
            }
        }

        for (int m : mList) {
            nBasis += 1;
            basisList.emplace_back(n, l, m, alpha);
        }
    }

    return basisList;
}

std::vector<SlaterPrimitive>
            SlaterBasisSet::
            getBasisSet(const std::string& basisFileName) const {

    auto it = basisSets.find(basisFileName);
    if (it != basisSets.end()) {
        return it->second;
    }
    return {}; // Return an empty vector if the basis name is not found
}

void SlaterBasisSet::
    displayBasisSetInfo(const std::vector<SlaterPrimitive> &basisList) {

    for (const auto& basis : basisList) {
        int n, l, m;
        basis.nlm(n, l, m);  // Retrieve the n, l, m values

        std::cout << "Slater Primitive: n=" << n
                  << ", l=" << l
                  << ", m=" << m
                  << ", alpha=" << basis.alpha()
                  << ", normalization constant=" << basis.normConst()
                  << std::endl;
    }

    // std::cout << "basis size =" << basisList.size() << std::endl;

}

int SlaterBasisSet::
    getBasisSize(const std::vector<SlaterPrimitive> &basisList) {
    return basisList.size() ;
}

int SlaterBasisSet::
getTotalBasisSize(const std::vector<Atom>& atoms) {

    int nBasis = 0;
    for (const auto& atom : atoms) {
        auto cBasisSet = this -> getBasisSet(atom.basisfile);
        nBasis += this -> getBasisSize(cBasisSet);
    }

    return nBasis;
}