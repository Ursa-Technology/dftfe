//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_SLATER_SLATERBASISSET_H
#define DFTFE_SLATER_SLATERBASISSET_H

#include <vector>
#include <unordered_map>
#include <string>
#include <utility>

namespace dftfe
{
  struct SlaterPrimitive
  {
    int    n;     // principal quantum number
    int    l;     // azimuthal (angular) quantum number
    int    m;     // magnetic quantum number
    double alpha; // exponent of the basis
  };

  struct SlaterBasisInfo
  {
    std::string            symbol; // atom name
    std::vector<double>    center; // atom center coordinates
    const SlaterPrimitive *sp;     // pointer to the Slater basis
  };

  class SlaterBasisSet
  {
  private:
    // std::unordered_map<std::string, std::string> d_atomToSlaterBasisName;

    std::unordered_map<std::string, std::vector<SlaterPrimitive *>>
      d_atomToSlaterPrimitivePtr;

    std::vector<SlaterBasisInfo> d_SlaterBasisInfo;

    std::unordered_map<std::string, std::string>
    readAtomToSlaterBasisName(const std::string &fileName);

    void
    addSlaterPrimitivesFromBasisFile(const std::string &atomSymbol,
                                     const std::string &basisName);

  public:
    SlaterBasisSet();  // Constructor
    ~SlaterBasisSet(); // Destructor


    void
    constructBasisSet(
      const std::vector<std::pair<std::string, std::vector<double>>>
        &                atomCoords,
      const std::string &auxBasisFileName);


    const std::vector<SlaterBasisInfo> &
    getSlaterBasisInfo() const;

    int
    getSlaterBasisSize() const;
  };
} // namespace dftfe

#endif // DFTFE_SLATER_SLATERBASISSET_H
