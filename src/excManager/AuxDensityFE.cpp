//
// Created by Sambit Das.
//

#include "AuxDensityFE.h"
#include <Exceptions.h>

namespace dftfe
{
  namespace
  {
    void
    fillDensityAttributeData(std::vector<double> &            attributeData,
                             const std::vector<double> &      values,
                             const std::pair<size_t, size_t> &indexRange)
    {
      size_t startIndex = indexRange.first;
      size_t endIndex   = indexRange.second;

      if (startIndex > endIndex || endIndex >= attributeData.size() ||
          endIndex >= values.size())
        {
          throw std::invalid_argument("Invalid index range for densityData");
        }

      for (size_t i = startIndex; i <= endIndex; ++i)
        {
          attributeData[i] = values[i];
        }
    }
  } // namespace


  void
  AuxDensityFE::applyLocalOperations(
    const std::vector<double> &                                     Points,
    std::map<DensityDescriptorDataAttributes, std::vector<double>> &densityData)
  {
    if (densityData.find(DensityDescriptorDataAttributes::valuesTotal) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesTotal],
          d_densityValsTotal,
          indexRange);
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinUp) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesSpinUp],
          d_densityValsSpinUp,
          indexRange);
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinDown) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesSpinDown],
          d_densityValsSpinDown,
          indexRange);
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValueSpinUp) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::gradValueSpinUp],
          d_gradDensityValsSpinUp,
          indexRange);
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValueSpinDown) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::gradValueSpinDown],
          d_gradDensityValsSpinDown,
          indexRange);
      }
  }

  void
  AuxDensityFE::projectDensityMatrixStart(
    std::unordered_map<std::string, std::vector<double>> &projectionInputs)

  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }

  void
  AuxDensityFE::projectDensityMatrixEnd()
  {
    std::string errMsg = "Not implemented";
    dftfe::utils::throwException(false, errMsg);
  }



  void
  AuxDensityFE::projectDensityStart(
    std::unordered_map<std::string, std::vector<double>> &projectionInputs);

  {
    d_quadPointsSet  = QPts;
    d_quadWeightsSet = QWt;
    d_densityValsTotal.resize(nQ, 0);
    d_densityValsSpinUp.resize(nQ, 0);
    d_densityValsSpinDown.resize(nQ, 0);
    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      d_densityValsSpinUp[iquad] = densityVals[iquad];

    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      d_densityValsSpinDown[iquad] = densityVals[nQ + iquad];

    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      d_densityValsTotal[iquad] =
        d_densityValsSpinUp[iquad] + d_densityValsSpinDown[iquad];


    d_gradDensityValsSpinUp.resize(nQ * 3, 0);
    d_gradDensityValsSpinDown.resize(nQ * 3, 0);

    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      for (idim = 0; idim < 3; idim++)
        d_gradDensityValsSpinUp[3 * iquad + idim] =
          gradDensityVals[3 * iquad + idim];

    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      for (idim = 0; idim < 3; idim++)
        d_gradDensityValsSpinDown[3 * iquad + idim] =
          gradDensityVals[3 * nQ + 3 * iquad + idim];
  }


  void
  AuxDensityFE::projectDensityEnd()
  {}


} // namespace dftfe
