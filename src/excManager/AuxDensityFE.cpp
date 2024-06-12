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
    const std::vector<double> &                                     points,
    std::map<DensityDescriptorDataAttributes, std::vector<double>> &densityData)
  {
    std::pair<size_t, size_t> indexRange;

    unsigned int minIndex = 0;
    for (unsigned int i = 0; i < d_quadWeightsAll.size(); i++)
      {
        if (std::abs(points[0] - d_quadPointsAll[3 * i + 0]) +
              std::abs(points[1] - d_quadPointsAll[3 * i + 1]) +
              std::abs(points[2] - d_quadPointsAll[3 * i + 2]) <
            1e-6)
          {
            minIndex = i;
            break;
          }
      }

    unsigned int maxIndex = 0;
    for (unsigned int i = 0; i < d_quadWeightsAll.size(); i++)
      {
        if (std::abs(points[points.size() - 3] - d_quadPointsAll[3 * i + 0]) +
              std::abs(points[points.size() - 2] - d_quadPointsAll[3 * i + 1]) +
              std::abs(points[points.size() - 1] - d_quadPointsAll[3 * i + 2]) <
            1e-6)
          {
            maxIndex = i;
            break;
          }
      }

    indexRange.first  = minIndex;
    indexRange.second = maxIndex;

    if (densityData.find(DensityDescriptorDataAttributes::valuesTotal) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesTotal],
          d_densityValsTotalAllQuads,
          indexRange);
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinUp) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesSpinUp],
          d_densityValsSpinUpAllQuads,
          indexRange);
      }

    if (densityData.find(DensityDescriptorDataAttributes::valuesSpinDown) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::valuesSpinDown],
          d_densityValsSpinDownAllQuads,
          indexRange);
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValueSpinUp) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::gradValueSpinUp],
          d_gradDensityValsSpinUpAllQuads,
          indexRange);
      }

    if (densityData.find(DensityDescriptorDataAttributes::gradValueSpinDown) ==
        densityData.end())
      {
        this->fillDensityAttributeData(
          densityData[DensityDescriptorDataAttributes::gradValueSpinDown],
          d_gradDensityValsSpinDownAllQuads,
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
    d_quadPointsAll  = projectionInputs["quadpts"];
    d_quadWeightsAll = projectionInputs["quadWt"];
    const std::vector<double> &densityVals =
      projectionInputs["densityFunc"]->second;
    const unsigned int nQ = d_quadPointsAll.size();
    d_densityValsTotalAllquads.resize(nQ, 0);
    d_densityValsSpinUpAllQuads.resize(nQ, 0);
    d_densityValsSpinDownAllQuads.resize(nQ, 0);
    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      d_densityValsSpinUpAllQuads[iquad] = densityVals[iquad];

    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      d_densityValsSpinDownAllQuads[iquad] = densityVals[nQ + iquad];

    for (unsigned int iquad = 0; iquad < nQ; iquad++)
      d_densityValsTotalAllQuads[iquad] = d_densityValsSpinUpAllQuads[iquad] +
                                          d_densityValsSpinDownAllQuads[iquad];

    if (projectionInputs.find("gradDensityFunc") != projectionInputs.end())
      {
        const std::vector<double> &gradDensityVals =
          projectionInputs["gradDensityFunc"];
        d_gradDensityValsSpinUpAllQuads.resize(nQ * 3, 0);
        d_gradDensityValsSpinDownAllQuads.resize(nQ * 3, 0);

        for (unsigned int iquad = 0; iquad < nQ; iquad++)
          for (idim = 0; idim < 3; idim++)
            d_gradDensityValsSpinUpAllQuads[3 * iquad + idim] =
              gradDensityVals[3 * iquad + idim];

        for (unsigned int iquad = 0; iquad < nQ; iquad++)
          for (idim = 0; idim < 3; idim++)
            d_gradDensityValsSpinDownAllQuads[3 * iquad + idim] =
              gradDensityVals[3 * nQ + 3 * iquad + idim];
      }
  }


  void
  AuxDensityFE::projectDensityEnd()
  {}


} // namespace dftfe
