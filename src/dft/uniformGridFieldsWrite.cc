// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Sambit Das
//

// source file for restart functionality in dftClass

//
//
#include <dft.h>
#include <densityCalculator.h>
#include <fileReaders.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <vectorUtilities.h>
#include <linearAlgebraOperations.h>
#include <QuadDataCompositeWrite.h>
#include <MPIWriteOnFile.h>

namespace dftfe
{
  template <unsigned int FEOrder, unsigned int FEOrderElectro,dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro,memorySpace>::computeAndPrintUniformGridFields()
  {
    std::map<dealii::CellId, std::vector<double>> uniformGridQuadPoints;
    std::map<dealii::CellId, std::vector<double>> uniformGridQuadWeights;

    bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> densityUniformGridQuadValues,gradDensityUniformGridQuadValues;
    d_basisOperationsPtrHost->reinit(0, 0, d_uniformGridQuadratureId, false);
    const unsigned int nQuadsPerCell =
      d_basisOperationsPtrHost->nQuadsPerCell();
    const unsigned int nCells = d_basisOperationsPtrHost->nCells();
    densityUniformGridQuadValues.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 :
                                                                       1);
    if (isGradDensityDataDependent)
      {
        gradDensityUniformGridQuadValues.resize(
          d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
      }
    for (unsigned int iComp = 0; iComp < d_densityOutQuadValues.size();
         ++iComp)
      densityUniformGridQuadValues[iComp].resize(nQuadsPerCell * nCells);

    for (unsigned int iComp = 0; iComp < d_gradDensityOutQuadValues.size();
         ++iComp)
      gradDensityUniformGridQuadValues[iComp].resize(3 * nQuadsPerCell * nCells);

#ifdef DFTFE_WITH_DEVICE
        if (d_dftParamsPtr->useDevice)
          computeRhoFromPSI(&d_eigenVectorsFlattenedDevice,
                            &d_eigenVectorsRotFracFlattenedDevice,
                            d_numEigenValues,
                            d_numEigenValuesRR,
                            eigenValues,
                            fermiEnergy,
                            fermiEnergyUp,
                            fermiEnergyDown,
                            d_basisOperationsPtrDevice,
                            d_BLASWrapperPtr,
                            d_densityDofHandlerIndex,
                            d_uniformGridQuadratureId,
                            d_kPointWeights,
                            densityUniformGridQuadValues,
                            gradDensityUniformGridQuadValues,
                            isGradDensityDataDependent,
                            d_mpiCommParent,
                            interpoolcomm,
                            interBandGroupComm,
                            *d_dftParamsPtr,
                            false);
#endif
        if (!d_dftParamsPtr->useDevice)
          computeRhoFromPSI(&d_eigenVectorsFlattenedHost,
                            &d_eigenVectorsRotFracDensityFlattenedHost,
                            d_numEigenValues,
                            d_numEigenValuesRR,
                            eigenValues,
                            fermiEnergy,
                            fermiEnergyUp,
                            fermiEnergyDown,
                            d_basisOperationsPtrHost,
                            d_BLASWrapperPtrHost,
                            d_densityDofHandlerIndex,
                            d_uniformGridQuadratureId,
                            d_kPointWeights,
                            densityUniformGridQuadValues,
                            gradDensityUniformGridQuadValues,
                            isGradDensityDataDependent,
                            d_mpiCommParent,
                            interpoolcomm,
                            interBandGroupComm,
                            *d_dftParamsPtr,
                            false);

 
    const dealii::Quadrature<3> &quadratureFormula =
      matrix_free_data.get_quadrature(d_uniformGridQuadratureId);
    const unsigned int n_q_points = quadratureFormula.size();
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();

    double integralElectronDensity = 0;


    dealii::FEValues<3> feValues(dofHandler.get_fe(),
                                 quadratureFormula,
                                 dealii::update_values |
                                   dealii::update_JxW_values |
                                   dealii::update_quadrature_points);


    cell = dofHandler.begin_active();

    unsigned int icell=0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          feValues.reinit(cell);


          std::vector<double> &uniformGridQuadPointsCell =
            uniformGridQuadPoints[cell->id()];
          uniformGridQuadPointsCell.resize(n_q_points * 3);

          std::vector<double> &uniformGridQuadWeightsCell =
            uniformGridQuadWeights[cell->id()];
          uniformGridQuadWeightsCell.resize(n_q_points);


          const double *cellRhoValues =
                  densityUniformGridQuadValues[0].data() + icell * n_q_points;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              const dealii::Point<3> &quadPoint =
                feValues.quadrature_point(q_point);
              const double jxw = feValues.JxW(q_point);

              uniformGridQuadPointsCell[3 * q_point + 0] = quadPoint[0];
              uniformGridQuadPointsCell[3 * q_point + 1] = quadPoint[1];
              uniformGridQuadPointsCell[3 * q_point + 2] = quadPoint[2];
              uniformGridQuadWeightsCell[q_point]        = jxw;


              integralElectronDensity += cellRhoValues[q_point] * jxw;


            }
          icell++;
        }



    MPI_Allreduce(MPI_IN_PLACE,
                  &integralElectronDensity,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);


    pcout << "integral electron density on uniform quad grid: "
          << integralElectronDensity << std::endl;


    const std::string Path = "uniformGridFieldsData.txt";

    const double HaToEv=27.211396;
    const double HaPerBohrToEvPerAngstrom=51.42208619083232;
    const double BohrToAng=0.529177208;
    const double HaPerBohr3ToeVPerAng3=27.211396/std::pow(BohrToAng,3.0);
    const double perBohr3ToPerAng3=1.0/std::pow(BohrToAng,3.0);


    std::vector<double> shift(3, 0.0);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            shift[i] += d_domainBoundingVectors[j][i] / 2.0;


    const unsigned int poolId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    const unsigned int bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);

    if (poolId == 0 && bandGroupId == 0)
      {
        std::vector<std::shared_ptr<dftUtils::CompositeData>> data(0);

        cell = dofHandler.begin_active();
        icell=0;
        for (; cell != endc; ++cell)
            if (cell->is_locally_owned())
              {
                  const std::vector<double> &uniformGridQuadPointsCell =
                    uniformGridQuadPoints.find(cell->id())->second;

                  const std::vector<double> &uniformGridQuadWeightsCell =
                    uniformGridQuadWeights.find(cell->id())->second;
          
                  const double *cellRhoValues =
                    densityUniformGridQuadValues[0].data() + icell * n_q_points;

                  for (unsigned int q_point = 0; q_point < n_q_points;
                       ++q_point)
                    {

                      std::vector<double> quadVals(0);

                      quadVals.push_back(
                        (uniformGridQuadPointsCell[3 * q_point + 0]+shift[0])*BohrToAng);
                      quadVals.push_back(
                        (uniformGridQuadPointsCell[3 * q_point + 1]+shift[1])*BohrToAng);
                      quadVals.push_back(
                        (uniformGridQuadPointsCell[3 * q_point + 2]+shift[2])*BohrToAng);
                      quadVals.push_back(uniformGridQuadWeightsCell[q_point]/perBohr3ToPerAng3);

                      quadVals.push_back(cellRhoValues[q_point]*perBohr3ToPerAng3);

                      data.push_back(
                        std::make_shared<dftUtils::QuadDataCompositeWrite>(
                          quadVals));
                    }
                  icell++;
              }


        std::vector<dftUtils::CompositeData *> dataRawPtrs(data.size());
        for (unsigned int i = 0; i < data.size(); ++i)
          dataRawPtrs[i] = data[i].get();
        dftUtils::MPIWriteOnFile().writeData(dataRawPtrs,
                                             Path,
                                             mpi_communicator);
      }
  }

#include "dft.inst.cc"
} // namespace dftfe
