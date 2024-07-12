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
#include <kineticEnergyDensityCalculator.h>
#include <fileReaders.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <vectorUtilities.h>
#include <linearAlgebraOperations.h>
#include <QuadDataCompositeWrite.h>
#include <MPIWriteOnFile.h>

namespace dftfe
{
  /*
  namespace internalEnergyDensity
  {
    double
    computeRepulsiveEnergy(
      const std::vector<std::vector<double>> &atomLocationsAndCharge,
      const bool                              isPseudopotential)
    {
      double energy = 0.0;
      for (unsigned int n1 = 0; n1 < atomLocationsAndCharge.size(); n1++)
        {
          for (unsigned int n2 = n1 + 1; n2 < atomLocationsAndCharge.size();
               n2++)
            {
              double Z1, Z2;
              if (isPseudopotential)
                {
                  Z1 = atomLocationsAndCharge[n1][1];
                  Z2 = atomLocationsAndCharge[n2][1];
                }
              else
                {
                  Z1 = atomLocationsAndCharge[n1][0];
                  Z2 = atomLocationsAndCharge[n2][0];
                }
              const dealii::Point<3> atom1(atomLocationsAndCharge[n1][2],
                                           atomLocationsAndCharge[n1][3],
                                           atomLocationsAndCharge[n1][4]);
              const dealii::Point<3> atom2(atomLocationsAndCharge[n2][2],
                                           atomLocationsAndCharge[n2][3],
                                           atomLocationsAndCharge[n2][4]);
              energy += (Z1 * Z2) / atom1.distance(atom2);
            }
        }
      return energy;
    }

  } // namespace internalEnergyDensity

   */

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeAndPrintKE()
  {
    //    std::map<dealii::CellId, std::vector<double>> uniformGridQuadPoints;
    //    std::map<dealii::CellId, std::vector<double>> uniformGridQuadWeights;

    std::map<dealii::CellId, std::vector<double>> kineticEnergyDensityValues;

    //
    //
    //
    //    std::map<dealii::CellId, std::vector<double>> rhoOutValuesUniformQuad;
    //    std::map<dealii::CellId, std::vector<double>>
    //                                                  rhoOutValuesSpinPolarizedUniformQuad;
    //    std::map<dealii::CellId, std::vector<double>>
    //    gradRhoOutValuesUniformQuad; std::map<dealii::CellId,
    //    std::vector<double>>
    //      gradRhoOutValuesSpinPolarizedUniformQuad;

    const dealii::Quadrature<3> &quadratureFormula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    const unsigned int n_q_points = quadratureFormula.size();

    //    typename dealii::DoFHandler<3>::active_cell_iterator
    //      cell = dofHandler.begin_active(),
    //      endc = dofHandler.end();
    //    for (; cell != endc; ++cell)
    //      if (cell->is_locally_owned())
    //        {
    //          const dealii::CellId cellId = cell->id();
    //          (rhoOutValuesUniformQuad)[cellId].resize(
    //            n_q_points,
    //            0.0); //      = std::vector<double>(numQuadPoints, 0.0);
    //          if (d_excManagerPtr->getDensityBasedFamilyType() ==
    //              densityFamilyType::GGA)
    //            (gradRhoOutValuesUniformQuad)[cellId].resize(
    //              3 * n_q_points,
    //              0.0); // =
    //                    // std::vector<double>(3 * numQuadPoints, 0.0);
    //
    //          if (d_dftParamsPtr->spinPolarized == 1)
    //            {
    //              (rhoOutValuesSpinPolarizedUniformQuad)[cellId].resize(
    //                2 * n_q_points, 0.0);
    //              if (d_excManagerPtr->getDensityBasedFamilyType() ==
    //                  densityFamilyType::GGA)
    //                (gradRhoOutValuesSpinPolarizedUniformQuad)[cellId].resize(
    //                  6 * n_q_points, 0.0);
    //            }
    //        }

    //#ifdef DFTFE_WITH_DEVICE
    //    if (d_dftParamsPtr->useDevice)
    //      computeRhoFromPSI(&d_eigenVectorsFlattenedDevice,
    //                        &d_eigenVectorsRotFracFlattenedDevice,
    //                        d_numEigenValues,
    //                        d_numEigenValuesRR,
    //                        eigenValues,
    //                        fermiEnergy,
    //                        fermiEnergyUp,
    //                        fermiEnergyDown,
    //                        basisOperationsPtrDevice,
    //                        d_densityDofHandlerIndex,
    //                        d_densityQuadratureId,
    //                        d_kPointWeights,
    //                        &rhoOutValuesUniformQuad,
    //                        &gradRhoOutValuesUniformQuad,
    //                        &rhoOutValuesSpinPolarizedUniformQuad,
    //                        &gradRhoOutValuesSpinPolarizedUniformQuad,
    //                        d_excManagerPtr->getDensityBasedFamilyType() ==
    //                          densityFamilyType::GGA,
    //                        d_mpiCommParent,
    //                        interpoolcomm,
    //                        interBandGroupComm,
    //                        *d_dftParamsPtr,
    //                        false);
    //#endif
    //    if (!d_dftParamsPtr->useDevice)
    //      computeRhoFromPSI(&d_eigenVectorsFlattenedHost,
    //                        &d_eigenVectorsRotFracDensityFlattenedHost,
    //                        d_numEigenValues,
    //                        d_numEigenValuesRR,
    //                        eigenValues,
    //                        fermiEnergy,
    //                        fermiEnergyUp,
    //                        fermiEnergyDown,
    //                        basisOperationsPtrHost,
    //                        d_densityDofHandlerIndex,
    //                        d_densityQuadratureId,
    //                        d_kPointWeights,
    //                        &rhoOutValuesUniformQuad,
    //                        &gradRhoOutValuesUniformQuad,
    //                        &rhoOutValuesSpinPolarizedUniformQuad,
    //                        &gradRhoOutValuesSpinPolarizedUniformQuad,
    //                        d_excManagerPtr->getDensityBasedFamilyType() ==
    //                          densityFamilyType::GGA,
    //                        d_mpiCommParent,
    //                        interpoolcomm,
    //                        interBandGroupComm,
    //                        *d_dftParamsPtr,
    //                        false);



    //
    // compute kinetic energy density values
    //
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      computeKineticEnergyDensity(*d_BLASWrapperPtr,
                                  &d_eigenVectorsFlattenedDevice,
                                  d_numEigenValues,
                                  eigenValues,
                                  fermiEnergy,
                                  fermiEnergyUp,
                                  fermiEnergyDown,
                                  d_basisOperationsPtrDevice,
                                  d_densityQuadratureId,
                                  d_kPointCoordinates,
                                  d_kPointWeights,
                                  kineticEnergyDensityValues,
                                  d_mpiCommParent,
                                  interpoolcomm,
                                  interBandGroupComm,
                                  mpi_communicator,
                                  *d_dftParamsPtr);
#endif
    if (!d_dftParamsPtr->useDevice)
      computeKineticEnergyDensity(*d_BLASWrapperPtrHost,
                                  &d_eigenVectorsFlattenedHost,
                                  d_numEigenValues,
                                  eigenValues,
                                  fermiEnergy,
                                  fermiEnergyUp,
                                  fermiEnergyDown,
                                  d_basisOperationsPtrHost,
                                  d_densityQuadratureId,
                                  d_kPointCoordinates,
                                  d_kPointWeights,
                                  kineticEnergyDensityValues,
                                  d_mpiCommParent,
                                  interpoolcomm,
                                  interBandGroupComm,
                                  mpi_communicator,
                                  *d_dftParamsPtr);

    MPI_Barrier(MPI_COMM_WORLD);



    double kineticEnergy = 0;


    dealii::FEValues<3> feValues(dofHandler.get_fe(),
                                 quadratureFormula,
                                 dealii::update_values |
                                   dealii::update_JxW_values |
                                   dealii::update_quadrature_points);



    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          feValues.reinit(cell);

          const std::vector<double> &kineticEnergyDensityValuesCell =
            kineticEnergyDensityValues.find(cell->id())->second;



          //          std::vector<double> &uniformGridQuadPointsCell =
          //            uniformGridQuadPoints[cell->id()];
          //          uniformGridQuadPointsCell.resize(n_q_points * 3);
          //
          //          std::vector<double> &uniformGridQuadWeightsCell =
          //            uniformGridQuadWeights[cell->id()];
          //          uniformGridQuadWeightsCell.resize(n_q_points);
          //
          //
          //          const std::vector<double> &rhoValuesCell =
          //            rhoOutValuesUniformQuad.find(cell->id())->second;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              const dealii::Point<3> &quadPoint =
                feValues.quadrature_point(q_point);
              const double jxw = feValues.JxW(q_point);

              kineticEnergy += kineticEnergyDensityValuesCell[q_point] * jxw;
            }
        }


    MPI_Allreduce(
      MPI_IN_PLACE, &kineticEnergy, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);


    pcout << "kinetic energy: " << kineticEnergy << std::endl;

    return kineticEnergy;
  }

#include "dft.inst.cc"
} // namespace dftfe
