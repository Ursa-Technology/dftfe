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
#include <AuxDensityMatrixFE.h>

namespace dftfe
{
  template <unsigned int FEOrder, unsigned int FEOrderElectro,dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro,memorySpace>::initCoreRhoUniformQuad(
    const unsigned int                             uniformQuadratureId,
    const dealii::MatrixFree<3, double> &          _matrix_free_data,
    std::map<dealii::CellId, std::vector<double>> &rhoCoreUniformQuad,
    std::map<dealii::CellId, std::vector<double>> &gradRhoCoreUniformQuad)
  {
    // clear existing data
    rhoCoreUniformQuad.clear();
    gradRhoCoreUniformQuad.clear();

    // Reading single atom rho initial guess
    pcout
      << std::endl
      << "Reading data for core electron-density to be used in nonlinear core-correction....."
      << std::endl;
    std::map<unsigned int, alglib::spline1dinterpolant> coreDenSpline;
    std::map<unsigned int, std::vector<std::vector<double>>>
                                         singleAtomCoreElectronDensity;
    std::map<unsigned int, double>       outerMostPointCoreDen;
    std::map<unsigned int, unsigned int> atomTypeNLCCFlagMap;
    const double                         truncationTol = 1e-12;
    unsigned int                         fileReadFlag  = 0;

    double maxCoreRhoTail = 0.0;
    // loop over atom types
    for (std::set<unsigned int>::iterator it = atomTypes.begin();
         it != atomTypes.end();
         it++)
      {
        char coreDensityFile[256];
        if (d_dftParamsPtr->isPseudopotential)
          {
            strcpy(coreDensityFile,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/coreDensity.inp")
                     .c_str());
          }

        unsigned int fileReadFlag =
          dftUtils::readPsiFile(2,
                                singleAtomCoreElectronDensity[*it],
                                coreDensityFile);

        atomTypeNLCCFlagMap[*it] = fileReadFlag;

        if (d_dftParamsPtr->verbosity >= 4)
          pcout << "Atomic number: " << *it << " NLCC flag: " << fileReadFlag
                << std::endl;

        if (fileReadFlag > 0)
          {
            unsigned int numRows =
              singleAtomCoreElectronDensity[*it].size() - 1;
            std::vector<double> xData(numRows), yData(numRows);

            unsigned int maxRowId = 0;
            for (unsigned int irow = 0; irow < numRows; ++irow)
              {
                xData[irow] = singleAtomCoreElectronDensity[*it][irow][0];
                yData[irow] =
                  std::abs(singleAtomCoreElectronDensity[*it][irow][1]);

                if (yData[irow] > truncationTol)
                  maxRowId = irow;
              }

            // interpolate rho
            alglib::real_1d_array x;
            x.setcontent(numRows, &xData[0]);
            alglib::real_1d_array y;
            y.setcontent(numRows, &yData[0]);
            alglib::ae_int_t natural_bound_type_L = 1;
            alglib::ae_int_t natural_bound_type_R = 1;
            // const double slopeL = (singleAtomCoreElectronDensity[*it][1][1]-
            // singleAtomCoreElectronDensity[*it][0][1])/(singleAtomCoreElectronDensity[*it][1][0]-singleAtomCoreElectronDensity[*it][0][0]);
            // const double slopeL = (yData[1]- yData[0])/(xData[1]-xData[0]);
            spline1dbuildcubic(x,
                               y,
                               numRows,
                               natural_bound_type_L,
                               0.0,
                               natural_bound_type_R,
                               0.0,
                               coreDenSpline[*it]);
            // spline1dbuildcubic(x, y, numRows, natural_bound_type_L, slopeL,
            // natural_bound_type_R, 0.0, coreDenSpline[*it]);
            outerMostPointCoreDen[*it] = xData[maxRowId];

            if (outerMostPointCoreDen[*it] > maxCoreRhoTail)
              maxCoreRhoTail = outerMostPointCoreDen[*it];

            if (d_dftParamsPtr->verbosity >= 4)
              pcout << " Atomic number: " << *it
                    << " Outermost Point Core Den: "
                    << outerMostPointCoreDen[*it] << std::endl;
          }
      }

    const double cellCenterCutOff = maxCoreRhoTail + 5.0;

    //
    // Initialize rho
    //
    const dealii::Quadrature<3> &quadrature_formula =
      _matrix_free_data.get_quadrature(uniformQuadratureId);
    dealii::FEValues<3> fe_values(FE,
                                  quadrature_formula,
                                  dealii::update_quadrature_points);
    const unsigned int  n_q_points = quadrature_formula.size();

    //
    // get number of global charges
    //
    const int numberGlobalCharges = atomLocations.size();

    //
    // get number of image charges used only for periodic
    //
    const int numberImageCharges = d_imageIdsTrunc.size();

    //
    // loop over elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    dealii::Tensor<1, 3, double> zeroTensor1;
    for (unsigned int i = 0; i < 3; i++)
      zeroTensor1[i] = 0.0;

    dealii::Tensor<2, 3, double> zeroTensor2;

    for (unsigned int i = 0; i < 3; i++)
      for (unsigned int j = 0; j < 3; j++)
        zeroTensor2[i][j] = 0.0;

    // loop over elements
    //
    cell = dofHandler.begin_active();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            std::vector<double> &rhoCoreQuadValues =
              rhoCoreUniformQuad[cell->id()];
            rhoCoreQuadValues.resize(n_q_points, 0.0);

            std::vector<double> &gradRhoCoreQuadValues =
              gradRhoCoreUniformQuad[cell->id()];
            gradRhoCoreQuadValues.resize(n_q_points * 3, 0.0);

            std::vector<dealii::Tensor<1, 3, double>> gradRhoCoreAtom(
              n_q_points, zeroTensor1);



            // loop over atoms
            for (unsigned int iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
              {
                dealii::Point<3> atom(atomLocations[iAtom][2],
                                      atomLocations[iAtom][3],
                                      atomLocations[iAtom][4]);
                bool             isCoreRhoDataInCell = false;

                if (atomTypeNLCCFlagMap[atomLocations[iAtom][0]] == 0)
                  continue;

                if (atom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                // loop over quad points
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    dealii::Point<3> quadPoint = fe_values.quadrature_point(q);
                    dealii::Tensor<1, 3, double> diff = quadPoint - atom;
                    double distanceToAtom = quadPoint.distance(atom);

                    if (d_dftParamsPtr->floatingNuclearCharges &&
                        distanceToAtom < 1.0e-4)
                      {
                        if (d_dftParamsPtr->verbosity >= 4)
                          std::cout
                            << "Atom close to quad point, iatom: " << iAtom
                            << std::endl;

                        distanceToAtom = 1.0e-4;
                        diff[0]        = (1.0e-4) / std::sqrt(3.0);
                        diff[1]        = (1.0e-4) / std::sqrt(3.0);
                        diff[2]        = (1.0e-4) / std::sqrt(3.0);
                      }

                    double value, radialDensityFirstDerivative,
                      radialDensitySecondDerivative;
                    if (distanceToAtom <=
                        outerMostPointCoreDen[atomLocations[iAtom][0]])
                      {
                        alglib::spline1ddiff(
                          coreDenSpline[atomLocations[iAtom][0]],
                          distanceToAtom,
                          value,
                          radialDensityFirstDerivative,
                          radialDensitySecondDerivative);

                        isCoreRhoDataInCell = true;
                      }
                    else
                      {
                        value                        = 0.0;
                        radialDensityFirstDerivative = 0.0;
                      }

                    rhoCoreQuadValues[q] += value;
                    gradRhoCoreAtom[q] =
                      radialDensityFirstDerivative * diff / distanceToAtom;
                    gradRhoCoreQuadValues[3 * q + 0] += gradRhoCoreAtom[q][0];
                    gradRhoCoreQuadValues[3 * q + 1] += gradRhoCoreAtom[q][1];
                    gradRhoCoreQuadValues[3 * q + 2] += gradRhoCoreAtom[q][2];

                  } // end loop over quad points
              }     // loop over atoms

            // loop over image charges
            for (unsigned int iImageCharge = 0;
                 iImageCharge < numberImageCharges;
                 ++iImageCharge)
              {
                const int masterAtomId = d_imageIdsTrunc[iImageCharge];
                if (atomTypeNLCCFlagMap[atomLocations[masterAtomId][0]] == 0)
                  continue;

                dealii::Point<3> imageAtom(
                  d_imagePositionsTrunc[iImageCharge][0],
                  d_imagePositionsTrunc[iImageCharge][1],
                  d_imagePositionsTrunc[iImageCharge][2]);

                if (imageAtom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                bool isCoreRhoDataInCell = false;

                // loop over quad points
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    dealii::Point<3> quadPoint = fe_values.quadrature_point(q);
                    dealii::Tensor<1, 3, double> diff = quadPoint - imageAtom;
                    double distanceToAtom = quadPoint.distance(imageAtom);

                    if (d_dftParamsPtr->floatingNuclearCharges &&
                        distanceToAtom < 1.0e-4)
                      {
                        distanceToAtom = 1.0e-4;
                        diff[0]        = (1.0e-4) / std::sqrt(3.0);
                        diff[1]        = (1.0e-4) / std::sqrt(3.0);
                        diff[2]        = (1.0e-4) / std::sqrt(3.0);
                      }

                    double value, radialDensityFirstDerivative,
                      radialDensitySecondDerivative;
                    if (distanceToAtom <=
                        outerMostPointCoreDen[atomLocations[masterAtomId][0]])
                      {
                        alglib::spline1ddiff(
                          coreDenSpline[atomLocations[masterAtomId][0]],
                          distanceToAtom,
                          value,
                          radialDensityFirstDerivative,
                          radialDensitySecondDerivative);

                        isCoreRhoDataInCell = true;
                      }
                    else
                      {
                        value                        = 0.0;
                        radialDensityFirstDerivative = 0.0;
                      }

                    rhoCoreQuadValues[q] += value;
                    gradRhoCoreAtom[q] =
                      radialDensityFirstDerivative * diff / distanceToAtom;

                    gradRhoCoreQuadValues[3 * q + 0] += gradRhoCoreAtom[q][0];
                    gradRhoCoreQuadValues[3 * q + 1] += gradRhoCoreAtom[q][1];
                    gradRhoCoreQuadValues[3 * q + 2] += gradRhoCoreAtom[q][2];

                  } // quad point loop

              } // end of image charges

          } // cell locally owned check

      } // cell loop
    pcout
      << std::endl
      << "Finished Reading data for core electron-density to be used in nonlinear core-correction....."
      << std::endl;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro,dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro,memorySpace>::computeXCEnergyDensityUniformQuad(
    const std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &                                            basisOperationsPtr,
    const unsigned int                             quadratureId,
    const std::shared_ptr<excManager<memorySpace>> excManagerPtr,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityOutValues,
    const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityOutValues,
    std::shared_ptr<AuxDensityMatrix<memorySpace>>
            auxDensityXCOutRepresentationPtr,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &xcEnergyDensity)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nCells        = basisOperationsPtr->nCells();
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    xcEnergyDensity.resize(nCells*nQuadsPerCell,0);

    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      xDensityOutDataOut;
    std::unordered_map<xcRemainderOutputDataAttributes, std::vector<double>>
      cDensityOutDataOut;

    std::vector<double> &xEnergyDensityOut =
      xDensityOutDataOut[xcRemainderOutputDataAttributes::e];
    std::vector<double> &cEnergyDensityOut =
      cDensityOutDataOut[xcRemainderOutputDataAttributes::e];


    auto quadPointsAll = basisOperationsPtr->quadPoints();

    auto quadWeightsAll = basisOperationsPtr->JxW();


    for (unsigned int iCell = 0; iCell < nCells; ++iCell)
      {
        std::vector<double> quadPointsInCell(nQuadsPerCell * 3);
        std::vector<double> quadWeightsInCell(nQuadsPerCell);
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          {
            for (unsigned int idim = 0; idim < 3; ++idim)
              quadPointsInCell[3 * iQuad + idim] =
                quadPointsAll[iCell * nQuadsPerCell * 3 + 3 * iQuad + idim];
            quadWeightsInCell[iQuad] =
              std::real(quadWeightsAll[iCell * nQuadsPerCell + iQuad]);
          }


        excManagerPtr->getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
          *auxDensityXCOutRepresentationPtr,
          quadPointsInCell,
          xDensityOutDataOut,
          cDensityOutDataOut);

        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
            xcEnergyDensity[iQuad]=xEnergyDensityOut[iQuad]+cEnergyDensityOut[iQuad];

      } // cell loop
  }

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

//compute electron-density
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
    //
    //compute kinetic energy density
    //
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          kineticEnergyDensityValues;
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
                                  d_uniformGridQuadratureId,
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
                                  d_uniformGridQuadratureId,
                                  d_kPointCoordinates,
                                  d_kPointWeights,
                                  kineticEnergyDensityValues,
                                  d_mpiCommParent,
                                  interpoolcomm,
                                  interBandGroupComm,
                                  mpi_communicator,
                                  *d_dftParamsPtr);


    //
    // compute xc energy density values
    //
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          xcEnergyDensityValues;
    std::map<dealii::CellId, std::vector<double>> rhoCoreValuesUniformQuad;
    std::map<dealii::CellId, std::vector<double>> gradRhoCoreValuesUniformQuad;

    if (d_dftParamsPtr->nonLinearCoreCorrection == true)
      initCoreRhoUniformQuad(d_uniformGridQuadratureId,
                             matrix_free_data,
                             rhoCoreValuesUniformQuad,
                             gradRhoCoreValuesUniformQuad);

    std::shared_ptr<AuxDensityMatrix<memorySpace>> auxDensityMatrixXCOutUniformQuadPtr;

    auxDensityMatrixXCOutUniformQuadPtr =std::make_shared<AuxDensityMatrixFE<memorySpace>>();

    updateAuxDensityXCMatrix(d_uniformGridQuadratureId,
                             densityUniformGridQuadValues,
                             gradDensityUniformGridQuadValues,
                             rhoCoreValuesUniformQuad,
                             gradRhoCoreValuesUniformQuad,
                             getEigenVectors(),
                             eigenValues,
                             fermiEnergy,
                             fermiEnergyUp,
                             fermiEnergyDown,
                             auxDensityMatrixXCOutUniformQuadPtr);

    computeXCEnergyDensityUniformQuad(d_basisOperationsPtrHost,
                                      d_uniformGridQuadratureId,
                                      d_excManagerPtr,
                                      densityUniformGridQuadValues,
                                      gradDensityUniformGridQuadValues,
                                      auxDensityMatrixXCOutUniformQuadPtr,
                                      xcEnergyDensityValues);



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

                  const double *cellkineticEdValues =
                    kineticEnergyDensityValues.data() + icell * n_q_points;

                  const double * xcEnergyDensityValuesCell =
                      xcEnergyDensityValues.data() + icell * n_q_points;

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

                      quadVals.push_back(cellkineticEdValues[q_point]*HaPerBohr3ToeVPerAng3);

                      quadVals.push_back(
                          xcEnergyDensityValuesCell[q_point]*HaPerBohr3ToeVPerAng3);

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
