// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

/*
 * @author Vishal Subramanian, Bikash Kanungo
 */

#include "InterpolateCellWiseDataToPoints.h"
#include "linearAlgebraOperationsInternal.h"
#include "linearAlgebraOperations.h"
#include "FECell.h"

#ifdef DFTFE_WITH_DEVICE
#  include "deviceDirectCCLWrapper.h"
#  include "elpaScalaManager.h"
#  include "dftParameters.h"
#  include <chebyshevOrthogonalizedSubspaceIterationSolverDevice.h>
#  include <dftUtils.h>
#  include <deviceKernelsGeneric.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <linearAlgebraOperations.h>
#  include <linearAlgebraOperationsDevice.h>
#  include <vectorUtilities.h>
#  include <deviceKernelsGeneric.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <vectorUtilities.h>
#endif

namespace dftfe
{
  namespace
  {
    template <typename T>
    void
    performCellWiseInterpolationToPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &
        BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<T,
                                                                                 dftfe::utils::MemorySpace::HOST> &inputVec,
                                         const unsigned int                    numberOfVectors,
                                         const unsigned int                    numCells,
                                         const unsigned int                    numDofsPerElement,
                                         const std::vector<size_type> &numPointsInCell,
                                         const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                                                           dftfe::utils::MemorySpace::HOST>&
                                           mapVecToCells,
                                         const  dftfe::utils::MemoryStorage<T,
                                                                           dftfe::utils::MemorySpace::HOST>& shapeFuncValues,
                                         const dftfe::utils::MemoryStorage<size_type, dftfe::utils::MemorySpace::HOST> &mapPointToCell,
                                         const dftfe::utils::MemoryStorage<global_size_type, dftfe::utils::MemorySpace::HOST> &mapPointToProcLocal,
                                         const  dftfe::utils::MemoryStorage<size_type, dftfe::utils::MemorySpace::HOST> &mapPointToShapeFuncIndex,
                                         const std::vector<unsigned int> &cellShapeFuncStartIndex,
                                         const std::vector<std::vector<unsigned int>> & mapCellLocalToProcLocal,
                                         dftfe::utils::MemoryStorage<T,
                                                                     dftfe::utils::MemorySpace::HOST> &cellLevelParentNodalMemSpace,
								     dftfe::utils::MemoryStorage<T,
                                                                     dftfe::utils::MemorySpace::HOST> &tempOutputdata,
                                         dftfe::utils::MemoryStorage<T,
                                                                     dftfe::utils::MemorySpace::HOST> &outputData)
    {
      const char         transA = 'N', transB = 'N';
      const double       scalarCoeffAlpha = 1.0;
      const double       scalarCoeffBeta  = 0.0;

      std::vector<double> cellLevelOutputPoints;
      const size_type inc = 1;
      
       cellLevelParentNodalMemSpace.resize(numberOfVectors*numDofsPerElement);

      for( size_type iElemSrc = 0 ; iElemSrc < numCells; iElemSrc++)
        {
          unsigned int numberOfPointsInSrcCell =
            numPointsInCell[iElemSrc];
          cellLevelOutputPoints.resize(numberOfPointsInSrcCell*numberOfVectors);

          for (unsigned int iNode = 0; iNode < numDofsPerElement;
               iNode++)
            {
              dcopy_(&numberOfVectors,
                     inputVec.data() +
                       mapVecToCells
                         [iElemSrc * numDofsPerElement + iNode],
                     &inc,
                     &cellLevelParentNodalMemSpace[numberOfVectors * iNode],
                     &inc);
            }

          xgemm(&transA,
                &transB,
                &numberOfVectors,
                &numberOfPointsInSrcCell,
                &numDofsPerElement,
                &scalarCoeffAlpha,
                &cellLevelParentNodalMemSpace[0],
                &numberOfVectors,
                &shapeFuncValues[cellShapeFuncStartIndex[iElemSrc]],
                &numDofsPerElement,
                &scalarCoeffBeta,
                &cellLevelOutputPoints[0],
                &numberOfVectors);

          for (unsigned int iPoint = 0; iPoint < numberOfPointsInSrcCell;
               iPoint++)
            {
              dcopy_(&numberOfVectors,
                     &cellLevelOutputPoints[iPoint*numberOfVectors],
                     &inc,
                     &outputData[mapCellLocalToProcLocal[iElemSrc][iPoint]*numberOfVectors],
                     &inc);
            }
        }
    }

#ifdef DFTFE_WITH_DEVICE
    template <typename T>
    void
    performCellWiseInterpolationToPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>> &
        BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<T,
                                                                                 dftfe::utils::MemorySpace::DEVICE> &inputVec,
                                         const unsigned int                    numberOfVectors,
                                         const unsigned int                    numCells,
                                         const unsigned int                    numDofsPerElement,
                                         const std::vector<size_type> &numPointsInCell,
                                         const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                                                           dftfe::utils::MemorySpace::DEVICE>&
                                           mapVecToCells,
                                         const  dftfe::utils::MemoryStorage<T,
                                                                           dftfe::utils::MemorySpace::DEVICE>& shapeFuncValues,
                                         const dftfe::utils::MemoryStorage<size_type, dftfe::utils::MemorySpace::DEVICE> &mapPointToCell,
                                         const dftfe::utils::MemoryStorage<global_size_type, dftfe::utils::MemorySpace::DEVICE> &mapPointToProcLocal,
                                         const  dftfe::utils::MemoryStorage<size_type, dftfe::utils::MemorySpace::DEVICE> &mapPointToShapeFuncIndex,
                                         const std::vector<unsigned int> &cellShapeFuncStartIndex,
                                         const std::vector<std::vector<unsigned int>> & mapCellLocalToProcLocal,
                                         dftfe::utils::MemoryStorage<T,
                                                                     dftfe::utils::MemorySpace::DEVICE> &cellLevelParentNodalMemSpace,
								     dftfe::utils::MemoryStorage<T,
                                                                     dftfe::utils::MemorySpace::DEVICE> &tempOutputdata,
                                         dftfe::utils::MemoryStorage<T,
                                                                     dftfe::utils::MemorySpace::DEVICE> &outputData)
    {

      const char         transA = 'N', transB = 'N';
      const double       scalarCoeffAlpha = 1.0;
      const double       scalarCoeffBeta  = 0.0;
      size_type pointsFoundInProc =  std::accumulate(numPointsInCell.begin(), numPointsInCell.end(),0.0);
      //cellLevelParentNodalMemSpace.resize(numCells*numberOfVectors*numDofsPerElement);
      
      
      dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
        numberOfVectors,
        numCells * numDofsPerElement,
        inputVec.data(),
        cellLevelParentNodalMemSpace.begin(),
        mapVecToCells.begin());

	      /*
      dftfe::utils::deviceKernelsGeneric::stridedCopyToBlockTranspose(
        numberOfVectors,
	numDofsPerElement,
        numCells * numDofsPerElement,
        inputVec.data(),
        cellLevelParentNodalMemSpace.begin(),
        mapVecToCells.begin());
*/
      dftfe::size_type pointStartIndex = 0;
      for(dftfe::size_type iCell = 0 ; iCell < numCells; iCell++)
        {
          BLASWrapperPtr->
          xgemm(transA,
                transB,
                  numberOfVectors,
                  numPointsInCell[iCell],
                  numDofsPerElement,
                  &scalarCoeffAlpha,
                  cellLevelParentNodalMemSpace.data() + numberOfVectors*numDofsPerElement*iCell,
                  numberOfVectors,
                  shapeFuncValues.data() + pointStartIndex*numDofsPerElement,
                  numDofsPerElement,
                &scalarCoeffBeta,
                  tempOutputdata.data() + pointStartIndex*numberOfVectors,
                  numberOfVectors);

          pointStartIndex += numPointsInCell[iCell];

      }
/*
      dftfe::utils::deviceKernelsGeneric::interpolateNodalDataToQuadDevice(
        numDofsPerElement,
        pointsFoundInProc,
        numberOfVectors,
        shapeFuncValues.data(),
        mapPointToCell.data(),
        mapPointToProcLocal.data(),
        mapPointToShapeFuncIndex.data(),
        cellLevelParentNodalMemSpace.data(),
        outputData.data());
*/

      dftfe::utils::deviceKernelsGeneric::stridedCopyFromBlock(
        numberOfVectors,
        pointsFoundInProc,
        tempOutputdata.data(),
        outputData.data(),
        mapPointToProcLocal.data());
    }
#endif

  }

  template <dftfe::utils::MemorySpace memorySpace>
  InterpolateCellWiseDataToPoints<memorySpace>::InterpolateCellWiseDataToPoints(const dealii::DoFHandler<3> &doFHandlerSrc,
                                                                             const std::vector<std::vector<double>> & targetPts,
                                                                             const MPI_Comm & mpiComm)
    :d_mapPoints(mpiComm),
    d_mpiComm(mpiComm)
  {

    MPI_Barrier(d_mpiComm);
    double startComp = MPI_Wtime();
    d_numLocalPtsSze = targetPts.size();
    size_type numInitialTargetPoints = targetPts.size();


    std::vector<std::vector<double>> paramPoints;

    typename dealii::DoFHandler<3>::active_cell_iterator
      cellSrcStart = doFHandlerSrc.begin_active(),
      cellSrcEnd = doFHandlerSrc.end();

    std::vector<std::shared_ptr<const dftfe::utils::Cell<3>>> srcCells(0);
    for (; cellSrcStart != cellSrcEnd; cellSrcStart++)
      {
        if (cellSrcStart->is_locally_owned())
          {
            auto srcCellPtr = std::make_shared<dftfe::utils::FECell<3>>(cellSrcStart);
            srcCells.push_back(srcCellPtr);
          }

      }
    MPI_Barrier(d_mpiComm);
    double startMapPoints = MPI_Wtime();


    size_type maxNumCells = srcCells.size();
    size_type maxNumPoints = numInitialTargetPoints;


    std::cout<<" NumPoints in proc = "<<numInitialTargetPoints<<" numCells per proc = "<<srcCells.size()<<"\n";
	    MPI_Allreduce(MPI_IN_PLACE,
                  &maxNumCells,
                  1,
                  dftfe::dataTypes::mpi_type_id(&maxNumCells),
                  MPI_MAX,
                  d_mpiComm);

	    MPI_Allreduce(MPI_IN_PLACE,
                  &maxNumPoints,
                  1,
                  dftfe::dataTypes::mpi_type_id(&maxNumPoints),
                  MPI_MAX,
                  d_mpiComm);

	    std::cout<<" maxNumPoints = "<<maxNumPoints<<" maxNumCells = "<<maxNumCells<<"\n";
    // create the RTree and the
    d_mapPoints.init(srcCells,
                     targetPts,
                     paramPoints,
                     d_mapCellLocalToProcLocal,
                     d_localRange,
                     d_ghostGlobalIds,
                     1e-10); // TODO this is hardcoded

    MPI_Barrier(d_mpiComm);
    double endMapPoints = MPI_Wtime();

    d_numCells = srcCells.size();

    d_cellPointStartIndex.resize(d_numCells);
    d_cellShapeFuncStartIndex.resize(d_numCells);


    d_numPointsLocal = 0;

    d_numDofsPerElement = doFHandlerSrc.get_fe().dofs_per_cell;

    d_numPointsInCell.resize(d_numCells);

    const dealii::FiniteElement<3> &feSrc  = doFHandlerSrc.get_fe();

    d_pointsFoundInProc = 0;
    for(size_type iCell = 0 ;iCell < d_numCells; iCell++)
      {
        d_numPointsInCell[iCell] = paramPoints[iCell].size()/3;
        d_pointsFoundInProc += d_numPointsInCell[iCell];
        //        std::cout<<" num points in cell = "<<d_numPointsInCell[iCell]<<'\n';
        if(iCell > 0 )
          {
            d_cellPointStartIndex[iCell] = d_cellPointStartIndex[iCell-1] + d_numPointsInCell[iCell-1];
            d_cellShapeFuncStartIndex[iCell] = d_cellShapeFuncStartIndex[iCell-1] +
                                               d_numDofsPerElement*(d_numPointsInCell[iCell-1]);
          }
        else
          {
            d_cellPointStartIndex[0] = 0;
            d_cellShapeFuncStartIndex[0]  = 0;
          }

      }

    MPI_Barrier(d_mpiComm);
    double startShapeFunc = MPI_Wtime();

    //    std::cout<<" ghost size = "<<d_ghostGlobalIds.size()<<"\n";

    d_numPointsLocal = targetPts.size() + d_ghostGlobalIds.size();
    d_shapeFuncValues.resize(d_pointsFoundInProc*d_numDofsPerElement);
    std::fill(d_shapeFuncValues.begin(),d_shapeFuncValues.end(),0.0);


    size_type numFinalTargetPoints = targetPts.size();

    //      std::cout<<" Target size initial = "<<numInitialTargetPoints<<" final = "<<numFinalTargetPoints<<"\n";
    //    size_type quadPointSizeDebug = targetPts.size()/d_numCells;

    //    std::cout<<"Num  quad points in cell = "<<quadPointSizeDebug<<"\n";

    size_type shapeFuncSizeDebug = d_shapeFuncValues.size();
    //      std::cout<<" size of shape func vec = "<<shapeFuncSizeDebug<<"\n";
    for(size_type iCell = 0 ;iCell < d_numCells; iCell++)
      {
        //          if( (d_cellShapeFuncStartIndex[iCell] - iCell*quadPointSizeDebug*d_numDofsPerElement) > 0 )
        //          {
        //              std::cout<<" error in cell start \n";
        //          }
        for( size_type iPoint = 0 ;iPoint < d_numPointsInCell[iCell]; iPoint++)
          {
            dealii::Point<3, double> pointParamCoord(paramPoints[iCell][3*iPoint+0],
                                                     paramPoints[iCell][3*iPoint+1],
                                                     paramPoints[iCell][3*iPoint+2]);

            if ((paramPoints[iCell][3*iPoint+0] < -1e-7 ) || (paramPoints[iCell][3*iPoint+0] > 1 + 1e-7 ))
              {
                std::cout<<" param point x coord is -ve\n";
              }
            if ((paramPoints[iCell][3*iPoint+1] < -1e-7 ) || (paramPoints[iCell][3*iPoint+1] > 1 + 1e-7 ))
              {
                std::cout<<" param point y coord is -ve\n";
              }
            if ((paramPoints[iCell][3*iPoint+2] < -1e-7 ) || (paramPoints[iCell][3*iPoint+2] > 1 + 1e-7 ))
              {
                std::cout<<" param point z coord is -ve\n";
              }
            for (unsigned int iNode = 0; iNode < d_numDofsPerElement;
                 iNode++)
              {

                //                if(d_cellShapeFuncStartIndex[iCell] +
                //                   iNode +
                //                   iPoint * d_numDofsPerElement > shapeFuncSizeDebug-1)
                //                {
                //                    std::cout<<" error in point id \n";
                //                }
                d_shapeFuncValues[d_cellShapeFuncStartIndex[iCell] +
                                  iNode +
                                  iPoint * d_numDofsPerElement] = feSrc.shape_value(iNode, pointParamCoord);
              }
          }
      }

    //    std::cout<<" local range start = "<<d_localRange.first<<" final = "<<d_localRange.second<<" ghost size = "<<d_ghostGlobalIds.size();
    //
    //    std::cout<<" size of shape func vec = "<<d_shapeFuncValues.size()<<"\n";

    global_size_type numTargetPointsInput = (global_size_type)targetPts.size();
    MPI_Allreduce(MPI_IN_PLACE,
                  &numTargetPointsInput,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numTargetPointsInput),
                  MPI_SUM,
                  d_mpiComm);

    global_size_type numTargetPointsFound = (global_size_type)d_shapeFuncValues.size()/d_numDofsPerElement;
    MPI_Allreduce(MPI_IN_PLACE,
                  &numTargetPointsFound,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numTargetPointsFound),
                  MPI_SUM,
                  d_mpiComm);


    global_size_type numLocalPlusGhost = (global_size_type)d_numPointsLocal;
    MPI_Allreduce(MPI_IN_PLACE,
                  &numLocalPlusGhost,
                  1,
                  dftfe::dataTypes::mpi_type_id(&numLocalPlusGhost),
                  MPI_SUM,
                  d_mpiComm);


    //std::cout<<" Number of points in  rank = "<<dealii::Utilities::MPI::this_mpi_process(d_mpiComm)<<" "<<d_shapeFuncValues.size()/d_numDofsPerElement<<"\n";

    std::cout<<std::flush;
    MPI_Barrier(d_mpiComm);
    //    std::cout<<" Total num of points in rank = "<<dealii::Utilities::MPI::this_mpi_process(d_mpiComm)<<" "<<d_numPointsLocal<<"\n";
    //      std::cout<<std::flush;
    //      MPI_Barrier(d_mpiComm);

    MPI_Barrier(d_mpiComm);
    double endShapeFunc = MPI_Wtime();
    d_mpiPatternP2PPtr = std::make_shared<dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>(d_localRange, d_ghostGlobalIds, d_mpiComm);


    d_mpiP2PPtrMemSpace = std::make_shared<dftfe::utils::mpi::MPIPatternP2P<memorySpace>>(d_localRange, d_ghostGlobalIds, d_mpiComm);
    std::vector<global_size_type> cellLocalToProcLocal;
    std::vector<size_type> shapeFuncIndex, pointToCellIndex;
    cellLocalToProcLocal.resize(d_pointsFoundInProc);
    shapeFuncIndex.resize(d_pointsFoundInProc);
    pointToCellIndex.resize(d_pointsFoundInProc);

    size_type pointIndex = 0;
    for ( size_type iCell = 0 ; iCell < d_numCells; iCell++)
      {
        for ( size_type iPoint = 0 ;iPoint < d_numPointsInCell[iCell]; iPoint++)
          {
            cellLocalToProcLocal[pointIndex] = d_mapCellLocalToProcLocal[iCell][iPoint];
            shapeFuncIndex[pointIndex] = d_cellShapeFuncStartIndex[iCell] + iPoint * d_numDofsPerElement;
            pointToCellIndex[pointIndex] = iCell;
            pointIndex++;
          }
      }

    d_shapeValuesMemSpace.resize(d_shapeFuncValues.size());
    d_shapeValuesMemSpace.copyFrom(d_shapeFuncValues);

    d_mapPointToCellIndexMemSpace.resize(pointToCellIndex.size());
    d_mapPointToCellIndexMemSpace.copyFrom(pointToCellIndex);

    d_mapPointToProcLocalMemSpace.resize(cellLocalToProcLocal.size());
    d_mapPointToProcLocalMemSpace.copyFrom(cellLocalToProcLocal);

    d_mapPointToShapeFuncIndexMemSpace.resize(shapeFuncIndex.size());
    d_mapPointToShapeFuncIndexMemSpace.copyFrom(shapeFuncIndex);



    MPI_Barrier(d_mpiComm);
    double endMPIPattern = MPI_Wtime();

    double nonLocalFrac = ((double)((double)(numLocalPlusGhost - numTargetPointsInput))/numTargetPointsInput);
    if( dealii::Utilities::MPI::this_mpi_process(d_mpiComm) == 0 )
      {
        std::cout<<" Total number of points provided as input = "<<numTargetPointsInput<<"\n";
        std::cout<<" Total number of points found from input = "<<numTargetPointsFound<<"\n";
        std::cout<<" Total number of points in all procs = "<<numLocalPlusGhost<<" fraction of non local pts = "<< nonLocalFrac<<"\n";

        dftfe::utils::throwException(numTargetPointsFound  >= numTargetPointsInput, " Number of points found is less than the input points \n");

        std::cout<<" Time for start Comp = "<<startMapPoints-startComp<<"\n";
        std::cout<<" Time for map Points init = "<<endMapPoints - startMapPoints<<"\n";
        std::cout<<" Time for shape func array = "<<startShapeFunc - endMapPoints<<"\n";
        std::cout<<" Time for computing shape func = "<<endShapeFunc -startShapeFunc<<"\n";
        std::cout<<" time for MPI pattern creation = "<<endMPIPattern - endShapeFunc<<"\n";
      }

  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void InterpolateCellWiseDataToPoints<memorySpace>::
    interpolateSrcDataToTargetPoints(
      const distributedCPUVec<T> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                        dftfe::utils::MemorySpace::HOST>&                  mapVecToCells,
      dftfe::utils::MemoryStorage<T,
                                  dftfe::utils::MemorySpace::HOST> &outputData, // this is not std::vector
      bool resizeData )
  {
    if(resizeData)
      {
        d_mpiCommP2PPtr = std::make_shared<dftfe::utils::mpi::MPICommunicatorP2P<dataTypes::number,dftfe::utils::MemorySpace::HOST>>(d_mpiPatternP2PPtr,numberOfVectors);
	d_mpiCommP2PPtr->setCommunicationPrecision(dftfe::utils::mpi::communicationPrecision::full);
        outputData.resize(d_numPointsLocal*numberOfVectors);
      }

    std::fill(outputData.begin(),outputData.end(),0.0);
    const double       scalarCoeffAlpha = 1.0;
    const double       scalarCoeffBeta  = 0.0;
    const char         transA = 'N', transB = 'N';
    const unsigned int inc = 1;


    unsigned int iElemSrc = 0;

    std::vector<double> cellLevelOutputPoints;
    std::vector<double> cellLevelInputVec(d_numDofsPerElement *
                                            numberOfVectors,
                                          0.0);
    for( size_type iElemSrc = 0 ; iElemSrc < d_numCells; iElemSrc++)
      {
        unsigned int numberOfPointsInSrcCell =
          d_numPointsInCell[iElemSrc];
        cellLevelOutputPoints.resize(numberOfPointsInSrcCell*numberOfVectors);

        for (unsigned int iNode = 0; iNode < d_numDofsPerElement;
             iNode++)
          {
            dcopy_(&numberOfVectors,
                   inputVec.begin() +
                     mapVecToCells
                       [iElemSrc * d_numDofsPerElement + iNode],
                   &inc,
                   &cellLevelInputVec[numberOfVectors * iNode],
                   &inc);
          }

        xgemm(&transA,
              &transB,
              &numberOfVectors,
              &numberOfPointsInSrcCell,
              &d_numDofsPerElement,
              &scalarCoeffAlpha,
              &cellLevelInputVec[0],
              &numberOfVectors,
              &d_shapeFuncValues[d_cellShapeFuncStartIndex[iElemSrc]],
              &d_numDofsPerElement,
              &scalarCoeffBeta,
              &cellLevelOutputPoints[0],
              &numberOfVectors);

        for (unsigned int iPoint = 0; iPoint < numberOfPointsInSrcCell;
             iPoint++)
          {
            dcopy_(&numberOfVectors,
                   &cellLevelOutputPoints[iPoint*numberOfVectors],
                   &inc,
                   &outputData[d_mapCellLocalToProcLocal[iElemSrc][iPoint]*numberOfVectors],
                   &inc);
          }
      }

    d_mpiCommP2PPtr->accumulateInsertLocallyOwned(outputData);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void InterpolateCellWiseDataToPoints<memorySpace>::
    interpolateSrcDataToTargetPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>> &
        BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<T,
                                                    memorySpace> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dealii::types::global_dof_index, memorySpace>
                                                                   &                  mapVecToCells,
      dftfe::utils::MemoryStorage<T,
                                  memorySpace> &outputData, // this is not std::vector
      bool resizeData )
  {
	  if( dealii::Utilities::MPI::this_mpi_process(d_mpiComm) == 0 )
      {
          std::cout<<" resizeData inside InterpolateCellWiseDataToPoints = "<<resizeData<<"\n";
      }

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
	  MPI_Barrier(d_mpiComm);
	  double startTime = MPI_Wtime();
    if(resizeData)
      {
        d_mpiCommPtrMemSpace = std::make_unique<dftfe::utils::mpi::MPICommunicatorP2P<T,memorySpace>>(d_mpiP2PPtrMemSpace,numberOfVectors);
	d_mpiCommPtrMemSpace->setCommunicationPrecision(dftfe::utils::mpi::communicationPrecision::full);
        outputData.resize(d_numPointsLocal*numberOfVectors);
        d_cellLevelParentNodalMemSpace.resize(d_numCells*d_numDofsPerElement*numberOfVectors);
   
       std::vector<global_size_type> cellLocalToProcLocal;
    cellLocalToProcLocal.resize(d_pointsFoundInProc);
	    size_type pointIndex = 0;
    for ( size_type iCell = 0 ; iCell < d_numCells; iCell++)
      {
        for ( size_type iPoint = 0 ;iPoint < d_numPointsInCell[iCell]; iPoint++)
          {
            cellLocalToProcLocal[pointIndex] = d_mapCellLocalToProcLocal[iCell][iPoint]*numberOfVectors;
            pointIndex++;
          }
      }


    d_mapPointToProcLocalMemSpace.resize(cellLocalToProcLocal.size());
    d_mapPointToProcLocalMemSpace.copyFrom(cellLocalToProcLocal);

   
   d_tempOutputMemSpace.resize(d_pointsFoundInProc*numberOfVectors);
      }

    #if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
          MPI_Barrier(d_mpiComm);
          double endResizeTime = MPI_Wtime();
    outputData.setValue(0.0);


    performCellWiseInterpolationToPoints<T>(
      BLASWrapperPtr,
      inputVec,
                                         numberOfVectors,
                                         d_numCells,
                                         d_numDofsPerElement,
                                         d_numPointsInCell,
                                         mapVecToCells,
                                         d_shapeValuesMemSpace,
                                         d_mapPointToCellIndexMemSpace,
                                         d_mapPointToProcLocalMemSpace,
                                         d_mapPointToShapeFuncIndexMemSpace,
                                         d_cellShapeFuncStartIndex,
                                         d_mapCellLocalToProcLocal,
                                         d_cellLevelParentNodalMemSpace,
                                         d_tempOutputMemSpace,
					 outputData);

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
          MPI_Barrier(d_mpiComm);
          double endCompTime = MPI_Wtime();
// TODO uncomment the following line after testing
    d_mpiCommPtrMemSpace->accumulateInsertLocallyOwned(outputData);
  
    #if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
          MPI_Barrier(d_mpiComm);
          double endCommTime = MPI_Wtime();

if( dealii::Utilities::MPI::this_mpi_process(d_mpiComm) == 0 )
      {
	  std::cout<<" resize Time = "<<endResizeTime-startTime<<" Comp time = "<<endCompTime-endResizeTime<<" comm time = "<<endCommTime-endCompTime<<"\n";
      }
  }

  template
  void InterpolateCellWiseDataToPoints<dftfe::utils::MemorySpace::HOST>::interpolateSrcDataToTargetPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &
        BLASWrapperPtr,
    const dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                              dftfe::utils::MemorySpace::HOST> &inputVec,
    const unsigned int                    numberOfVectors,
    const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                        dftfe::utils::MemorySpace::HOST> &mapVecToCells,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST> &outputData,
    bool resizeData) ;

  template
  void InterpolateCellWiseDataToPoints<dftfe::utils::MemorySpace::HOST>::interpolateSrcDataToTargetPoints(
    const distributedCPUVec<dataTypes::number> &inputVec,
    const unsigned int                    numberOfVectors,
    const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                      dftfe::utils::MemorySpace::HOST>&                  mapVecToCells,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &outputData,
    bool resizeData) ;

  template class InterpolateCellWiseDataToPoints<dftfe::utils::MemorySpace::HOST>;


#ifdef DFTFE_WITH_DEVICE

    template
  void InterpolateCellWiseDataToPoints<dftfe::utils::MemorySpace::DEVICE>::interpolateSrcDataToTargetPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>> &
        BLASWrapperPtr,
    const dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                              dftfe::utils::MemorySpace::DEVICE> &inputVec,
    const unsigned int                    numberOfVectors,
    const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                        dftfe::utils::MemorySpace::DEVICE> &mapVecToCells,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE> &outputData,
    bool resizeData) ;

  template
  void InterpolateCellWiseDataToPoints<dftfe::utils::MemorySpace::DEVICE>::interpolateSrcDataToTargetPoints(
    const distributedCPUVec<dataTypes::number> &inputVec,
    const unsigned int                    numberOfVectors,
    const dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                                      dftfe::utils::MemorySpace::HOST>&                  mapVecToCells,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &outputData,
    bool resizeData) ;

  template class InterpolateCellWiseDataToPoints<dftfe::utils::MemorySpace::DEVICE>;
#endif
}

//    double errorGhostPoints = 0.0;
//      std::vector<double> valuesAtGhostPoints ;
//      size_type ghostNumPoints = d_ghostGlobalIds.size();
//      valuesAtGhostPoints.resize(ghostNumPoints,0.0);
//
//      for(size_type iPoint = 0; iPoint < ghostNumPoints; iPoint++)
//      {
//          valuesAtGhostPoints[iPoint] = value(d_ghostIdRealCoords[iPoint][0],d_ghostIdRealCoords[iPoint][1],d_ghostIdRealCoords[iPoint][2],1);
//
//          errorGhostPoints += (valuesAtGhostPoints[iPoint] - outputData[(iPoint + d_numLocalPtsSze)*numberOfVectors + 0 ])*
//                  (valuesAtGhostPoints[iPoint] - outputData[(iPoint + d_numLocalPtsSze)*numberOfVectors + 0 ]);
//      }
//
//      MPI_Allreduce(MPI_IN_PLACE,
//                    &errorGhostPoints,
//                    1,
//                    MPI_DOUBLE,
//                    MPI_SUM,
//                    d_mpiComm);
//
//      std::cout<<" Error in ghost pts = "<<errorGhostPoints<<"\n";
//
//
//      size_type thisRankId = dealii::Utilities::MPI::this_mpi_process(d_mpiComm);

//      for( size_type iPoint = 0; iPoint < d_numLocalPtsSze; iPoint++)
//      {
//          outputData[(iPoint*numberOfVectors) + 0 ] = d_localRange.first + iPoint;
//      }
//
//      for(size_type iPoint = 0; iPoint < ghostNumPoints; iPoint++)
//      {
//          outputData[(iPoint + d_numLocalPtsSze)*numberOfVectors + 0 ] = d_ghostGlobalIds[iPoint];
//          std::cout<<" rank = "<<thisRankId<<" local Id = "<<iPoint + d_numLocalPtsSze<<" ghost Id = "<<d_ghostGlobalIds[iPoint]<<" output = "<<outputData[(iPoint + d_numLocalPtsSze)*numberOfVectors + 0 ]<<"\n";
//      }
//
//      d_mpiCommP2PPtr->accumulateInsertLocallyOwned(outputData);
//
//
//      double errorLocalPts = 0.0;
//
//      double errorLocallyOwnedPts = 0.0;
//      double errorLocallyOwnedFromGhostPts = 0.0;
//
//      std::cout<<" rank = "<<thisRankId<<" num Local pts = "<<d_numLocalPtsSze<<" local start = "<<d_localRange.first<<"\n";
//      for( size_type iPoint = 0; iPoint < d_numLocalPtsSze; iPoint++)
//      {
//          errorLocalPts += (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint))*
//                  (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint));
//
//          if(d_foundLocally[iPoint])
//          {
//              errorLocallyOwnedPts += (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint))*
//                                      (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint));
//          }
//          else
//          {
//              errorLocallyOwnedFromGhostPts += (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint))*
//                                      (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint));
//          }
//      }
//
//
//      std::cout<<" rank = "<<thisRankId<<" errorLocallyOwnedPts = "<<errorLocallyOwnedPts<<"\n";
//      std::cout<<" rank = "<<thisRankId<<" errorLocallyOwnedFromGhostPts = "<<errorLocallyOwnedFromGhostPts<<"\n";
////      MPI_Allreduce(MPI_IN_PLACE,
////                    &errorLocalPts,
////                    1,
////                    MPI_DOUBLE,
////                    MPI_SUM,
////                    d_mpiComm);
//
//      std::cout<<" rank = "<<thisRankId<<" Error in accumulate insert pts = "<<errorLocalPts<<"\n";
//
//
//      for( size_type iPoint = 0; iPoint < d_numLocalPtsSze; iPoint++)
//      {
//          outputData[(iPoint*numberOfVectors) + 0 ] = 0.0;
//      }
//
//      for(size_type iPoint = 0; iPoint < ghostNumPoints; iPoint++)
//      {
//          outputData[(iPoint + d_numLocalPtsSze)*numberOfVectors + 0 ] = d_ghostGlobalIds[iPoint];
//      }
//
//      d_mpiCommP2PPtr->accumulateAddLocallyOwned(outputData);
//
//
//      errorLocalPts = 0.0;
//
//      errorLocallyOwnedPts = 0.0;
//      errorLocallyOwnedFromGhostPts = 0.0;
//
//      for( size_type iPoint = 0; iPoint < d_numLocalPtsSze; iPoint++)
//      {
//
//
//          if(d_foundLocally[iPoint])
//          {
//              errorLocalPts += (outputData[(iPoint*numberOfVectors) + 0 ] )*
//                               (outputData[(iPoint*numberOfVectors) + 0 ] );
//
//              errorLocallyOwnedPts += (outputData[(iPoint*numberOfVectors) + 0 ] )*
//                                      (outputData[(iPoint*numberOfVectors) + 0 ] );
//          }
//          else
//          {
//              errorLocalPts += (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint))*
//                               (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint));
//
//              if( std::abs(outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint)) > 0.5)
//              {
//                  std::cout<<" rank = "<<thisRankId<<" Id "<<iPoint<<" outDat "<<outputData[(iPoint*numberOfVectors) + 0 ]<<" correct "<<(d_localRange.first + iPoint)<<"\n";
//
//              }
//              errorLocallyOwnedFromGhostPts += (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint))*
//                                               (outputData[(iPoint*numberOfVectors) + 0 ] - (d_localRange.first + iPoint));
//          }
//      }
//
//
//      std::cout<<" rank = "<<thisRankId<<" errorLocallyOwnedPts = "<<errorLocallyOwnedPts<<"\n";
//      std::cout<<" rank = "<<thisRankId<<" errorLocallyOwnedFromGhostPts = "<<errorLocallyOwnedFromGhostPts<<"\n";
////      MPI_Allreduce(MPI_IN_PLACE,
////                    &errorLocalPts,
////                    1,
////                    MPI_DOUBLE,
////                    MPI_SUM,
////                    d_mpiComm);
//
//      std::cout<<" rank = "<<thisRankId<<" Error in accumulate insert pts = "<<errorLocalPts<<"\n";

