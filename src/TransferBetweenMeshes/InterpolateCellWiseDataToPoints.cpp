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
//  namespace
//  {
//    template <typename T>
//    void
//    performCellWiseInterpolationToPoints(
//      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &
//        BLASWrapperPtr,
//      const dftfe::linearAlgebra::MultiVector<T,
//                                                                                 dftfe::utils::MemorySpace::HOST> &inputVec,
//                                         const unsigned int                    numberOfVectors,
//                                         const unsigned int                    numCells,
//      const unsigned int totalDofsInCells,
//                                         const std::vector<unsigned int>                   &numDofsPerElement,
//      const std::vector<unsigned int>                   &cumulativeDofsPerElement,
//                                         const std::vector<size_type> &numPointsInCell,
//                                         const dftfe::utils::MemoryStorage<dftfe::global_size_type,
//                                                                           dftfe::utils::MemorySpace::HOST>&
//                                           mapVecToCells,
//                                         const  dftfe::utils::MemoryStorage<T,
//                                                                           dftfe::utils::MemorySpace::HOST>& shapeFuncValues,
//                                         const dftfe::utils::MemoryStorage<size_type, dftfe::utils::MemorySpace::HOST> &mapPointToCell,
//                                         const dftfe::utils::MemoryStorage<global_size_type, dftfe::utils::MemorySpace::HOST> &mapPointToProcLocal,
//                                         const  dftfe::utils::MemoryStorage<size_type, dftfe::utils::MemorySpace::HOST> &mapPointToShapeFuncIndex,
//                                         const std::vector<unsigned int> &cellShapeFuncStartIndex,
//                                         const std::vector<std::vector<unsigned int>> & mapCellLocalToProcLocal,
//                                         dftfe::utils::MemoryStorage<T,
//                                                                     dftfe::utils::MemorySpace::HOST> &cellLevelParentNodalMemSpace,
//								     dftfe::utils::MemoryStorage<T,
//                                                                     dftfe::utils::MemorySpace::HOST> &tempOutputdata,
//                                         dftfe::utils::MemoryStorage<T,
//                                                                     dftfe::utils::MemorySpace::HOST> &outputData)
//    {
//      const char         transA = 'N', transB = 'N';
//      const T       scalarCoeffAlpha = 1.0;
//      const T       scalarCoeffBeta  = 0.0;
//
//      std::vector<T> cellLevelOutputPoints;
//      const size_type inc = 1;
//
//
//
//      for( size_type iElemSrc = 0 ; iElemSrc < numCells; iElemSrc++)
//        {
//          cellLevelParentNodalMemSpace.resize(numberOfVectors*numDofsPerElement[iElemSrc]);
//          unsigned int numberOfPointsInSrcCell =
//            numPointsInCell[iElemSrc];
//          cellLevelOutputPoints.resize(numberOfPointsInSrcCell*numberOfVectors);
//
//          for (unsigned int iNode = 0; iNode < numDofsPerElement[iElemSrc];
//               iNode++)
//            {
//              BLASWrapperPtr->xcopy(numberOfVectors,
//                     inputVec.data() +
//                       mapVecToCells
//                         [cumulativeDofsPerElement[iElemSrc] + iNode],
//                     inc,
//                     &cellLevelParentNodalMemSpace[numberOfVectors * iNode],
//                     inc);
//            }
//
//          BLASWrapperPtr->xgemm(transA,
//                transB,
//                numberOfVectors,
//                numberOfPointsInSrcCell,
//                numDofsPerElement[iElemSrc],
//                &scalarCoeffAlpha,
//                &cellLevelParentNodalMemSpace[0],
//                numberOfVectors,
//                &shapeFuncValues[cellShapeFuncStartIndex[iElemSrc]],
//                numDofsPerElement[iElemSrc],
//                &scalarCoeffBeta,
//                &cellLevelOutputPoints[0],
//                numberOfVectors);
//
//          for (unsigned int iPoint = 0; iPoint < numberOfPointsInSrcCell;
//               iPoint++)
//            {
//              BLASWrapperPtr->xcopy(numberOfVectors,
//                     &cellLevelOutputPoints[iPoint*numberOfVectors],
//                     inc,
//                     &outputData[mapCellLocalToProcLocal[iElemSrc][iPoint]*numberOfVectors],
//                     inc);
//            }
//        }
//    }
//
//#ifdef DFTFE_WITH_DEVICE
//    template <typename T>
//    void
//    performCellWiseInterpolationToPoints(
//      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>> &
//        BLASWrapperPtr,
//      const dftfe::linearAlgebra::MultiVector<T,
//                                                                                 dftfe::utils::MemorySpace::DEVICE> &inputVec,
//                                         const unsigned int                    numberOfVectors,
//                                         const unsigned int                    numCells,
//      const unsigned int totalDofsInCells,
//      const std::vector<unsigned int>                   &numDofsPerElement,
//      const std::vector<unsigned int>                   &cumulativeDofsPerElement,
//                                         const std::vector<size_type> &numPointsInCell,
//                                         const dftfe::utils::MemoryStorage<dftfe::global_size_type,
//                                                                           dftfe::utils::MemorySpace::DEVICE>&
//                                           mapVecToCells,
//                                         const  dftfe::utils::MemoryStorage<T,
//                                                                           dftfe::utils::MemorySpace::DEVICE>& shapeFuncValues,
//                                         const dftfe::utils::MemoryStorage<size_type, dftfe::utils::MemorySpace::DEVICE> &mapPointToCell,
//                                         const dftfe::utils::MemoryStorage<global_size_type, dftfe::utils::MemorySpace::DEVICE> &mapPointToProcLocal,
//                                         const  dftfe::utils::MemoryStorage<size_type, dftfe::utils::MemorySpace::DEVICE> &mapPointToShapeFuncIndex,
//                                         const std::vector<unsigned int> &cellShapeFuncStartIndex,
//                                         const std::vector<std::vector<unsigned int>> & mapCellLocalToProcLocal,
//                                         dftfe::utils::MemoryStorage<T,
//                                                                     dftfe::utils::MemorySpace::DEVICE> &cellLevelParentNodalMemSpace,
//								     dftfe::utils::MemoryStorage<T,
//                                                                     dftfe::utils::MemorySpace::DEVICE> &tempOutputdata,
//                                         dftfe::utils::MemoryStorage<T,
//                                                                     dftfe::utils::MemorySpace::DEVICE> &outputData)
//    {
//
//      const char         transA = 'N', transB = 'N';
//      const T       scalarCoeffAlpha = 1.0;
//      const T       scalarCoeffBeta  = 0.0;
//      size_type pointsFoundInProc =  std::accumulate(numPointsInCell.begin(), numPointsInCell.end(),0.0);
//      //cellLevelParentNodalMemSpace.resize(numCells*numberOfVectors*numDofsPerElement);
//
//
//      dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
//        numberOfVectors,
//        totalDofsInCells,
//        inputVec.data(),
//        cellLevelParentNodalMemSpace.begin(),
//        mapVecToCells.begin());
//
//	      /*
//      dftfe::utils::deviceKernelsGeneric::stridedCopyToBlockTranspose(
//        numberOfVectors,
//	numDofsPerElement,
//        numCells * numDofsPerElement,
//        inputVec.data(),
//        cellLevelParentNodalMemSpace.begin(),
//        mapVecToCells.begin());
//*/
//      dftfe::size_type pointStartIndex = 0;
//      for(dftfe::size_type iCell = 0 ; iCell < numCells; iCell++)
//        {
//          BLASWrapperPtr->
//          xgemm(transA,
//                transB,
//                  numberOfVectors,
//                  numPointsInCell[iCell],
//                  numDofsPerElement[iCell],
//                  &scalarCoeffAlpha,
//                  cellLevelParentNodalMemSpace.data() + numberOfVectors*cumulativeDofsPerElement[iCell],
//                  numberOfVectors,
//                  shapeFuncValues.data() + cellShapeFuncStartIndex[iCell],
//                  numDofsPerElement,
//                &scalarCoeffBeta,
//                  tempOutputdata.data() + pointStartIndex*numberOfVectors,
//                  numberOfVectors);
//
//          pointStartIndex += numPointsInCell[iCell];
//
//      }
///*
//      dftfe::utils::deviceKernelsGeneric::interpolateNodalDataToQuadDevice(
//        numDofsPerElement,
//        pointsFoundInProc,
//        numberOfVectors,
//        shapeFuncValues.data(),
//        mapPointToCell.data(),
//        mapPointToProcLocal.data(),
//        mapPointToShapeFuncIndex.data(),
//        cellLevelParentNodalMemSpace.data(),
//        outputData.data());
//*/
//
//      dftfe::utils::deviceKernelsGeneric::stridedCopyFromBlock(
//        numberOfVectors,
//        pointsFoundInProc,
//        tempOutputdata.data(),
//        outputData.data(),
//        mapPointToProcLocal.data());
//    }
//#endif
//
//  }

  template <typename T, dftfe::utils::MemorySpace memorySpace>
  InterpolateCellWiseDataToPoints<T,memorySpace>::InterpolateCellWiseDataToPoints(const std::vector<std::shared_ptr<const dftfe::utils::Cell<3>>> &srcCells,
                                                                                 std::vector<std::shared_ptr<InterpolateFromCellToLocalPoints<memorySpace>>> interpolateLocalObj,
                                                                             const std::vector<std::vector<double>> & targetPts,
                                                                             const std::vector<unsigned int> &numDofsPerElem,
                                                                             const MPI_Comm & mpiComm)
    :d_mapPoints(mpiComm),
    d_mpiComm(mpiComm)
  {

    MPI_Barrier(d_mpiComm);
    double startComp = MPI_Wtime();
    d_numLocalPtsSze = targetPts.size();
    size_type numInitialTargetPoints = targetPts.size();

    d_interpolateLocalObj = interpolateLocalObj;

    std::vector<std::vector<double>> coordinatesOfPointsInCell;

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
                       coordinatesOfPointsInCell,
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

    d_numDofsPerElement = numDofsPerElem; //doFHandlerSrc.get_fe().dofs_per_cell;

    d_cumulativeDofs.resize(d_numCells);
    d_numPointsInCell.resize(d_numCells);

//    const dealii::FiniteElement<3> &feSrc  = doFHandlerSrc.get_fe();

    d_pointsFoundInProc = 0;
    d_cumulativeDofs[0] = 0;
    global_size_type shapeFuncSize = 0;
    for(size_type iCell = 0 ;iCell < d_numCells; iCell++)
      {
        d_numPointsInCell[iCell] = coordinatesOfPointsInCell[iCell].size()/3;
        d_pointsFoundInProc += d_numPointsInCell[iCell];

        shapeFuncSize += (global_size_type)(d_numPointsInCell[iCell]*d_numDofsPerElement[iCell]);
        //        std::cout<<" num points in cell = "<<d_numPointsInCell[iCell]<<'\n';
        if(iCell > 0 )
          {
            d_cellPointStartIndex[iCell] = d_cellPointStartIndex[iCell-1] + d_numPointsInCell[iCell-1];
            d_cellShapeFuncStartIndex[iCell] = d_cellShapeFuncStartIndex[iCell-1] +
                                               d_numDofsPerElement[iCell-1]*(d_numPointsInCell[iCell-1]);
            d_cumulativeDofs[iCell] = d_cumulativeDofs[iCell -1] + d_numDofsPerElement[iCell-1];
          }
        else
          {
            d_cellPointStartIndex[0] = 0;
            d_cellShapeFuncStartIndex[0]  = 0;
            d_cumulativeDofs[0] = 0;
          }

      }
    totalDofsInCells = std::accumulate(d_numDofsPerElement.begin(), d_numDofsPerElement.end(),0.0);

    MPI_Barrier(d_mpiComm);
    double startShapeFunc = MPI_Wtime();

    //    std::cout<<" ghost size = "<<d_ghostGlobalIds.size()<<"\n";

    d_numPointsLocal = targetPts.size() + d_ghostGlobalIds.size();
//    d_shapeFuncValues.resize(shapeFuncSize);
//    std::fill(d_shapeFuncValues.begin(),d_shapeFuncValues.end(),0.0);


    size_type numFinalTargetPoints = targetPts.size();

    //      std::cout<<" Target size initial = "<<numInitialTargetPoints<<" final = "<<numFinalTargetPoints<<"\n";
    //    size_type quadPointSizeDebug = targetPts.size()/d_numCells;

    //    std::cout<<"Num  quad points in cell = "<<quadPointSizeDebug<<"\n";

//    size_type shapeFuncSizeDebug = d_shapeFuncValues.size();
    //      std::cout<<" size of shape func vec = "<<shapeFuncSizeDebug<<"\n";
    for(size_type iCell = 0 ;iCell < d_numCells; iCell++)
      {
        d_interpolateLocalObj[iCell]->getRealCoordinatesOfLocalPoints(d_numPointsInCell[iCell],
                                                                      coordinatesOfPointsInCell[iCell]);
        //          if( (d_cellShapeFuncStartIndex[iCell] - iCell*quadPointSizeDebug*d_numDofsPerElement) > 0 )
        //          {
        //              std::cout<<" error in cell start \n";
        //          }
//        for( size_type iPoint = 0 ;iPoint < d_numPointsInCell[iCell]; iPoint++)
//          {
            //
//            dealii::Point<3, double> pointParamCoord(paramPoints[iCell][3*iPoint+0],
//                                                     paramPoints[iCell][3*iPoint+1],
//                                                     paramPoints[iCell][3*iPoint+2]);
//
//            if ((paramPoints[iCell][3*iPoint+0] < -1e-7 ) || (paramPoints[iCell][3*iPoint+0] > 1 + 1e-7 ))
//              {
//                std::cout<<" param point x coord is -ve\n";
//              }
//            if ((paramPoints[iCell][3*iPoint+1] < -1e-7 ) || (paramPoints[iCell][3*iPoint+1] > 1 + 1e-7 ))
//              {
//                std::cout<<" param point y coord is -ve\n";
//              }
//            if ((paramPoints[iCell][3*iPoint+2] < -1e-7 ) || (paramPoints[iCell][3*iPoint+2] > 1 + 1e-7 ))
//              {
//                std::cout<<" param point z coord is -ve\n";
//              }
//            for (unsigned int iNode = 0; iNode < d_numDofsPerElement;
//                 iNode++)
//              {
//
//                //                if(d_cellShapeFuncStartIndex[iCell] +
//                //                   iNode +
//                //                   iPoint * d_numDofsPerElement > shapeFuncSizeDebug-1)
//                //                {
//                //                    std::cout<<" error in point id \n";
//                //                }
//                d_shapeFuncValues[d_cellShapeFuncStartIndex[iCell] +
//                                  iNode +
//                                  iPoint * d_numDofsPerElement] = feSrc.shape_value(iNode, pointParamCoord);
//              }
//          }
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

    global_size_type numTargetPointsFound = (global_size_type)d_pointsFoundInProc;
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
//    std::vector<size_type> shapeFuncIndex;
//    std::vector<size_type> pointToCellIndex;
    cellLocalToProcLocal.resize(d_pointsFoundInProc);
//    shapeFuncIndex.resize(d_pointsFoundInProc);
//    pointToCellIndex.resize(d_pointsFoundInProc);

    size_type pointIndex = 0;
    for ( size_type iCell = 0 ; iCell < d_numCells; iCell++)
      {
        for ( size_type iPoint = 0 ;iPoint < d_numPointsInCell[iCell]; iPoint++)
          {
            cellLocalToProcLocal[pointIndex] = d_mapCellLocalToProcLocal[iCell][iPoint];
//            shapeFuncIndex[pointIndex] = d_cellShapeFuncStartIndex[iCell] + iPoint * d_numDofsPerElement[iCell];
//            pointToCellIndex[pointIndex] = iCell;
            pointIndex++;
          }
      }

//    d_shapeValuesMemSpace.resize(d_shapeFuncValues.size());
//    d_shapeValuesMemSpace.copyFrom(d_shapeFuncValues);

//    d_mapPointToCellIndexMemSpace.resize(pointToCellIndex.size());
//    d_mapPointToCellIndexMemSpace.copyFrom(pointToCellIndex);

    d_mapPointToProcLocalMemSpace.resize(cellLocalToProcLocal.size());
    d_mapPointToProcLocalMemSpace.copyFrom(cellLocalToProcLocal);

//    d_mapPointToShapeFuncIndexMemSpace.resize(shapeFuncIndex.size());
//    d_mapPointToShapeFuncIndexMemSpace.copyFrom(shapeFuncIndex);



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

  template <typename T,dftfe::utils::MemorySpace memorySpace>
  void InterpolateCellWiseDataToPoints<T,memorySpace>::
    interpolateSrcDataToTargetPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &
        BLASWrapperPtr,
      const distributedCPUVec<T> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                        dftfe::utils::MemorySpace::HOST>&                  mapVecToCells,
      dftfe::utils::MemoryStorage<T,
                                  dftfe::utils::MemorySpace::HOST> &outputData, // this is not std::vector
      bool resizeData )
  {
    if(resizeData)
      {
        d_mpiCommP2PPtr = std::make_shared<dftfe::utils::mpi::MPICommunicatorP2P<T,dftfe::utils::MemorySpace::HOST>>(d_mpiPatternP2PPtr,numberOfVectors);
	d_mpiCommP2PPtr->setCommunicationPrecision(dftfe::utils::mpi::communicationPrecision::full);
        outputData.resize(d_numPointsLocal*numberOfVectors);
      }

    std::fill(outputData.begin(),outputData.end(),0.0);
    const T       scalarCoeffAlpha = 1.0;
    const T       scalarCoeffBeta  = 0.0;
    const char         transA = 'N', transB = 'N';
    const unsigned int inc = 1;


    unsigned int iElemSrc = 0;

    std::vector<T> cellLevelOutputPoints;

    for( size_type iElemSrc = 0 ; iElemSrc < d_numCells; iElemSrc++)
      {
        std::vector<T> cellLevelInputVec(d_numDofsPerElement[iElemSrc] *
                                           numberOfVectors,
                                         0.0);
        unsigned int numberOfPointsInSrcCell =
          d_numPointsInCell[iElemSrc];
        cellLevelOutputPoints.resize(numberOfPointsInSrcCell*numberOfVectors);

        for (unsigned int iNode = 0; iNode < d_numDofsPerElement[iElemSrc];
             iNode++)
          {
            BLASWrapperPtr->xcopy(numberOfVectors,
                   inputVec.begin() +
                     mapVecToCells
                       [d_cumulativeDofs[iElemSrc] + iNode],
                   inc,
                   &cellLevelInputVec[numberOfVectors * iNode],
                   inc);
          }

        d_interpolateLocalObj[iElemSrc]->interpolate(BLASWrapperPtr,
                                           numberOfVectors,
                                           cellLevelInputVec,
                                           cellLevelOutputPoints);

        for (unsigned int iPoint = 0; iPoint < numberOfPointsInSrcCell;
             iPoint++)
          {
            BLASWrapperPtr->xcopy(numberOfVectors,
                   &cellLevelOutputPoints[iPoint*numberOfVectors],
                   inc,
                   &outputData[d_mapCellLocalToProcLocal[iElemSrc][iPoint]*numberOfVectors],
                   inc);
          }
      }

    d_mpiCommP2PPtr->accumulateInsertLocallyOwned(outputData);
  }

  template <typename T,dftfe::utils::MemorySpace memorySpace>
  void InterpolateCellWiseDataToPoints<T,memorySpace>::
    interpolateSrcDataToTargetPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>> &
        BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<T,
                                                    memorySpace> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
                                                                   &                  mapVecToCells,
      dftfe::utils::MemoryStorage<T,
                                  memorySpace> &outputData, // this is not std::vector
      bool resizeData )
  {

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
        d_cellLevelParentNodalMemSpace.resize(totalDofsInCells*numberOfVectors);
   
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

    BLASWrapperPtr->stridedCopyToBlock(
      numberOfVectors,
      totalDofsInCells,
      inputVec.data(),
      d_cellLevelParentNodalMemSpace.begin(),
      mapVecToCells.data());

    dftfe::size_type pointStartIndex = 0;
    for(dftfe::size_type iCell = 0 ; iCell < d_numCells; iCell++)
      {
        d_interpolateLocalObj[iCell]->interpolate(BLASWrapperPtr,
                                           numberOfVectors,
                                           d_cellLevelParentNodalMemSpace.data() + numberOfVectors*d_cumulativeDofs[iCell],
                                           d_tempOutputMemSpace.data() + pointStartIndex*numberOfVectors);

//        BLASWrapperPtr->
//          xgemm(transA,
//                transB,
//                numberOfVectors,
//                d_numPointsInCell[iCell],
//                d_numDofsPerElement[iCell],
//                &scalarCoeffAlpha,
//                d_cellLevelParentNodalMemSpace.data() + numberOfVectors*d_cumulativeDofs[iCell],
//                numberOfVectors,
//                d_shapeValuesMemSpace.data() + d_cellShapeFuncStartIndex[iCell],
//                d_numDofsPerElement[iCell],
//                &scalarCoeffBeta,
//                d_tempOutputMemSpace.data() + pointStartIndex*numberOfVectors,
//                numberOfVectors);

        pointStartIndex += d_numPointsInCell[iCell];

      }

    BLASWrapperPtr->axpyStridedBlockAtomicAdd(
      numberOfVectors,
      d_pointsFoundInProc,
      d_tempOutputMemSpace.data(),
      outputData.begin(),
      d_mapPointToProcLocalMemSpace.begin());

//    performCellWiseInterpolationToPoints<T>(
//      BLASWrapperPtr,
//      inputVec,
//                                         numberOfVectors,
//                                         d_numCells,
//      totalDofsInCells,
//                                         d_numDofsPerElement,
//      d_cumulativeDofs,
//                                         d_numPointsInCell,
//                                         mapVecToCells,
//                                         d_shapeValuesMemSpace,
//                                         d_mapPointToCellIndexMemSpace,
//                                         d_mapPointToProcLocalMemSpace,
//                                         d_mapPointToShapeFuncIndexMemSpace,
//                                         d_cellShapeFuncStartIndex,
//                                         d_mapCellLocalToProcLocal,
//                                         d_cellLevelParentNodalMemSpace,
//                                         d_tempOutputMemSpace,
//					 outputData);

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
          MPI_Barrier(d_mpiComm);
          double endCompTime = MPI_Wtime();
    d_mpiCommPtrMemSpace->accumulateInsertLocallyOwned(outputData);
  
    #if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
          MPI_Barrier(d_mpiComm);
          double endCommTime = MPI_Wtime();

          int thisRankId;
          MPI_Comm_rank(d_mpiComm, &thisRankId);
if( thisRankId == 0 )
      {
	  std::cout<<" resize Time = "<<endResizeTime-startTime<<" Comp time = "<<endCompTime-endResizeTime<<" comm time = "<<endCommTime-endCompTime<<"\n";
      }
  }


  template class InterpolateCellWiseDataToPoints<dftfe::dataTypes::number, dftfe::utils::MemorySpace::HOST>;


#ifdef DFTFE_WITH_DEVICE

  template class InterpolateCellWiseDataToPoints<dftfe::dataTypes::number,dftfe::utils::MemorySpace::DEVICE>;
#endif
}

