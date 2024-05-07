
#include <algorithm>
namespace dftfe
{
  namespace utils
  {

    namespace
    {
      template <typename T>
      void appendToVec(std::vector<T> & dst,
                  const std::vector<T> & src)
      {
        dst.insert(dst.end(), src.begin(), src.end());
      }

      std::pair<global_size_type, global_size_type>
      getLocallyOwnedRange(const MPI_Comm & mpiComm,
                           const size_type myProcRank,
                           const size_type nProcs,
                           const size_type nLocalPoints)
      {
        std::vector<size_type> numPointsInProcs(nProcs,0);
        std::fill(numPointsInProcs.begin(), numPointsInProcs.end(), 0);
        numPointsInProcs[myProcRank] = nLocalPoints;
        MPI_Allreduce(MPI_IN_PLACE,
                      &numPointsInProcs[0],
                      nProcs,
                      dftfe::dataTypes::mpi_type_id(&numPointsInProcs[0]),
                      MPI_SUM,
                      mpiComm);

        global_size_type locallyOwnedStart = 0, locallyOwnedEnd = 0;

        for(unsigned int iProc = 0; iProc < myProcRank; iProc++)
          {
            locallyOwnedStart += (global_size_type)numPointsInProcs[iProc];
          }

        locallyOwnedEnd = locallyOwnedStart + numPointsInProcs[myProcRank];
        return (std::make_pair(locallyOwnedStart,locallyOwnedEnd));
      }

      template <size_type dim>
      void
      getProcBoundingBox(std::vector<std::shared_ptr<const Cell<dim>>> & cells,
                         std::vector<double> & lowerLeft,
                         std::vector<double> & upperRight)
      {
        lowerLeft.resize(dim);
        upperRight.resize(dim);
        const size_type nCells = cells.size();
        // First index is dimension and second index is cell Id
        // For each cell store both the lower left and upper right
        // limit in each dimension
        std::vector<std::vector<double>> cellsLowerLeft(dim,
                                                        std::vector<double>(nCells));
        std::vector<std::vector<double>> cellsUpperRight(dim,
                                                         std::vector<double>(nCells));
        for(size_type iCell = 0; iCell < nCells; ++iCell)
          {
            auto boundingBox = cells[iCell]->getBoundingBox();
            for(size_type iDim = 0; iDim < dim; ++iDim)
              {
                cellsLowerLeft[iDim][iCell] = boundingBox.first[iDim];
                cellsUpperRight[iDim][iCell] = boundingBox.second[iDim];
              }
          }

        // sort the cellLimits
        for(size_type iDim = 0; iDim < dim; ++iDim)
          {
            std::sort(cellsLowerLeft[iDim].begin(), cellsLowerLeft[iDim].end());
            std::sort(cellsUpperRight[iDim].begin(), cellsUpperRight[iDim].end());
            lowerLeft[iDim] = cellsLowerLeft[iDim][0];
            upperRight[iDim] = cellsUpperRight[iDim][nCells-1];
          }
      }


      void
      getAllProcsBoundingBoxes(const std::vector<double> & procLowerLeft,
                               const std::vector<double> & procUpperRight,
                               const size_type myProcRank,
                               const size_type nProcs,
                               const MPI_Comm & mpiComm,
                               std::vector<double> & allProcsBoundingBoxes)
      {
        const size_type dim = procLowerLeft.size();
        allProcsBoundingBoxes.resize(2*dim*nProcs);
        std::fill(allProcsBoundingBoxes.begin(), allProcsBoundingBoxes.end(), 0.0);

        for(unsigned int j = 0; j < dim; j++)
          {
            allProcsBoundingBoxes[2*dim*myProcRank + j] = procLowerLeft[j];
            allProcsBoundingBoxes[2*dim*myProcRank + dim + j] = procUpperRight[j];
          }

        MPI_Allreduce(MPI_IN_PLACE,
                      &allProcsBoundingBoxes[0],
                      2*dim*nProcs,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpiComm);
      }


      template <size_type dim, size_type M>
      void
      pointsToCell(std::vector<std::shared_ptr<const Cell<dim>>> & srcCells,
                   const std::vector<std::vector<double>> & targetPts,
                   std::vector<std::vector<size_type>> & cellFoundIds,
                   std::vector<std::vector<double>> & cellParamCoords,
                   std::vector<bool> & pointsFound,
                   const double paramCoordsTol)
      {
        RTreePoint<dim,M> rTreePoint(targetPts);
        const size_type numCells = srcCells.size();
        pointsFound.resize(targetPts.size());
        std::fill(pointsFound.begin(), pointsFound.end(), false);
        cellFoundIds.resize(numCells, std::vector<size_type>(0));
        cellParamCoords.resize(numCells, std::vector<double>(0));
        for(size_type iCell = 0; iCell < numCells; iCell++)
          {
            auto bbCell = srcCells[iCell]->getBoundingBox();
            auto targetPointList = rTreePoint.getPointIdsInsideBox(bbCell.first,
                                                                   bbCell.second);

            for(size_type iPoint = 0; iPoint < targetPointList.size(); iPoint++)
              {
                size_type pointIndex = targetPointList[iPoint];
                if(!pointsFound[pointIndex])
                  {
                    auto paramPoint = srcCells[iCell]->getParametricPoint(targetPts[pointIndex]);
                    bool pointInside = true;
                    for( unsigned int j = 0 ; j <dim; j++)
                      {
                        if((paramPoint[j] < -paramCoordsTol) || (paramPoint[j] > 1.0 + paramCoordsTol))
                          {
                            pointInside = false;
                          }
                      }
                    if (pointInside)
                      {
                        pointsFound[pointIndex] = true;
                        for(size_type iDim = 0; iDim < dim; iDim++)
                          {
                            cellParamCoords[iCell].push_back(paramPoint[iDim]);
                          }
                        cellFoundIds[iCell].push_back(pointIndex);
                      }
                  }
              }
          }

      }


      template<size_type dim, size_type M>
      void
      getTargetPointsToSend(const std::vector<std::shared_ptr<const Cell<dim>>> & srcCells,
                            const std::vector<size_type> & nonLocalPointLocalIds,
                            const std::vector<std::vector<double>> & nonLocalPointCoordinates,
                            const std::vector<double> & allProcsBoundingBoxes,
                            const global_size_type locallyOwnedStart,
                            const size_type myProcRank,
                            const size_type nProcs,
                            std::vector<size_type> &sendToProcIds,
                            std::vector<std::vector<global_size_type>> & sendToPointsGlobalIds,
                            std::vector<std::vector<double>> & sendToPointsCoords)
      {
        sendToProcIds.resize(0);
        sendToPointsGlobalIds.resize(0,
                                     std::vector<global_size_type>(0));
        sendToPointsCoords.resize(0,
                                  std::vector<double>(0));

        RTreePoint<dim,M> rTree(nonLocalPointCoordinates);
        for (size_type iProc = 0 ; iProc < nProcs; iProc++)
          {
            if(iProc != myProcRank)
              {
                std::vector<double> llProc(dim,0.0);
                std::vector<double> urProc(dim,0.0);
                for(size_type iDim = 0; iDim < dim; iDim++)
                  {
                    llProc[iDim] = allProcsBoundingBoxes[2*dim*iProc+iDim];
                    urProc[iDim] = allProcsBoundingBoxes[2*dim*iProc+dim+iDim];
                  }
                auto targetPointList = rTree.getPointIdsInsideBox(llProc,
                                                                  urProc);

                size_type numTargetPointsToSend = targetPointList.size();
                if(numTargetPointsToSend>0)
                  {
                    std::vector<global_size_type> globalIds(numTargetPointsToSend, -1);
                    sendToProcIds.push_back(iProc);
                    std::vector<double> pointCoordinates(0);
                    for(size_type iPoint = 0; iPoint < targetPointList.size(); iPoint++)
                      {
                        size_type pointIndex = targetPointList[iPoint];

                        appendToVec(pointCoordinates,
                                    nonLocalPointCoordinates[pointIndex]);
                        globalIds[iPoint] = locallyOwnedStart + nonLocalPointLocalIds[targetPointList[iPoint]];
                      }
                    // also have to send the coordinates and the indices.
                    sendToPointsGlobalIds.push_back(globalIds);
                    sendToPointsCoords.push_back(pointCoordinates);
                  }
              }
          }

      }

      template <size_type dim>
      void
      receivePoints(const std::vector<size_type> & sendToProcIds,
                    const std::vector<std::vector<global_size_type>> & sendToPointsGlobalIds,
                    const std::vector<std::vector<double>> & sendToPointsCoords,
                    std::vector<global_size_type> & receivedPointsGlobalIds,
                    std::vector<std::vector<double>> & receivedPointsCoords,
                    const MPI_Comm & mpiComm)
      {

        size_type thisRankId = dealii::Utilities::MPI::this_mpi_process(mpiComm);
        dftfe::utils::mpi::MPIRequestersNBX mpiRequestersNBX(sendToProcIds, mpiComm);
        std::vector<size_type> receiveFromProcIds = mpiRequestersNBX.getRequestingRankIds();

        size_type numMaxProcsSendTo = sendToProcIds.size() ;
        MPI_Allreduce(MPI_IN_PLACE,
                      &numMaxProcsSendTo,
                      1,
                      dftfe::dataTypes::mpi_type_id(&numMaxProcsSendTo),
                      MPI_MAX,
                      mpiComm);

        size_type numMaxProcsReceiveFrom = receiveFromProcIds.size();
        MPI_Allreduce(MPI_IN_PLACE,
                      &numMaxProcsReceiveFrom,
                      1,
                      dftfe::dataTypes::mpi_type_id(&numMaxProcsReceiveFrom),
                      MPI_MAX,
                      mpiComm);

        if(thisRankId == 0)
          {
            std::cout<<" Max number of procs to send to = "<<numMaxProcsSendTo<<"\n";
            std::cout<<" Max number of procs to receive from = "<<numMaxProcsReceiveFrom<<"\n";
          }



        std::vector<std::vector<double>> receivedPointsCoordsProcWise(receiveFromProcIds.size(),
                                                                      std::vector<double>(0));
        std::vector<size_type> numPointsReceived(receiveFromProcIds.size(),-1);

        std::vector<size_type> numPointsToSend(sendToPointsGlobalIds.size(),-1);
        std::vector<MPI_Request> sendRequests(sendToProcIds.size());
        std::vector<MPI_Status>  sendStatuses(sendToProcIds.size());
        std::vector<MPI_Request> recvRequests(receiveFromProcIds.size());
        std::vector<MPI_Status>  recvStatuses(receiveFromProcIds.size());
        const int tag = static_cast<int>(dftfe::utils::mpi::MPITags::MPI_P2P_PATTERN_TAG);
        for(size_type i = 0; i < sendToProcIds.size(); ++i)
          {
            size_type procId = sendToProcIds[i];
            numPointsToSend[i] = sendToPointsGlobalIds[i].size();
            MPI_Isend(&numPointsToSend[i], 1,
                      //                            MPI_UNSIGNED,
                      dftfe::dataTypes::mpi_type_id(&numPointsToSend[i]),
                      procId,
                      procId, // setting the tag to procId
                      mpiComm,
                      &sendRequests[i]);

            //                  std::cout<<"root size id = "<<thisRankId <<" send size to "<<procId<<" id val size = "<<numPointsToSend[i]<<"\n";
          }

        for(size_type i = 0; i < receiveFromProcIds.size(); ++i)
          {
            size_type procId = receiveFromProcIds[i];
            MPI_Irecv(&numPointsReceived[i],
                      1,
                      //                            MPI_UNSIGNED,
                      dftfe::dataTypes::mpi_type_id(&numPointsReceived[i]),
                      procId,
                      thisRankId, // the tag is set to the receiving id
                      mpiComm,
                      &recvRequests[i]);

            //                  std::cout<<"root size id = "<<thisRankId <<" receive size from "<<procId<<" id val size = "<<numPointsReceived[i]<<"\n";
          }


        if (sendRequests.size() > 0)
          {
            int err    = MPI_Waitall(sendToProcIds.size(),
                                  sendRequests.data(),
                                  sendStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            int err    = MPI_Waitall(receiveFromProcIds.size(),
                                  recvRequests.data(),
                                  recvStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        //              for(size_type i = 0; i < receiveFromProcIds.size(); ++i)
        //              {
        //                  size_type procId = receiveFromProcIds[i];
        //                  std::cout<<"root size id = "<<thisRankId <<" receive size from "<<procId<<" id val size = "<<numPointsReceived[i]<<"\n";
        //              }

        const size_type numTotalPointsReceived =
          std::accumulate(numPointsReceived.begin(),
                          numPointsReceived.end(), 0);
        receivedPointsGlobalIds.resize(numTotalPointsReceived, -1);

        for(size_type i = 0; i < sendToProcIds.size(); ++i)
          {
            size_type procId = sendToProcIds[i];
            size_type nPointsToSend = sendToPointsGlobalIds[i].size();
            MPI_Isend(&sendToPointsGlobalIds[i][0],
                      nPointsToSend,
                      dftfe::dataTypes::mpi_type_id(&sendToPointsGlobalIds[i][0]),
                      procId,
                      tag,
                      mpiComm,
                      &sendRequests[i]);

            //                  std::cout<<"root id = "<<thisRankId <<" send to "<<procId<<" id size = "<<nPointsToSend<<"\n";
          }

        size_type offset = 0;
        for(size_type i = 0; i < receiveFromProcIds.size(); ++i)
          {
            size_type procId = receiveFromProcIds[i];
            MPI_Irecv(&receivedPointsGlobalIds[offset],
                      numPointsReceived[i],
                      dftfe::dataTypes::mpi_type_id(&receivedPointsGlobalIds[offset]),
                      procId,
                      tag,
                      mpiComm,
                      &recvRequests[i]);

            //                  std::cout<<"root id = "<<thisRankId <<" receive from "<<procId<<" id size = "<<numPointsReceived[i]<<" offset = "<<offset<<"\n";
            offset += numPointsReceived[i];
          }


        if (sendRequests.size() > 0)
          {
            int err    = MPI_Waitall(sendToProcIds.size(),
                                  sendRequests.data(),
                                  sendStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            int err    = MPI_Waitall(receiveFromProcIds.size(),
                                  recvRequests.data(),
                                  recvStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        //              std::vector<global_size_type> receivedPointsGlobalIdsDummy = receivedPointsGlobalIds;
        //
        //              std::sort(receivedPointsGlobalIdsDummy.begin(),receivedPointsGlobalIdsDummy.end());

        //              if (receivedPointsGlobalIdsDummy.size()>0)
        //              {
        //                  std::cout<<" received from 1  min Ind = "<<receivedPointsGlobalIdsDummy[0]<<" max ind = "<<receivedPointsGlobalIdsDummy[receivedPointsGlobalIdsDummy.size()-1]<<"\n";
        //              }
        for(size_type i = 0; i < sendToProcIds.size(); ++i)
          {
            size_type procId = sendToProcIds[i];
            size_type nPointsToSend = sendToPointsGlobalIds[i].size();
            MPI_Isend(&sendToPointsCoords[i][0],
                      nPointsToSend*dim,
                      MPI_DOUBLE,
                      procId,
                      tag,
                      mpiComm,
                      &sendRequests[i]);
          }

        for(size_type i = 0; i < receiveFromProcIds.size(); ++i)
          {
            size_type procId = receiveFromProcIds[i];
            receivedPointsCoordsProcWise[i].resize(numPointsReceived[i]*dim);
            MPI_Irecv(&receivedPointsCoordsProcWise[i][0],
                      numPointsReceived[i]*dim,
                      MPI_DOUBLE,
                      procId,
                      tag,
                      mpiComm,
                      &recvRequests[i]);
          }

        if (sendRequests.size() > 0)
          {
            int err    = MPI_Waitall(sendToProcIds.size(),
                                  sendRequests.data(),
                                  sendStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            int err    = MPI_Waitall(receiveFromProcIds.size(),
                                  recvRequests.data(),
                                  recvStatuses.data());
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        receivedPointsCoords.resize(numTotalPointsReceived,
                                    std::vector<double>(dim,0.0));
        size_type count = 0;
        for(size_type i = 0; i <receiveFromProcIds.size(); i++)
          {
            std::vector<double> & pointsCoordProc = receivedPointsCoordsProcWise[i];
            for( size_type iPoint = 0 ; iPoint < numPointsReceived[i]; iPoint++)
              {
                for(size_type iDim = 0; iDim  <dim; iDim++)
                  {
                    receivedPointsCoords[count][iDim] = pointsCoordProc[iPoint*dim + iDim];
                  }
                count++;
              }
          }
      }
    }

    template<size_type dim, size_type M>
    MapPointsToCells<dim,M>::MapPointsToCells(const MPI_Comm & mpiComm)
      :d_mpiComm(mpiComm),
      d_numMPIRank(dealii::Utilities::MPI::n_mpi_processes(d_mpiComm)),
      d_thisRank(dealii::Utilities::MPI::this_mpi_process(d_mpiComm))
    {

    }

    template <size_type dim, size_type M>
    void
    MapPointsToCells<dim, M>::init(std::vector<std::shared_ptr<const Cell<dim>>> srcCells,
                                   const std::vector<std::vector<double>> & targetPts,
                                   std::vector<std::vector<double>> &mapCellsToParamCoordinates,
                                   std::vector<std::vector<size_type>> &mapCellLocalToProcLocal,
                                   std::pair<global_size_type,global_size_type> &locallyOwnedRange,
                                   std::vector<global_size_type> & ghostGlobalIds,
                                   const double paramCoordsTol)
    {

      MPI_Barrier(d_mpiComm);
      double startComp = MPI_Wtime();
      size_type numCells =  srcCells.size();
      size_type numPoints = targetPts.size();
      mapCellLocalToProcLocal.resize(numCells,std::vector<size_type>(0));
      mapCellsToParamCoordinates.resize(numCells,std::vector<double>(0));

      std::vector<std::vector<global_size_type>> mapCellLocalToGlobal;
      mapCellLocalToGlobal.resize(numCells,std::vector<global_size_type>(0));

      std::vector<size_type> numLocalPointsInCell(numCells,0);

      // TODO what to do when there are no cells
      std::vector<double> procLowerLeft(dim,0.0);
      std::vector<double> procUpperRight(dim,0.0);
      if(numCells > 0 )
        {
          getProcBoundingBox<dim>(srcCells, procLowerLeft, procUpperRight);
        }

      // get bounding boxes of all the processors
      std::vector<double> allProcsBoundingBoxes(0);
      getAllProcsBoundingBoxes(procLowerLeft,
                               procUpperRight,
                               d_thisRank,
                               d_numMPIRank,
                               d_mpiComm,
                               allProcsBoundingBoxes);

      locallyOwnedRange =
        getLocallyOwnedRange(d_mpiComm,
                             d_thisRank,
                             d_numMPIRank,
                             numPoints);
      const global_size_type locallyOwnedStart = locallyOwnedRange.first;
      const global_size_type locallyOwnedEnd = locallyOwnedRange.second;

      std::vector<bool> pointsFoundLocally(numPoints,false);
      std::vector<std::vector<size_type>> cellLocalFoundIds;
      std::vector<std::vector<double>> cellLocalFoundParamCoords;
      pointsToCell<dim,M>(srcCells,
                           targetPts,
                           cellLocalFoundIds,
                           cellLocalFoundParamCoords,
                           pointsFoundLocally,
                           paramCoordsTol);

      size_type numLocallyFoundPoints = 0;
      for(size_type iCell = 0; iCell < numCells; iCell++)
        {
          numLocalPointsInCell[iCell] = cellLocalFoundIds[iCell].size();

          appendToVec(mapCellLocalToProcLocal[iCell],
                      cellLocalFoundIds[iCell]);

          //                // This does not work because
          //                // the data types are different
          //
          //                appendToVec(mapCellLocalToGlobal[iCell],
          //                        cellLocalFoundIds[iCell]);
          //
          //                // offset each entry of mapCellLocalToGlobal by locallyOwnedStart
          //                std::for_each(mapCellLocalToGlobal[iCell].begin(),
          //                        mapCellLocalToGlobal[iCell].end(),
          //                        [](global_size_type & x) { x += locallyOwnedStart;});

          // initialSize should be zero
          size_type initialSize = mapCellLocalToGlobal[iCell].size();
          size_type finalSize = initialSize + cellLocalFoundIds[iCell].size();
          mapCellLocalToGlobal[iCell].resize(finalSize);
          for(size_type indexVal =  initialSize; indexVal < finalSize; indexVal++)
            {
              mapCellLocalToGlobal[iCell][indexVal] = cellLocalFoundIds[iCell][indexVal - initialSize] + locallyOwnedStart;
              numLocallyFoundPoints++;
            }


          appendToVec(mapCellsToParamCoordinates[iCell],
                      cellLocalFoundParamCoords[iCell]);
        }

      //            std::cout<<" In rank = "<<d_thisRank<<" num points foudn locally  = "<<numLocallyFoundPoints<<"\n";
      MPI_Barrier(d_mpiComm);
      double endLocalComp = MPI_Wtime();
      // get the points that are not found locally
      std::vector<size_type> nonLocalPointLocalIds(0);
      std::vector<std::vector<double>> nonLocalPointCoordinates(0);
      for(size_type iPoint = 0; iPoint < numPoints; iPoint++)
        {
          if(!pointsFoundLocally[iPoint])
            {
              nonLocalPointLocalIds.push_back(iPoint);
              nonLocalPointCoordinates.push_back(targetPts[iPoint]); // TODO will this work ?
            }
        }

      //            std::cout<<" my rank = "<<d_thisRank<<" local start = "<<locallyOwnedStart<<" local end = "<<locallyOwnedEnd<<"\n";
      //            std::vector<size_type> nonLocalIdsDummy = nonLocalPointLocalIds;
      //            std::sort(nonLocalIdsDummy.begin(),nonLocalIdsDummy.end());
      //            if(nonLocalPointLocalIds.size() > 0 )
      //            {
      //                std::cout<<"my rank = "<<d_thisRank<<" Num non local points = "<<nonLocalPointLocalIds.size()<< " start = "<< nonLocalIdsDummy[0]<<" end = "<< nonLocalIdsDummy[nonLocalIdsDummy.size()-1]<<"\n";
      //
      //            }

      std::vector<size_type> sendToProcIds(0);
      std::vector<std::vector<global_size_type>> sendToPointsGlobalIds;
      std::vector<std::vector<double>> sendToPointsCoords;

      getTargetPointsToSend<dim,M>(srcCells,
                                    nonLocalPointLocalIds,
                                    nonLocalPointCoordinates,
                                    allProcsBoundingBoxes,
                                    locallyOwnedStart,
                                    d_thisRank,
                                    d_numMPIRank,
                                    sendToProcIds,
                                    sendToPointsGlobalIds,
                                    sendToPointsCoords);

      std::vector<global_size_type> receivedPointsGlobalIds;
      std::vector<std::vector<double>> receivedPointsCoords;


      //            for(size_type iProcIndex = 0 ; iProcIndex < sendToProcIds.size(); iProcIndex++)
      //            {
      //                size_type iProcId = sendToProcIds[iProcIndex];
      //                std::cout<<"my rank "<<d_thisRank<<" send to proc = "<< iProcId<<" Send to proc size = "<<sendToPointsCoords[iProcIndex].size()<<"\n";
      //                if(sendToPointsGlobalIds[iProcIndex].size() != sendToPointsCoords[iProcIndex].size()/3)
      //                {
      //                    std::cout<<" my rank = "<<d_thisRank<<" send to proc = "<< iProcId<<" error in size of the list sent \n";
      //                }
      //
      //                std::vector<global_size_type> sendToProcIdsDummy = sendToPointsGlobalIds[iProcIndex];
      //
      //                std::sort(sendToProcIdsDummy.begin(),sendToProcIdsDummy.end());
      //
      //                if(sendToProcIdsDummy.size() >  0 )
      //                {
      //                    std::cout<<"My rank = "<<d_thisRank<<" send non local range start = "<<sendToProcIdsDummy[0] << " local range end = "<<sendToProcIdsDummy[sendToProcIdsDummy.size()-1]<<"\n";
      //                }
      //
      //                const bool hasDuplicatesSend = std::adjacent_find(sendToProcIdsDummy.begin(), sendToProcIdsDummy.end()) != sendToProcIdsDummy.end();
      //
      //                std::cout<<"My rank = "<<d_thisRank<<" send to proc = "<< iProcId<<" send has duplicates = "<<hasDuplicatesSend<<"\n";
      //            }

      receivePoints<dim>(sendToProcIds,
                         sendToPointsGlobalIds,
                         sendToPointsCoords,
                         receivedPointsGlobalIds,
                         receivedPointsCoords,
                         d_mpiComm);

      MPI_Barrier(d_mpiComm);
      double endReceive = MPI_Wtime();

      std::cout<<std::flush;
      MPI_Barrier(d_mpiComm);


      size_type numTotalPointsReceived = receivedPointsCoords.size();
      std::vector<std::vector<size_type>> cellReceivedPointsFoundIds;
      std::vector<std::vector<double>> cellReceivedPointsFoundParamCoords;
      std::vector<bool> receivedPointsFound(numTotalPointsReceived, false);
      pointsToCell<dim,M>(srcCells,
                           receivedPointsCoords,
                           cellReceivedPointsFoundIds,
                           cellReceivedPointsFoundParamCoords,
                           receivedPointsFound,
                           paramCoordsTol);


      std::cout<<std::flush;
      MPI_Barrier(d_mpiComm);
      double endNonLocalComp = MPI_Wtime();

      //            ghostIdCoords.resize(0,std::vector<double>(dim,0.0));

      ghostGlobalIds.resize(0);
      std::set<global_size_type> ghostGlobalIdsSet;
      for(size_type iCell = 0; iCell < numCells; iCell++)
        {
          const size_type numPointsReceivedFound = cellReceivedPointsFoundIds[iCell].size();
          const size_type mapCellLocalToGlobalCurrIndex = mapCellLocalToGlobal[iCell].size();
          mapCellLocalToGlobal[iCell].resize(mapCellLocalToGlobalCurrIndex + numPointsReceivedFound);
          for(size_type i = 0; i < numPointsReceivedFound; ++i)
            {
              const size_type pointIndex = cellReceivedPointsFoundIds[iCell][i];
              const global_size_type globalId = receivedPointsGlobalIds[pointIndex];
              mapCellLocalToGlobal[iCell][mapCellLocalToGlobalCurrIndex + i] = globalId;

              //                    if (ghostGlobalIdsSet.find(globalId) != ghostGlobalIdsSet.end())
              //                    {
              //                        std::cout<<" Error ghost detected multiple times \n";
              //                    }
              ghostGlobalIdsSet.insert(globalId);
              //                    ghostGlobalIds.insert(globalId);

            }

          appendToVec(mapCellsToParamCoordinates[iCell],
                      cellReceivedPointsFoundParamCoords[iCell]);
        }

      MPI_Barrier(d_mpiComm);
      double endNonLocalVecComp = MPI_Wtime();
      //            std::cout<<"Non local points found = "<<ghostGlobalIdsSet.size()<<"\n";

      OptimizedIndexSet<global_size_type> ghostGlobalIdsOptIndexSet(ghostGlobalIdsSet);

      std::string errMsgInFindingPoint = "Error in finding ghost index in mapPointsToCells.cpp.";
      for(size_type iCell = 0; iCell < numCells; iCell++)
        {
          const size_type startId = numLocalPointsInCell[iCell];
          const size_type endId = mapCellLocalToGlobal[iCell].size();
          for(size_type iPoint = startId; iPoint < endId; ++iPoint)
            {
              size_type globalId = mapCellLocalToGlobal[iCell][iPoint];
              size_type pos = -1;
              bool found = true;
              ghostGlobalIdsOptIndexSet.getPosition(globalId, pos, found);
              //throwException(found, errMsgInFindingPoint);
              mapCellLocalToProcLocal[iCell].push_back(numPoints+pos);
            }
        }

      size_type ghostSetSize = ghostGlobalIdsSet.size();
      ghostGlobalIds.resize(ghostSetSize,-1);
      size_type ghostIndex = 0;
      for(auto it  = ghostGlobalIdsSet.begin(); it !=  ghostGlobalIdsSet.end(); it++)
        {
          ghostGlobalIds[ghostIndex] = *it;

          ghostIndex++;
        }


      // check of the ghostGlobal Ids is same as the received global ids. Is should match.

      std::cout<<std::flush;
      MPI_Barrier(d_mpiComm);

      double endCompAll =MPI_Wtime();

      global_size_type numNonLocalPointsReceived = numTotalPointsReceived ;
      MPI_Allreduce(MPI_IN_PLACE,
                    &numNonLocalPointsReceived,
                    1,
                    dftfe::dataTypes::mpi_type_id(&numNonLocalPointsReceived),
                    MPI_MAX,
                    d_mpiComm);

      if(d_thisRank ==0)
        {
          std::cout<<" Max number of non local pts received = "<<numNonLocalPointsReceived<<"\n";
          std::cout<<" Time taken for local pts = "<<endLocalComp-startComp<<"\n";
          std::cout<<" Time taken for transfer = "<<endReceive - endLocalComp<<"\n";
          std::cout<<" Time taken for non-local pts = "<<endNonLocalComp - endReceive<<"\n";
          std::cout<<" Time taken for non-local vec gen = "<<endNonLocalVecComp - endNonLocalComp<<"\n";
          std::cout<<" Time for remaining comp = "<<endCompAll - endNonLocalVecComp<<"\n";
        }
    }
  } // end of namespace utils
} // end of namespace dftfe

//for(size_type iCell = 0; iCell < numCells; iCell++)
//{
//    auto bbCell = srcCells[iCell]->getBoundingBox();
//    auto targetPointList = rTreePoint.getPointIdsInsideBox(bbCell.first,
//            bbCell.second); // TODO it should return an empty vector if no point inside cell

//    for(size_type iPoint = 0; iPoint < targetPointList.size(); iPoint++)
//    {
//        size_type pointIndex = targetPointList[iPoint];
//        if(!pointsFoundLocally[pointIndex])
//        {
//            auto paramPoint = srcCells.getParametricPoint(targetPts[pointIndex]); // TODO this will throw an error
//            bool pointInside = true;
//            for( unsigned int j = 0 ; j <dim; j++)
//            {
//                if((paramPoint[j] < -paramCoordsTol) || (paramPoint[j] > 1.0 + paramCoordsTol))
//                {
//                    pointInside = false;
//                }
//            }
//            if (pointInside)
//            {
//                pointsFoundLocally[pointIndex] = true;
//                for(size_type iDim = 0; iDim < dim; iDim++)
//                {
//                    mapCellsToParamCoordinates[iCell].push_back(paramPoint[iDim]);
//                }
//                mapCellLocalToProcLocal[iCell].push_back(pointIndex);
//                mapCellLocalToGlobal[iCell].push_back(pointIndex+locallyOwnedStart);
//            }
//        }
//    }
//}


//RTreePoint rTreePointNonLocal(nonLocalPointCoordinates);
//for (size_type iProc = 0 ; iProc < d_numMPIRank; iProc++)
//{
//    if(iProc != d_thisRank)
//    {
//        auto bbCell = srcCells[iCell]->getBoundingBox();
//        std::vector<double> llProc(dim,0.0);
//        std::vector<double> urProc(dim,0.0);
//        for(size_type iDim = 0; iDim < dim; iDim++)
//        {
//            llProc[iDim] = allProcsBoundingBoxes[2*dim*iProc+j];
//            urProc[iDim] = allProcsBoundingBoxes[2*dim*iProc+dim+j];
//        }
//        auto targetPointList = rTreePointNonLocal.getPointIdsInsideBox(llProc,
//                urProc); // TODO if no points it should return an empty vector ??

//        size_type numTargetPointsToSend = targetPointList.size();
//        std::vector<size_type> globalPointList;
//        globalPointList.resize(numTargetPointsToSend);
//        if(numtargetPointsToSend>0)
//        {
//            sendToProcIds.push_back(iProc);
//            std::vector<double> pointCoordinates;
//            for(size_type iPoint = 0; iPoint < targetPointList.size(); iPoint++)
//            {
//                size_type pointIndex = targetPointList[iPoint];
//                for(size_type iDim = 0; iDim  <dim; iDim++)
//                {
//                    pointCoordinates.push_back(nonLocalPointCoordinates[pointIndex][idim]);
//                }
//                globalPointList[iPoint] = locallyOwnedStart + nonLocalPointIds[targetPointList[iPoint]];
//            }
//            // also have to send the coordinates and the indices.
//            targetPointListToSend[iProc] = globalPointList;
//            targetPointCoordinatesToSend[iProc] = pointCoordinates;
//        }
//    }
//}

//// Not you know what coordinates to send to what procs
//// Use the NBx algorithm to create the communication pattern and
//// send the data.
//// Similarly, receive data other processors.
//MPIRequestersNBX mpiRequestersNBX(sendToProcIds, d_mpiComm);
//std::vector<size_type> receiveFromProcIds = mpiRequestersNBX.getRequestingRankIds();


//std::vector<size_type> numPointsReceived(receiveFromProcIds.size());

//std::vector<MPI_Request> sendRequests(sendToProcIds.size());
//std::vector<MPI_Status>  sendStatuses(sendToProcIds.size());
//std::vector<MPI_Request> recvRequests(receiveFromProcIds.size());
//std::vector<MPI_Status>  recvStatuses(receivedFromProcIds.size());
//const int tag = static_cast<int>(MPITags::MPI_P2P_PATTERN_TAG);
//for(size_type i = 0; i < sendToProcIds.size(); ++i)
//{
//    size_type procId = sendToProcIds[i];
//    size_type nPointsToSend = targetPointListToSend[procId].size();
//    MPI_Isend(&nPointsToSend, 1, MPI_UNSIGNED, procId, tag, d_mpiComm,
//            &sendRequests[i]);
//}

//for(size_type i = 0; i < receiveFromProcIds.size(); ++i)
//{
//    size_type procId = receiveFromProcIds[i];
//    MPI_Irecv(&numPointsReceived[i], 1, MPI_UNSIGNED, procId,
//            tag,
//            d_mpiComm,
//            &recvRequests[i]);
//}
//if (sendRequests.size() > 0)
//{
//    err    = MPI_Waitall(sendToProcIds.size(),
//            sendRequests.data(),
//            sendStatuses.data());
//    errMsg = "Error occured while using MPI_Waitall. "
//        "Error code: " +
//        std::to_string(err);
//    throwException(err == MPI_SUCCESS, errMsg);
//}

//if (recvRequests.size() > 0)
//{
//    err    = MPI_Waitall(receiveFromProcIds.size(),
//            recvRequests.data(),
//            recvStatuses.data());
//    errMsg = "Error occured while using MPI_Waitall. "
//        "Error code: " +
//        std::to_string(err);
//    throwException(err == MPI_SUCCESS, errMsg);
//}

//// send the point coordinates and the global ids.
//std::vector<MPI_Request> sendRequestsCoordinates(sendToProcIds.size());
//std::vector<MPI_Status>  sendStatusesCoordinates(sendToProcIds.size());
//std::vector<MPI_Request> recvRequestsCoordinates(receiveFromProcIds.size());
//std::vector<MPI_Status>  recvStatusesCoordinates(receivedFromProcIds.size());
//// preferably a different value as before.
//const int tagCoordinates = static_cast<int>(MPITags::MPI_P2P_PATTERN_TAG) + 5;

//std::vector<MPI_Request> sendRequestsGlobalIds(sendToProcIds.size());
//std::vector<MPI_Status>  sendStatusesGlobalIds(sendToProcIds.size());
//std::vector<MPI_Request> recvRequestsGlobalIds(receiveFromProcIds.size());
//std::vector<MPI_Status>  recvStatusesGlobalIds(receivedFromProcIds.size());
//const int tagGlobalIds = static_cast<int>(MPITags::MPI_P2P_PATTERN_TAG) + 10;

//for(size_type i = 0; i < sendToProcIds.size(); ++i)
//{
//    size_type procId = sendToProcIds[i];
//    size_type nPointsToSend = targetPointListToSend[procId].size();
//    MPI_Isend(&targetPointCoordinatesToSend[procId][0], nPointsToSend*dim, MPI_DOUBLE, procId, tagCoordinates, d_mpiComm,
//            &sendRequestsCoordinates[i]);

//    MPI_Isend(&targetPointListToSend[procId][0], nPointsToSend, MPI_UNSIGNED, procId, tagGlobalIds, d_mpiComm,
//            &sendRequestsGlobalIds[i]);
//}

//for(size_type i = 0; i < receiveFromProcIds.size(); ++i)
//{
//    size_type procId = receiveFromProcIds[i];
//    receivedPointList[procId] = std::vector<double>(0);
//    receivedPointList[procId].resize(numPointsReceived[i]);
//    receivedPointListCoordinates[procId] = std::vector<double>(0);
//    receivedPointListCoordinates[procId].resize(numPointsReceived[i]*dim);

//    MPI_Irecv(&receivedPointListCoordinates[procId][0], numPointsReceived[i]*dim, MPI_DOUBLE, procId,
//            tagCoordinates,
//            d_mpiComm,
//            &recvRequestsCoordinates[i]);

//    MPI_Irecv(&receivedPointList[procId][0], numPointsReceived[i], MPI_UNSIGNED, procId,
//            tagGlobalIds,
//            d_mpiComm,
//            &recvRequestsGlobalIds[i]);
//}

//if (sendRequestsCoordinates.size() > 0)
//{
//    err    = MPI_Waitall(sendToProcIds.size(),
//            sendRequestsCoordinates.data(),
//            sendStatusesCoordinates.data());
//    errMsg = "Error occured while using MPI_Waitall. "
//        "Error code: " +
//        std::to_string(err);
//    throwException(err == MPI_SUCCESS, errMsg);
//}

//if (recvRequestsCoordinates.size() > 0)
//{
//    err    = MPI_Waitall(receiveFromProcIds.size(),
//            recvRequestsCoordinates.data(),
//            recvStatusesCoordinates.data());
//    errMsg = "Error occured while using MPI_Waitall. "
//        "Error code: " +
//        std::to_string(err);
//    throwException(err == MPI_SUCCESS, errMsg);
//}

//if (sendRequestsGlobalIds.size() > 0)
//{
//    err    = MPI_Waitall(sendToProcIds.size(),
//            sendRequestsGlobalIds.data(),
//            sendStatusesGlobalIds.data());
//    errMsg = "Error occured while using MPI_Waitall. "
//        "Error code: " +
//        std::to_string(err);
//    throwException(err == MPI_SUCCESS, errMsg);
//}

//if (recvRequestsGlobalIds.size() > 0)
//{
//    err    = MPI_Waitall(receiveFromProcIds.size(),
//            recvRequestsGlobalIds.data(),
//            recvStatusesGlobalIds.data());
//    errMsg = "Error occured while using MPI_Waitall. "
//        "Error code: " +
//        std::to_string(err);
//    throwException(err == MPI_SUCCESS, errMsg);
//}


//std::vector<std::vector<double> pointsReceived(numTotalPointsReceived,
//        std::vector<double>(dim,0.0));
//std::vector<global_size_type> pointsReceivedGlobalIds(numTotalPointsReceived, -1);
//size_type pointIndex = 0;
//for(size_type i = 0; i <receiveFromProcIds.size(); i++)
//{
//    size_type iProc = receiveFromProcIds[i];
//    std::vector<size_type> & pointList = receivedPointList[iProc];
//    std::vector<double> & pointsCoord = receivedPointListCoordinates[iProc];
//    for( size_type iPoint = 0 ; iPoint< pointList.size(); iPoint++)
//    {
//        std::vector<double> coord(dim);
//        pointsReceivedGlobalIds[pointIndex] = pointList[iPoint];
//        for(size_type iDim = 0; iDim  <dim; iDim++)
//        {
//            coord[iDim] = pointsCoord[dim*iPoint + iDim];
//        }
//        pointsReceived[pointIndex] = coord;
//        pointIndex++;
//    }
//}


//// search through the points
//RTreePoint rTreeReceivedPoint(pointCoordsReceived);
//std::vector<bool> receivedPointsFoundInACell(numTotalPointsReceived, false);
//for(size_type iCell = 0; iCell < numCells; iCell++)
//{
//    auto bbCell = srcCells[iCell]->getBoundingBox();

//    auto otherProcPointList = rTreeReceivedPoint.getPointIdsInsideBox(bbCell.first,
//            bbCell.second); // TODO it should return an empty vector if no point inside cell

//    for(size_type iPoint = 0; iPoint < otherProcPointList.size(); iPoint++)
//    {
//        size_type pointIndex = otherPointList[iPoint];
//        if(!receivedPointsFoundInACell[pointIndex])
//        {
//            auto paramPoint = srcCells.getParametricPoint(pointCoordsReceived[pointIndex]); // TODO this will throw an error
//            bool pointInside = true;
//            for( unsigned int j = 0 ; j <dim; j++)
//            {
//                if((paramPoint[j] < -paramCoordsTol) || (paramPoint[j] > 1.0 + paramCoordsTol))
//                {
//                    pointInside = false;
//                }
//            }
//            if(pointInside)
//            {
//                receivedPointsFoundInACell[pointIndex] = true;
//                for(size_type iDim = 0; iDim < dim; iDim++)
//                {
//                    mapCellsToParamCoordinates[iCell].push_back(paramPoint[iDim]);
//                }
//                ghostGlobalIdsSet.insert(pointGlobalIdsReceived[pointIndex]);
//                mapCellLocalToGlobal[iCell].push_back(pointGlobalIdsReceived[pointIndex]);
//            }
//        }
//    }
//}
//            // check if the ghost indices are in ascending order.
//
////        sendToProcIds,
////                sendToPointsGlobalIds,
////                sendToPointsCoords
//
//        {
//            size_type numProcsToSendDummy = sendToProcIds.size();
//
//            for( size_type iProc = 0 ; iProc < numProcsToSendDummy; iProc++)
//            {
//                size_type numPointsToSend =  sendToPointsGlobalIds[iProc].size();
//
//                bool ghostIdsInAscending = true;
//                double errorAtCoord = 0.0;
//                for (size_type iPoint = 0; iPoint < numPointsToSend ; iPoint++)
//                {
//                    if (iPoint > 0 )
//                        if(sendToPointsGlobalIds[iProc][iPoint] <=  sendToPointsGlobalIds[iProc][iPoint-1])
//                        {
//                            ghostIdsInAscending = false;
//                        }
//
//                    size_type localNodeIdDummy = sendToPointsGlobalIds[iProc][iPoint] - locallyOwnedStart;
//                    errorAtCoord += (sendToPointsCoords[iProc][3*iPoint + 0] - targetPts[localNodeIdDummy][0])*
//                                    (sendToPointsCoords[iProc][3*iPoint + 0] - targetPts[localNodeIdDummy][0])
//
//                                    + (sendToPointsCoords[iProc][3*iPoint + 1] - targetPts[localNodeIdDummy][1])*
//                                      (sendToPointsCoords[iProc][3*iPoint + 1] - targetPts[localNodeIdDummy][1])
//
//                                    + (sendToPointsCoords[iProc][3*iPoint + 2] - targetPts[localNodeIdDummy][2])*
//                                      (sendToPointsCoords[iProc][3*iPoint + 2] - targetPts[localNodeIdDummy][2]);
//
//                }
//
//                errorAtCoord = std::sqrt(errorAtCoord);
//
//                std::cout<<" sending from proc "<<d_thisRank<<" to "<<sendToProcIds[iProc]<<" ghost are ascending "<<ghostIdsInAscending<<"\n";
//
//                std::cout<<" sending from proc "<<d_thisRank<<" to "<<sendToProcIds[iProc]<<" error at coord = "<<errorAtCoord<<"\n";
//
//
//            }
//        }
//
//
//            // check if the data being sent is correct
//
//
//            // check the received data and ghost global ids are correct
//
//        {
//            bool ghostIdsInAscending = true;
//            for (size_type iPoint = 0; iPoint < receivedPointsGlobalIds.size() ; iPoint++)
//            {
//                if (iPoint > 0 )
//                    if(receivedPointsGlobalIds[iPoint] <=  receivedPointsGlobalIds[iPoint-1])
//                    {
//                        ghostIdsInAscending = false;
//                    }
//            }
//
//            std::cout<<" received data in proc "<<d_thisRank<<" ghost are ascending "<<ghostIdsInAscending<<"\n";
//        }
//
//            // check the coordinates are correct
//
//
//        std::cout<<std::flush;
//        MPI_Barrier(d_mpiComm);
//
//        {
//            if (d_thisRank == 0) {
//
//                for (size_type iPoint = 0; iPoint< 10;iPoint++)
//                {
//                    std::cout<<" receive "<<iPoint
//                             <<" "<<receivedPointsCoords[iPoint][0]
//                             <<" "<<receivedPointsCoords[iPoint][1]
//                             <<" "<<receivedPointsCoords[iPoint][2]<<"\n";
//                }
//
//            }
//
//            std::cout << std::flush;
//            MPI_Barrier(d_mpiComm);
//
//            if (d_thisRank == 1) {
//
//                for (size_type iPoint = 0; iPoint< 10;iPoint++)
//                {
//                    std::cout<<" send "<<iPoint
//                    <<" "<<sendToPointsCoords[0][3*iPoint +0]
//                    <<" "<<sendToPointsCoords[0][3*iPoint +1]
//                    <<" "<<sendToPointsCoords[0][3*iPoint +2]<<"\n";
//                }
//
//            }
//        }
//
//        std::cout<<std::flush;
//        MPI_Barrier(d_mpiComm);


// get the list of locally owned and make sure it is correct before and after accumulate insert
//        {
//            ghostIdCoords.resize(ghostSetSize,std::vector<double>(dim,0.0));
//
//            for( size_type iPoint = 0; iPoint < ghostSetSize; iPoint++)
//            {
//                global_size_type globalIdVal = ghostGlobalIds[iPoint];
//
//                size_type globalIdIndex  = -1;
//
//                bool idFound = false;
//                for( size_type receivedId = 0; receivedId < receivedPointsGlobalIds.size(); receivedId++)
//                {
//                    if(receivedPointsGlobalIds[receivedId] == globalIdVal)
//                    {
//                        globalIdIndex = receivedId;
//                        idFound = true;
//                    }
//                }
//
//                if( !idFound)
//                {
//                    std::cout<<" Errrorrr global id not found in received list \n";
//                }
//                ghostIdCoords[iPoint][0] = receivedPointsCoords[globalIdIndex][0];
//                ghostIdCoords[iPoint][1] = receivedPointsCoords[globalIdIndex][1];
//                ghostIdCoords[iPoint][2] = receivedPointsCoords[globalIdIndex][2];
//            }
//        }
