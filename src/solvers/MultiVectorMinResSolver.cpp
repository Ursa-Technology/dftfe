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
// @author Vishal Subramanian
//

#include "MultiVectorMinResSolver.h"

namespace dftfe
{
  namespace
  {
    void
    assignVecIds(std::vector<unsigned int> &nVecsInProc,
                 std::vector<unsigned int> &procVecStartId,
                 unsigned int               numVecs,
                 const unsigned int         n_mpi_processes,
                 const unsigned int         this_mpi_process)
    {
      nVecsInProc.resize(n_mpi_processes, 0);
      procVecStartId.resize(n_mpi_processes, 0);
      procVecStartId[0] = 0;
      for (unsigned int iProc = 0; iProc < n_mpi_processes; ++iProc)
        {
          nVecsInProc[iProc] = numVecs / n_mpi_processes;
          if (iProc < numVecs % n_mpi_processes)
            {
              nVecsInProc[iProc]++;
            }
        }

      for (unsigned int iProc = 1; iProc < n_mpi_processes; ++iProc)
        {
          procVecStartId[iProc] =
            procVecStartId[iProc - 1] + nVecsInProc[iProc - 1];
        }
    }
  }

  // constructor
  MultiVectorMinResSolver::MultiVectorMinResSolver(
    const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain)
    : mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {}

  template <dftfe::utils::MemorySpace memorySpace, typename T>
  void MultiVectorMinResSolver::solve(MultiVectorLinearSolverProblem<memorySpace> &  problem,
                                 dftfe::linearAlgebra::MultiVector<T,
                                                                   memorySpace> &  xMemSpace,
                                 dftfe::linearAlgebra::MultiVector<T,
                                                                   memorySpace> &  NDBCVec,
                                 unsigned int                      locallyOwned,
                                 unsigned int                      blockSize,
                                 const double                      absTolerance,
                                 const unsigned int                maxNumberIterations,
                                 const unsigned int                debugLevel,
                                 bool                              distributeFlag)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);

    dealii::TimerOutput computing_timer(mpi_communicator,
                                        pcout,
                                        dealii::TimerOutput::summary,
                                        dealii::TimerOutput::wall_times);

    bool   iterate = true;
    double omega   = 1.0;

    dftfe::linearAlgebra::MultiVector<double,
                                      memorySpace> d_onesDevice;
    d_onesDevice.resize(locallyOwned);
    d_onesDevice.setValue(1.0);

    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> tempVec;
    tempVec(xMemSpace);

    // TODO use dft parameters to get this
    const double rhsNormTolForZero = 1e-15;
    computing_timer.enter_subsection("Compute Rhs MPI");
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> bMemSpace = problem.computeRhs(NDBCVec, xMemSpace, blockSize);

    pcout << "Compute rhs ... done \n";

    computing_timer.leave_subsection("Compute Rhs MPI");

    computing_timer.enter_subsection("MINRES solver MPI");
    computing_timer.enter_subsection("MINRES initial MPI");

    //
    // assign vector Ids to processors uniformly
    //
    std::vector<unsigned int> nVecsInProcs(0);
    std::vector<unsigned int> procVecStartId(0);
    assignVecIds(nVecsInProcs,
                 procVecStartId,
                 blockSize,
                 n_mpi_processes,
                 this_mpi_process);

    std::vector<T> negOneHost(blockSize, -1.0);
    std::vector<T> beta1Host(blockSize, 0.0);


    dftfe::utils::MemoryStorage<T, memorySpace>
      beta1MemSpace(blockSize, 0.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      alphaMemSpace(blockSize, 0.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      oneMemSpace(blockSize, 1.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      sMemSpace(blockSize, 1.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      negBetaByBetaOldMemSpace(blockSize, 1.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      negAlphaByBetaMemSpace(blockSize, 1.0);


    dftfe::utils::MemoryStorage<T, memorySpace>
      negOldepsMemSpace(blockSize, 1.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      negOldepsMemSpace(blockSize, 1.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      negDeltaMemSpace(blockSize, 1.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      denomMemSpace(blockSize, 1.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      phiMemSpace(blockSize, 1.0);

    dftfe::utils::MemoryStorage<T, memorySpace>
      coeffForXMemInMemSpace(blockSize, 0.0);
    dftfe::utils::MemoryStorage<T, memorySpace>
      coeffForXTmpMemSpace(blockSize, 0.0);

    // allocate the vectors
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> xTmpMemSpace, yMemSpace, vMemSpace, r1MemSpace, r2MemSpace, wMemSpace, w1MemSpace, w2MemSpace;
    xTmpMemSpace.reinit(bMemSpace);
    yMemSpace.reinit(bMemSpace);
    vMemSpace.reinit(bMemSpace);
    r1MemSpace.reinit(bMemSpace);
    r2MemSpace.reinit(bMemSpace);
    wMemSpace.reinit(bMemSpace);
    w1MemSpace.reinit(bMemSpace);
    w2MemSpace.reinit(bMemSpace);

    BLASWrapperPtr->axpby(locallyOwned*blockSize,
                          1.0,
                          x.begin(),
                          0.0,
                          xTmpMemSpace.begin());

    problem.vmult(r1MemSpace, xTmpMemSpace, blockSize);


    BLASWrapperPtr->multiVectorScaleAndAdd(blockSize,locallyOwned, bMemSpace.data(), negOneMemSpace, r1MemSpace.begin());

    BLASWrapperPtr->axpby(locallyOwned*blockSize,
                          0.0,
                          xTmpMemSpace.begin(),
                          -1.0,
                          r1MemSpace.begin());

    problem.precondition_Jacobi(yMemSpace, r1MemSpace, omega);

    BLASWrapperPtr->MultiVectorXDot(blockSize,
                                    locallyOwned,
                                    r1MemSpace.begin(),
                                    yMemSpace.begin(),
                                    d_onesDevice.begin(),
                                    tempVec.begin(),
                                    beta1MemSpace.begin(),
                                    mpi_communicator,
                                    beta1Host.begin());

    bool notPosDef = std::any_of(beta1Host.begin(), beta1Host.end(), [](double val) {
      return val <= 0.0;
    });

    for (unsigned int i = 0; i < blockSize; ++i)
      beta1Host[i] = std::sqrt(beta1Host[i]) + rhsNormTolForZero;

    std::vector<double> epsHost(blockSize, std::numeric_limits<double>::epsilon());
    std::vector<double> oldbHost(blockSize, 0.0);
    std::vector<double> betaHost(beta1Host);
    std::vector<double> dbarHost(blockSize, 0.0);
    std::vector<double> epslnHost(blockSize, 0.0);
    std::vector<double> oldepsHost(blockSize, 0.0);
    std::vector<double> qrnormHost(beta1);
    std::vector<double> phiHost(blockSize, 0.0);
    std::vector<double> phibarHost(beta1);
    std::vector<double> csHost(blockSize, -1.0);
    std::vector<double> snHost(blockSize, 0.0);
    std::vector<double> alphaHost(blockSize, 0.0);
    std::vector<double> gammaHost(blockSize, 0.0);
    std::vector<double> deltaHost(blockSize, 0.0);
    std::vector<double> gbarHost(blockSize, 0.0);
    std::vector<double> rnormHost(blockSize, 0.0);

    scale(r1.data(), oneMemSpace.begin(), nLocallyOwned, blockSize, r2.data());


    BLASWrapperPtr->axpby(locallyOwned*blockSize,
                          1.0,
                          r1MemSpace.begin(),
                          0.0,
                          r2MemSpace.begin());

    bool                      hasAllConverged = false;
    std::vector<bool>         hasConvergedHost(blockSize, false);
    std::vector<double>       sHost(blockSize, 0.0);
    std::vector<double>       negBetaByBetaOldHost(blockSize, 0.0);
    std::vector<double>       negAlphaByBetaHost(blockSize, 0.0);
    std::vector<double>       denomHost(blockSize, 0.0);
    std::vector<double>       negOldepsHost(blockSize, 0.0);
    std::vector<double>       negDeltaHost(blockSize, 0.0);
    std::vector<unsigned int> lanczosSizeHost(blockSize, 0);
    unsigned int              iter = 0;
    computing_timer.leave_subsection("MINRES initial MPI");
    while (iter < maxNumberIterations && hasAllConverged == false)
      {
        computing_timer.enter_subsection("MINRES vmult MPI");
        for (unsigned int i = 0; i < blockSize; ++i)
          sHost[i] = 1.0 / betaHost[i];

        BLASWrapperPtr->axpby(locallyOwned*blockSize,
                              1.0,
                              yMemSpace.begin(),
                              0.0,
                              vMemSpace.begin());
        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         sMemSpace.begin(),
                                                                                         sHost.begin());

        BLASWrapperPtr->stridedBlockScaleColumnWise(blockSize,locallyOwned,sMemSpace,vMemSpace);

        problem.vmult(yMemSpace, vMemSpace, blockSize);

        if (iter > 0)
          {
            for (unsigned int i = 0; i < blockSize; ++i)
              negBetaByBetaOldHost[i] = -betaHost[i] / oldbHost[i];

            dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                             negBetaByBetaOldMemSpace.begin(),
                                                                                             negBetaByBetaOldHost.begin());

            BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(blockSize,locallyOwned, r1, negBetaByBetaOldMemSpace, y.begin());

          }
        computing_timer.leave_subsection("MINRES vmult MPI");
        computing_timer.enter_subsection("MINRES linalg MPI");

        BLASWrapperPtr->MultiVectorXDot(blockSize,
                                        locallyOwned,
                                        vMemSpace.begin(),
                                        yMemSpace.begin(),
                                        d_onesDevice.begin(),
                                        tempVec.begin(),
                                        alphaMemSpace.begin(),
                                        mpi_communicator,
                                        alphaHost.begin());

        for (unsigned int i = 0; i < blockSize; ++i)
          negAlphaByBetaHost[i] = -alphaHost[i] / betaHost[i];

        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         negAlphaByBetaMemSpace.begin(),
                                                                                         negAlphaByBetaHost.begin());

        BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(blockSize,locallyOwned, r2, negAlphaByBetaMemSpace, y.begin());

        BLASWrapperPtr->axpby(locallyOwned*blockSize,
                              1.0,
                              r2MemSpace.begin(),
                              0.0,
                              r1MemSpace.begin());

        BLASWrapperPtr->axpby(locallyOwned*blockSize,
                              1.0,
                              yMemSpace.begin(),
                              0.0,
                              r2MemSpace.begin());

        problem.precondition_Jacobi(yMemSpace, r2MemSpace, omega);

        oldbHost = betaHost;

        BLASWrapperPtr->MultiVectorXDot(blockSize,
                                        locallyOwned,
                                        r2MemSpace.begin(),
                                        yMemSpace.begin(),
                                        d_onesDevice.begin(),
                                        tempVec.begin(),
                                        betaMemSpace.begin(),
                                        mpi_communicator,
                                        betaHost.begin());

        notPosDef = std::any_of(betaHost.begin(), betaHost.end(), [](double val) {
          return val <= 0.0;
        });

        for (unsigned int i = 0; i < blockSize; ++i)
          {
            betaHost[i] = std::sqrt(betaHost[i]) + rhsNormTolForZero;
          }

        for (unsigned int i = 0; i < blockSize; ++i)
          {
            oldepsHost[i] = epslnHost[i];
            deltaHost[i]  = csHost[i] * dbarHost[i] + snHost[i] * alphaHost[i];
            gbarHost[i]   = snHost[i] * dbarHost[i] - csHost[i] * alphaHost[i];
            epslnHost[i]  = snHost[i] * betaHost[i];
            dbarHost[i]   = -csHost[i] * betaHost[i];

            // Compute next plane rotation Q_k
            gammaHost[i]  = sqrt(gbarHost[i] * gbarHost[i] + betaHost[i] * betaHost[i]); // gamma_k
            gammaHost[i]  = std::max(gammaHost[i], epsHost[i]);
            csHost[i]     = gbarHost[i] / gammaHost[i]; // c_k
            snHost[i]     = betaHost[i] / gammaHost[i]; // s_k
            phiHost[i]    = csHost[i] * phibarHost[i];  // phi_k
            phibarHost[i] = snHost[i] * phibarHost[i];  // phibar_{k+1}
            denomHost[i]  = 1.0 / gammaHost[i];
            negOldepsHost[i] = -oldepsHost[i];
            negDeltaHost[i]  = -deltaHost[i];
          }

        BLASWrapperPtr->axpby(locallyOwned*blockSize,
                              1.0,
                              w2MemSpace.begin(),
                              0.0,
                              w1MemSpace.begin());

        BLASWrapperPtr->axpby(locallyOwned*blockSize,
                              1.0,
                              wMemSpace.begin(),
                              0.0,
                              w2MemSpace.begin());

        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         negOldepsMemSpace.begin(),
                                                                                         negOldepsHost.begin());

        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         negDeltaMemSpace.begin(),
                                                                                         negDeltaHost.begin());

        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         denomMemSpace.begin(),
                                                                                         denomHost.begin());

        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         phiMemSpace.begin(),
                                                                                         phiHost.begin());
          BLASWrapperPtr->stridedBlockScaleAndAddTwoVecColumnWise(
            blockSize,
            locallyOwned,
            w1,
            negOldepsMemSpace,
            w2,
            negOldepsMemSpace,
            w);

          BLASWrapperPtr->stridedBlockScaleAndAddTwoVecColumnWise(
            blockSize,
            locallyOwned,
            v,
            denomMemSpace,
            w,
            denomMemSpace,
            w);

          BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(blockSize,locallyOwned, wMemSpace, phiDeviceMemSpace, xTmpMemSpace.begin());

        for (unsigned int i = 0; i < blockSize; ++i)
          {
            qrnormHost[i] = phibarHost[i];
            rnormHost[i]  = qrnormHost[i];
          }

        pcout << " iter = " << iter << "\n";
        bool                updateFlag = false;
        std::vector<double> coeffForXMemHost(blockSize, 1.0);
        std::vector<double> coeffForXTmpHost(blockSize, 0.0);
        for (unsigned int i = 0; i < blockSize; ++i)
          {
            if (rnorm[i] < absTolerance && hasConverged[i] == false)
              {
                updateFlag         = true;
                hasConverged[i]    = true;
                lanczosSize[i]     = iter + 1;
                coeffForXMemHost[i] = 0.0;
                coeffForXTmpHost[i]    = 1.0;
              }
          }
        if (updateFlag)
          {
            dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                             coeffForXMemInMemSpace.begin(),
                                                                                             coeffForXMemHost.begin());

            dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                             coeffForXTmpMemSpace.begin(),
                                                                                             coeffForXTmpHost.begin());

            BLASWrapperPtr->stridedBlockScaleAndAddTwoVecColumnWise(
              blockSize,
              locallyOwned,
              xMemSpace,
              coeffForXMemInMemSpace,
              xTmpMemSpace,
              coeffForXTmpMemSpace,
              xMemSpace);
          }

        if (std::all_of(hasConverged.begin(),
                        hasConverged.end(),
                        [](bool boolVal) { return boolVal; }))
          {
            hasAllConverged = true;
          }

        computing_timer.leave_subsection("MINRES linalg MPI");
        iter++;
      }

    computing_timer.enter_subsection("MINRES dist MPI");

    bool                updateUncovergedFlag = false;
    std::vector<double> coeffForXMemHost(blockSize, 1.0);
    std::vector<double> coeffForXTmpHost(blockSize, 0.0);
    for (unsigned int i = 0; i < blockSize; ++i)
      {
        if (hasConverged[i] == false)
          {
            pcout << " MINRES SOlVER not converging for iBlock = " << i << "\n";
            updateUncovergedFlag = true;
            coeffForXMemHost[i]   = 0.0;
            coeffForXTmpHost[i]      = 1.0;
          }
      }

    if (updateUncovergedFlag)
      {
        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         coeffForXMemInMemSpace.begin(),
                                                                                         coeffForXMemHost.begin());

        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         coeffForXTmpMemSpace.begin(),
                                                                                         coeffForXTmpHost.begin());

        BLASWrapperPtr->stridedBlockScaleAndAddTwoVecColumnWise(
          blockSize,
          locallyOwned,
          xMemSpace,
          coeffForXMemInMemSpace,
          xTmpMemSpace,
          coeffForXTmpMemSpace,
          xMemSpace);
      }

    problem.distributeX();
    computing_timer.leave_subsection("MINRES dist MPI");
    dftfe::utils::deviceSynchronize();
  }
}