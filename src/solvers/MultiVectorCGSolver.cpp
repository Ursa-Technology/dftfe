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

#include "MultiVectorCGSolver.h"
namespace dftfe
{
  // constructor
  MultiVectorCGSolver::MultiVectorCGSolver(
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
  void
  MultiVectorCGSolver::solve(MultiVectorLinearSolverProblem<memorySpace> &  problem,
                             std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                                                             BLASWrapperPtr,
        dftfe::linearAlgebra::MultiVector<T,
                                          memorySpace> &  x,
        dftfe::linearAlgebra::MultiVector<T,
                                          memorySpace> &  NDBCVec,
        unsigned int                      locallyOwned,
        unsigned int                      blockSize,
        const double                      absTolerance,
        const unsigned int                maxNumberIterations,
        const unsigned int                debugLevel     = 0,
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
    double omega   = 0.3;
    computing_timer.enter_subsection("Compute Rhs MPI");
    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> rhs_one;
    auto rhs = problem.computeRhs(NDBCVec, x, blockSize);
    computing_timer.leave_subsection("Compute Rhs MPI");

    computing_timer.enter_subsection("CG solver MPI");

    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> g, d, h;

    dftfe::linearAlgebra::MultiVector<T,
                                      memorySpace> tempVec;
    tempVec(x);



    int                 it = 0;
    dftfe::linearAlgebra::MultiVector<double,
                                      memorySpace> resMemSpace, alphaMemSpace, initial_resMemSpace;
    std::vector<double> resHost, alphaHost, initial_resHost;


    dftfe::linearAlgebra::MultiVector<double,
                                      memorySpace> d_onesDevice;
    d_onesDevice.resize(locallyOwned);
    d_onesDevice.setValue(1.0);


    resMemSpace.resize(blockSize);
    alphaMemSpace.resize(blockSize);
    initial_resMemSpace.resize(blockSize);

    resHost.resize(blockSize);
    alphaHost.resize(blockSize);
    initial_resHost.resize(blockSize);


    // resize the vectors, but do not set
    // the values since they'd be overwritten
    // soon anyway.
    g.reinit(x);
    d.reinit(x);
    h.reinit(x);


    // These should be array of size blockSize

    dftfe::linearAlgebra::MultiVector<double,
                                      memorySpace> ghMemSpace, betaMemSpace;

    ghMemSpace.resize(blockSize);
    betaMemSpace.resize(blockSize);


    std::vector<double> ghHost, betaHost;
    ghHost.resize(blockSize);
    betaHost.resize(blockSize);

    problem.vmult(g, x, blockSize);
    BLASWrapperPtr->add(g,rhs,-1.0,blockSize*locallyOwned); // g = g - rhs;

    BLASWrapperPtr->MultiVectorXDot(blockSize,locallyOwned,g.begin(),g.begin(),d_onesDevice.begin(), tempVec.begin(), resMemSpace.begin(), mpi_communicator, resHost.begin());

    pcout << "initial residuals = \n";
    for (unsigned int i = 0; i < blockSize; i++)
      {
        resHost[i]         = std::sqrt(resHost[i]);
        initial_resHost[i] = resHost[i];
        pcout << initial_resHost[i] << "\n";
      }
    pcout << "\n";


    problem.precondition_Jacobi(h, g, omega);
    //    d.equ(-1., h);
    d.setValue(0.0);
    BLASWrapperPtr->add(d,h,-1.0,blockSize*locallyOwned); // d = d - h;

    BLASWrapperPtr->MultiVectorXDot(blockSize,locallyOwned,g.begin(),h.begin(),d_onesDevice.begin(), tempVec.begin(), ghMemSpace.begin(), mpi_communicator, ghHost.begin());
    while (iterate)
      {
        it++;
        problem.vmult(h, d, blockSize);

        BLASWrapperPtr->MultiVectorXDot(blockSize,locallyOwned,h.begin(),d.begin(),d_onesDevice.begin(), tempVec.begin(), alphaMemSpace.begin(), mpi_communicator, alphaHost.begin());
        for (unsigned int i = 0; i < blockSize; i++)
          {
            alphaHost[i] = ghHost[i] / alphaHost[i];
          }

        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         alphaMemSpace.begin(),
                                                                                         alphaHost.begin());
        BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(blockSize,locallyOwned, d, alphaMemSpace, x);
        BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(blockSize,locallyOwned, h, alphaMemSpace, g);

        BLASWrapperPtr->MultiVectorXDot(blockSize,locallyOwned,g.begin(),g.begin(),d_onesDevice.begin(), tempVec.begin(), resMemSpace.begin(), mpi_communicator, resHost.begin());

        for (unsigned int i = 0; i < blockSize; i++)
          {
            resHost[i] = std::sqrt(resHost[i]);
          }
        problem.precondition_Jacobi(h, g, omega);
        betaHost = ghHost;

        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         ghMemSpace.begin(),
                                                                                         ghHost.begin());


        BLASWrapperPtr->MultiVectorXDot(blockSize,locallyOwned,g.begin(),h.begin(),d_onesDevice.begin(), tempVec.begin(), ghMemSpace.begin(), mpi_communicator, ghHost.begin());

        for (unsigned int i = 0; i < blockSize; i++)
          {
            betaHost[i] = (ghHost[i] / betaHost[i]);
          }

        dftfe::utils::MemoryTransfer::copy<memorySpace,dftfe::utils::MemorySpace::HOST>(blockSize,
                                                                                         betaMemSpace.begin(),
                                                                                         betaHost.begin());

        BLASWrapperPtr->stridedBlockScaleColumnWise(blockSize,locallyOwned,betaMemSpace,d);

        BLASWrapperPtr->add(d,h,-1.0,blockSize*locallyOwned); // d = d - h;
        bool convergeStat = true;
        for (unsigned int id = 0; id < blockSize; id++)
          {
            if (std::abs(resHost[id]) > absTolerance)
              convergeStat = false;
          }

        if ((convergeStat) || (it > maxNumberIterations))
          iterate = false;
      }

    problem.distributeX();
    if (it > maxNumberIterations)
      {
        pcout
          << "MultiVector Poisson Solve did not converge. Try increasing the number of iterations or check the input\n";
        pcout << "initial abs. residual: " << initial_res[0]
              << " , current abs. residual: " << res[0] << " , nsteps: " << it
              << " , abs. tolerance criterion:  " << absTolerance << "\n\n";
      }

    else
      pcout << "initial abs. residual: " << initial_res[0]
            << " , current abs. residual: " << res[0] << " , nsteps: " << it
            << " , abs. tolerance criterion:  " << absTolerance << "\n\n";
    computing_timer.leave_subsection("CG solver MPI");
  }
}
