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

#ifndef DFTFE_MULTIVECTORCGSOLVER_H
#define DFTFE_MULTIVECTORCGSOLVER_H

namespace dftfe
{
  class MultiVectorCGSolver : public MultiVectorSolverClass
  {
  public :

    MultiVectorCGSolver( const MPI_Comm &mpi_comm_parent,
                        const MPI_Comm &mpi_comm_domain);
    template <dftfe::utils::MemorySpace memorySpace, typename T>
    void
    solve(MultiVectorLinearSolverProblem<memorySpace> &  problem,
          dftfe::linearAlgebra::MultiVector<T,
                                            memorySpace> &  x,
          dftfe::linearAlgebra::MultiVector<T,
                                            memorySpace> &  NDBCVec,
          unsigned int                      blockSize,
          const double                      absTolerance,
          const unsigned int                maxNumberIterations,
          const unsigned int                debugLevel     = 0,
          bool                              distributeFlag = true) override;

  private:
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };
}

#endif // DFTFE_MULTIVECTORCGSOLVER_H
