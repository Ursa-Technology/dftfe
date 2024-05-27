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

#ifndef DFTFE_MULTIVECTORMINRESSOLVER_H
#define DFTFE_MULTIVECTORMINRESSOLVER_H

#include "MultiVectorLinearSolverProblem.h"
#include "headers.h"
#include "MultiVectorSolverClass.h"
#include "BLASWrapper.h"
namespace dftfe
{
  class MultiVectorMinResSolver //: public MultiVectorSolverClass
  {
  public :

    MultiVectorMinResSolver( const MPI_Comm &mpi_comm_parent,
                        const MPI_Comm &mpi_comm_domain);
    template <dftfe::utils::MemorySpace memorySpace>
    void
    solve(MultiVectorLinearSolverProblem<memorySpace> &  problem,
          std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                                          BLASWrapperPtr,
          dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                            memorySpace> &  xMemSpace,
          dftfe::linearAlgebra::MultiVector<dataTypes::number ,
                                            memorySpace> &  NDBCVec,
          unsigned int                      locallyOwned,
          unsigned int                      blockSize,
          const double                      absTolerance,
          const unsigned int                maxNumberIterations,
          const unsigned int                debugLevel = 0,
          bool                              distributeFlag = true) ;//override;

  private:
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };
}

#endif // DFTFE_MULTIVECTORMINRESSOLVER_H
