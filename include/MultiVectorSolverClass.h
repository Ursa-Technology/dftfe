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

#ifndef DFTFE_MULTIVECTORSOLVERCLASS_H
#define DFTFE_MULTIVECTORSOLVERCLASS_H

namespace dftfe
{
  class MultiVectorSolverClass
  {
  public:
//    template <dftfe::utils::MemorySpace memorySpace, typename T>
//    virtual void
//    solve(MultiVectorLinearSolverProblem<memorySpace> &  problem,
//          dftfe::linearAlgebra::MultiVector<T,
//                                            memorySpace> &  x,
//          dftfe::linearAlgebra::MultiVector<T,
//                                            memorySpace> & NDBCVec,
//          unsigned int                      blockSize,
//          const double                      absTolerance,
//          const unsigned int                maxNumberIterations,
//          const unsigned int                debugLevel     = 0,
//          bool                              distributeFlag = true) = 0;
  };
}


#endif // DFTFE_MULTIVECTORSOLVERCLASS_H
