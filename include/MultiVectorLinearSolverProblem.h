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


#ifndef DFTFE_MULTIVECTORLINEARSOLVERPROBLEM_H
#define DFTFE_MULTIVECTORLINEARSOLVERPROBLEM_H

#include "headers.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class MultiVectorLinearSolverProblem
  {
  public:
    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
     template <typename T>
    virtual dftfe::linearAlgebra::MultiVector<T,
                                              memorySpace> &
    computeRhs(
               dftfe::linearAlgebra::MultiVector<T,
                                                 memorySpace> &       NDBCVec,
               dftfe::linearAlgebra::MultiVector<T,
                                                 memorySpace> &       outputVec,
               unsigned int                      blockSizeInput) = 0;

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
     template <typename T>
    virtual void vmult(dftfe::linearAlgebra::MultiVector<T,
                                             memorySpace> &Ax,
           dftfe::linearAlgebra::MultiVector<T,
                                             memorySpace> &x,
          unsigned int               blockSize) = 0;

    /**
     * @brief Apply the constraints to the solution vector.
     *
     */
    virtual void
    distributeX() = 0;

    /**
     * @brief Jacobi preconditioning function.
     *
     */
    template <typename T>
    virtual void
    precondition_Jacobi(dftfe::linearAlgebra::MultiVector<T,
                                                          memorySpace> &      dst,
                        const dftfe::linearAlgebra::MultiVector<T,
                                                                memorySpace> &src,
                        const double                     omega) const = 0;

    /**
     * @brief Apply square-root of the Jacobi preconditioner function.
     *
     */
    template <typename T>
    virtual void
    precondition_JacobiSqrt(ddftfe::linearAlgebra::MultiVector<T,
                                                               memorySpace> &      dst,
                            const dftfe::linearAlgebra::MultiVector<T,
                                                                    memorySpace> &src,
                            const double omega) const = 0 ;


  };

} // end of namespace dftfe

#endif // DFTFE_MULTIVECTORLINEARSOLVERPROBLEM_H
