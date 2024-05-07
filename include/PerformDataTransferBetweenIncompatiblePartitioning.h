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

#ifndef DFTFE_PERFORMDATATRANSFERBETWEENINCOMPATIBLEMESHES_H
#define DFTFE_PERFORMDATATRANSFERBETWEENINCOMPATIBLEMESHES_H


#include "headers.h"
#include "dftUtils.h"

namespace dftfe
{

  void performDataTransferBetweenIncompatibleMeshes(const MPI_Comm & mpi_comm_domain,
                                               const size_type & feOrder);

} // end of namespace dftfe

#endif //DFTFE_PERFORMDATATRANSFERBETWEENINCOMPATIBLEMESHES_H
