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

#include "TransferBetweenMeshesIncompatiblePartitioning.h"

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>::
    TransferDataBetweenMeshesIncompatiblePartitioning(const dealii::MatrixFree<3, double> &matrixFreeMesh1,
                                                      const unsigned int                   matrixFreeMesh1VectorComponent,
                                                      const unsigned int matrixFreeMesh1QuadratureComponent,
                                                      const dealii::MatrixFree<3, double> &matrixFreeMesh2,
                                                      const unsigned int                   matrixFreeMesh2VectorComponent,
                                                      const unsigned int matrixFreeMesh2QuadratureComponent,
                                                      const MPI_Comm & mpiComm):
    d_mpiComm(mpiComm)
  {

    d_matrixFreeMesh1Ptr             = &matrixFreeMesh1;
    d_matrixFreeMesh1VectorComponent     = matrixFreeMesh1VectorComponent;
    d_matrixFreeMesh1QuadratureComponent = matrixFreeMesh1QuadratureComponent;


    d_matrixFreeMesh2Ptr             = &matrixFreeMesh2;
    d_matrixFreeMesh2VectorComponent     = matrixFreeMesh2VectorComponent;
    d_matrixFreeMesh2QuadratureComponent = matrixFreeMesh2QuadratureComponent;

    const dealii::DoFHandler<3> *dofHandlerMesh1 = &d_matrixFreeMesh1Ptr->get_dof_handler(
      matrixFreeMesh1VectorComponent);


    const dealii::DoFHandler<3> *dofHandlerMesh2 = &d_matrixFreeMesh2Ptr->get_dof_handler(
      matrixFreeMesh2VectorComponent);

    dealii::Quadrature<3> quadratureMesh1 = d_matrixFreeMesh1Ptr->get_quadrature(
      d_matrixFreeMesh1QuadratureComponent);


    dealii::Quadrature<3> quadratureMesh2 = d_matrixFreeMesh2Ptr->get_quadrature(
      d_matrixFreeMesh2QuadratureComponent);

    dealii::FEValues<3> fe_valuesMesh1(dofHandlerMesh1->get_fe(),
                                       quadratureMesh1,
                                       dealii::update_values |
                                         dealii::update_quadrature_points);

    dealii::FEValues<3> fe_valuesMesh2(dofHandlerMesh2->get_fe(),
                                       quadratureMesh2,
                                       dealii::update_values |
                                         dealii::update_quadrature_points);

    size_type numberQuadraturePointsMesh1 = quadratureMesh1.size();
    size_type numberQuadraturePointsMesh2 = quadratureMesh2.size();

    size_type totallyOwnedCellsMesh1 = d_matrixFreeMesh1Ptr->n_physical_cells();
    size_type totallyOwnedCellsMesh2 = d_matrixFreeMesh2Ptr->n_physical_cells();

    std::vector<std::vector<double>> quadPointsMesh1 (totallyOwnedCellsMesh1*numberQuadraturePointsMesh1,
                                                     std::vector<double>(3,0.0));

    std::vector<std::vector<double>> quadPointsMesh2 (totallyOwnedCellsMesh2*numberQuadraturePointsMesh2,
                                                     std::vector<double>(3,0.0));



    //      std::cout<<" totla cell = "<<totallyOwnedCellsMesh1<<" quad size = "<<quadPointsMesh2.size()<<"\n";
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellMesh1 = dofHandlerMesh1->begin_active(),
      endcMesh1 = dofHandlerMesh1->end();
    // iterate through child cells
    size_type iElemIndex = 0;
    for (; cellMesh1 != endcMesh1; cellMesh1++)
      {
        if (cellMesh1->is_locally_owned())
          {
            fe_valuesMesh1.reinit(cellMesh1);
            for (unsigned int iQuad = 0; iQuad < numberQuadraturePointsMesh1; iQuad++)
              {
                dealii::Point<3, double> qPointVal =
                  fe_valuesMesh1.quadrature_point(iQuad);

                for (size_type iDim = 0; iDim < 3; iDim++)
                  {
                    quadPointsMesh1[iElemIndex*numberQuadraturePointsMesh1 + iQuad][iDim] =
                      qPointVal[iDim];
                  }
              }
            iElemIndex++;
          }

      }

    //    std::cout<<" numcell = "<<iElemIndex<<" totla cell = "<<totallyOwnedCellsMesh1<<" quad size = "<<quadPointsMesh1.size()<<"\n";
    typename dealii::DoFHandler<3>::active_cell_iterator
      cellMesh2 = dofHandlerMesh2->begin_active(),
      endcMesh2 = dofHandlerMesh2->end();
    // iterate through child cells
    iElemIndex = 0;
    for (; cellMesh2 != endcMesh2; cellMesh2++)
      {
        if (cellMesh2->is_locally_owned())
          {
            fe_valuesMesh2.reinit(cellMesh2);
            for (unsigned int iQuad = 0; iQuad < numberQuadraturePointsMesh2; iQuad++)
              {
                dealii::Point<3, double> qPointVal =
                  fe_valuesMesh2.quadrature_point(iQuad);

                for (size_type iDim = 0; iDim < 3; iDim++)
                  {
                    quadPointsMesh2[iElemIndex*numberQuadraturePointsMesh2 + iQuad][iDim] = qPointVal[iDim];
                  }
              }
            iElemIndex++;
          }

      }

    std::cout<<std::flush;
    MPI_Barrier(d_mpiComm);

    double startMapMesh1To2 = MPI_Wtime();

    d_mesh1toMesh2 = std::make_shared<InterpolateCellWiseDataToPoints<memorySpace>>(*dofHandlerMesh1,
                                                                            quadPointsMesh2,
                                                                            d_mpiComm);

    std::cout<<std::flush;
    MPI_Barrier(d_mpiComm);
    double endMapMesh1To2 = MPI_Wtime();

    d_mesh2toMesh1 = std::make_shared<InterpolateCellWiseDataToPoints<memorySpace>>(*dofHandlerMesh2,
                                                                            quadPointsMesh1,
                                                                            d_mpiComm);

    std::cout<<std::flush;
    MPI_Barrier(d_mpiComm);

    double endMapMesh2To1 = MPI_Wtime();

    if(  dealii::Utilities::MPI::this_mpi_process(d_mpiComm) == 0 )
      {
        std::cout<<" Time taken to create the map from 1 to 2 = "<<endMapMesh1To2 - startMapMesh1To2<<"\n";
        std::cout<<" Time taken to create the map from 2 to 1 = "<<endMapMesh2To1 - endMapMesh1To2<<"\n";
      }
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>::
    interpolateMesh1DataToMesh2QuadPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &
        BLASWrapperPtr,
      const dftfe::linearAlgebra::MultiVector<T,
                                              memorySpace> &inputVec,
      const unsigned int                    numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
      dftfe::utils::MemoryStorage<T, memorySpace> &outputQuadData,
      bool resizeOutputVec)
  {
    d_mesh1toMesh2->interpolateSrcDataToTargetPoints(
      BLASWrapperPtr,
      inputVec,
      numberOfVectors,
      fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
      outputQuadData,
      resizeOutputVec );
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>::interpolateMesh2DataToMesh1QuadPoints(
    const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &
      BLASWrapperPtr,
    const dftfe::linearAlgebra::MultiVector<T,
                                            memorySpace> &inputVec,
    const unsigned int                    numberOfVectors,
    const dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace> &fullFlattenedArrayCellLocalProcIndexIdMapMesh2,
    dftfe::utils::MemoryStorage<T, memorySpace> &                 outputQuadData,
    bool resizeOutputVec)
  {
    d_mesh2toMesh1->interpolateSrcDataToTargetPoints(
      BLASWrapperPtr,
      inputVec,
      numberOfVectors,
      fullFlattenedArrayCellLocalProcIndexIdMapMesh2,
      outputQuadData,
      resizeOutputVec );
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>::interpolateMesh2DataToMesh1QuadPoints(
    const distributedCPUVec<T> &inputVec,
    const unsigned int               numberOfVectors,
    const dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::HOST> &mapVecToCells,
    dftfe::utils::MemoryStorage<T,
                                dftfe::utils::MemorySpace::HOST> &            outputQuadData,
    bool resizeOutputVec)
  {

    d_mesh2toMesh1->interpolateSrcDataToTargetPoints(
      inputVec,
      numberOfVectors,
      mapVecToCells,
      outputQuadData,
      resizeOutputVec );
  }

  template <dftfe::utils::MemorySpace memorySpace>
  template <typename T>
  void
  TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>::interpolateMesh1DataToMesh2QuadPoints(
    const distributedCPUVec<T> &inputVec,
    const unsigned int               numberOfVectors,
    const dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::HOST> & fullFlattenedArrayCellLocalProcIndexIdMapParent,
    dftfe::utils::MemoryStorage<T,
                                dftfe::utils::MemorySpace::HOST> &outputQuadData,
    bool resizeOutputVec)
  {
    d_mesh1toMesh2->interpolateSrcDataToTargetPoints(
      inputVec,
      numberOfVectors,
      fullFlattenedArrayCellLocalProcIndexIdMapParent,
      outputQuadData,
      resizeOutputVec );
  }


  template
  void
    TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::HOST>
      ::interpolateMesh1DataToMesh2QuadPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &
        BLASWrapperPtr,
    const dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                              dftfe::utils::MemorySpace::HOST> &inputVec,
    const unsigned int                    numberOfVectors,
    const dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::HOST> &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
    dftfe::utils::MemoryStorage<dataTypes::number, dftfe::utils::MemorySpace::HOST> &outputQuadData,
    bool resizeOutputVec); // override;

  template
  void
  TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::HOST>
    ::interpolateMesh2DataToMesh1QuadPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>> &
        BLASWrapperPtr,
    const dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                              dftfe::utils::MemorySpace::HOST> &inputVec,
    const unsigned int                    numberOfVectors,
    const dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::HOST> &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
    dftfe::utils::MemoryStorage<dataTypes::number, dftfe::utils::MemorySpace::HOST> &                 outputQuadData,
    bool resizeOutputVec); // override;

  template
  void
    TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::HOST>
      ::interpolateMesh1DataToMesh2QuadPoints(
    const distributedCPUVec<dataTypes::number> &inputVec,
    const unsigned int numberOfVectors,
    const dftfe::utils::MemoryStorage<dealii::types::global_dof_index, dftfe::utils::MemorySpace::HOST>
                                                                 &fullFlattenedArrayCellLocalProcIndexIdMapParent,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &outputQuadData,
    bool resizeOutputVec) ; //override;

  template
  void
    TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::HOST>
      ::interpolateMesh2DataToMesh1QuadPoints(
    const distributedCPUVec<dataTypes::number> &inputVec,
    const unsigned int               numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::HOST> &mapVecToCells,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &            outputQuadData,
    bool resizeOutputVec) ; //override;

  template class TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::HOST>;

#ifdef DFTFE_WITH_DEVICE

    template
  void
    TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::DEVICE>
      ::interpolateMesh1DataToMesh2QuadPoints(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>> &
        BLASWrapperPtr,
    const dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                              dftfe::utils::MemorySpace::DEVICE> &inputVec,
    const unsigned int                    numberOfVectors,
    const dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::DEVICE> &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
    dftfe::utils::MemoryStorage<dataTypes::number, dftfe::utils::MemorySpace::DEVICE> &outputQuadData,
    bool resizeOutputVec); // override;

  template
  void
  TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::DEVICE>
    ::interpolateMesh2DataToMesh1QuadPoints(
        const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>> &
          BLASWrapperPtr,
    const dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                              dftfe::utils::MemorySpace::DEVICE> &inputVec,
    const unsigned int                    numberOfVectors,
    const dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::DEVICE> &fullFlattenedArrayCellLocalProcIndexIdMapMesh1,
    dftfe::utils::MemoryStorage<dataTypes::number, dftfe::utils::MemorySpace::DEVICE> &                 outputQuadData,
    bool resizeOutputVec); // override;

  template
  void
    TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::DEVICE>
      ::interpolateMesh1DataToMesh2QuadPoints(
    const distributedCPUVec<dataTypes::number> &inputVec,
    const unsigned int numberOfVectors,
    const dftfe::utils::MemoryStorage<dealii::types::global_dof_index, dftfe::utils::MemorySpace::HOST>
                                                                 &fullFlattenedArrayCellLocalProcIndexIdMapParent,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &outputQuadData,
    bool resizeOutputVec) ; //override;

  template
  void
    TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::DEVICE>
      ::interpolateMesh2DataToMesh1QuadPoints(
    const distributedCPUVec<dataTypes::number> &inputVec,
    const unsigned int               numberOfVectors,
      const dftfe::utils::MemoryStorage<dftfe::global_size_type, dftfe::utils::MemorySpace::HOST> &mapVecToCells,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &            outputQuadData,
    bool resizeOutputVec) ; //override;

  template class TransferDataBetweenMeshesIncompatiblePartitioning<dftfe::utils::MemorySpace::DEVICE>;

#endif 


}
