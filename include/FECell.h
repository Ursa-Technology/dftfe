/******************************************************************************
* Copyright (c) 2021.                                                        *
* The Regents of the University of Michigan and DFT-EFE developers.          *
*                                                                            *
* This file is part of the DFT-EFE code.                                     *
*                                                                            *
* DFT-EFE is free software: you can redistribute it and/or modify            *
*   it under the terms of the Lesser GNU General Public License as           *
*   published by the Free Software Foundation, either version 3 of           *
*   the License, or (at your option) any later version.                      *
*                                                                            *
* DFT-EFE is distributed in the hope that it will be useful, but             *
*   WITHOUT ANY WARRANTY; without even the implied warranty                  *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
*   See the Lesser GNU General Public License for more details.              *
*                                                                            *
* You should have received a copy of the GNU Lesser General Public           *
*   License at the top level of DFT-EFE distribution.  If not, see           *
*   <https://www.gnu.org/licenses/>.                                         *
******************************************************************************/

/*
* @author Vishal Subramanian, Bikash Kanungo
 */

#ifndef DFTFE_FECELL_H
#define DFTFE_FECELL_H

#include "Cell.h"

namespace dftfe
{
  namespace utils
  {
    template <size_type dim>
    class FECell : public Cell<dim>
    {
    public :

      using DealiiFECellIterator =
        typename dealii::DoFHandler<dim>::active_cell_iterator;

      FECell( typename dealii::DoFHandler<dim>::active_cell_iterator dealiiFECellIter);

      void reinit(DealiiFECellIterator dealiiFECellIter);

      void
      getVertices(std::vector<std::vector<double>> &points) const override;

      void
      getVertex(size_type i, std::vector<double> &point) const override;

      std::pair<std::vector<double>, std::vector<double>> getBoundingBox() const override;

      bool isPointInside(const std::vector<double> &point,
                    const double tol) const override;

      std::vector<double> getParametricPoint (const std::vector<double> &realPoint) const override;

      DealiiFECellIterator &
      getDealiiFECellIter();

    private:

      std::vector<double> d_lowerLeft;
      std::vector<double> d_upperRight;

      DealiiFECellIterator d_dealiiFECellIter;

      dealii::MappingQ1<dim,dim> d_mappingQ1;

    }; // end of class FECell
  } // end of namespace utils
} // end of namespace dftfe

#include "../utils/FECell.t.cc"

//#include "../utils/FECell.t.cc"
#endif // DFTFE_FECELL_H
