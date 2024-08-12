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
// @author Arghadwip Paul, Bikash Kanungo 
//

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <iostream>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/special_functions/factorials.hpp>

#include "Exceptions.h"
#include "SlaterBasis.h"

namespace dftfe
{

	// local namespace
	namespace 
	{
    int factorial(int n)
    {
      if(n==0) 
        return 1;
      else 
        return n*factorial(n-1);
    }

		void
			convertCartesianToSpherical(const std::vector<double> & x, 
                                  double & r, 
                                  double & theta, 
                                  double & phi, 
                                  const double angleTol)
			{
				r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
				if(r == 0)
				{
					theta = 0.0;
					phi = 0.0;
				}

				else
				{
					theta = acos(x[2]/r);
					//
					// check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
					// If yes, assign phi = 0.0.
					// NOTE: In case theta = 0 or PI, phi is undetermined. The actual value 
					// of phi doesn't matter in computing the function value or 
					// its gradient. We assign phi = 0.0 here just as a dummy value
					//
					if(fabs(theta - 0.0) >= angleTol && fabs(theta - M_PI) >= angleTol)
						phi = atan2(x[1],x[0]);

					else
						phi = 0.0;
				}
			}

		std::unordered_map<std::string, std::string>
			readAtomToSlaterBasisName(const std::string &fileName)
			{
				std::unordered_map<std::string, std::string> atomToSlaterBasisName;
				std::ifstream                                file(fileName);
				if (file.is_open())
				{
					std::string line;
					int         lineNumber = 0;
					while (std::getline(file, line))
					{
						lineNumber++;
						std::istringstream iss(line);
						std::string        atomSymbol;
						std::string        slaterBasisName;

						if (iss >> atomSymbol >> slaterBasisName)
						{
							std::string extra;
							std::string msg = "Error: More than two entries in line " +
										std::to_string(lineNumber) + "in" + fileName;
              if(iss >> extra)
                utils::throwException(true, msg);
							
              atomToSlaterBasisName[atomSymbol] = slaterBasisName;
						}
						else
						{
              std::string msg = "Error: Invalid format in line " +
									std::to_string(lineNumber) + "in" +
									fileName;
              utils::throwException(true, msg);
						}
					}
					file.close();
				}
				else
				{
          std::string msg = "Unable to open the file." + fileName;
          utils::throwException(true, msg);
				}
				return atomToSlaterBasisName;
			}
  
    void
    getSlaterPrimitivesFromBasisFile(
        const std::string &atomSymbol,
        const std::string &basisName,
        std::unordered_map<std::string, std::vector<SlaterPrimitive *>> & atomToSlaterPrimitivePtr)
    {
      /*
       * Written in a format that ignores the first line
       */
      std::ifstream file(basisName);
      if (file.is_open())
      {
        std::string                          line;
        std::unordered_map<std::string, int> lStringToIntMap = {
          {"S", 0}, {"P", 1}, {"D", 2}, {"F", 3}, {"G", 4}, {"H", 5}};
        // First line - Ignore
        std::getline(file, line);
        std::istringstream iss(line);
        if (iss.fail())
        {
          std::string msg = "Error reading line in file: " +
            basisName;
          utils::throwException(true, msg);
        }
        std::string atomType, extra;
        iss >> atomType;
        if (iss >> extra)
        {
          std::string msg = "Error: More than one entry in line 1 in" +
            basisName;
          utils::throwException(true, msg);
        }
        // Second Line Onwards
        while (std::getline(file, line))
        {
          std::istringstream iss(line);
          if (iss.fail())
          {
            std::string msg = "Error reading line in file: " +
              basisName;
            utils::throwException(true, msg);
          }

          std::string nlString;
          double      alpha;
          iss >> nlString >> alpha;
          if (iss >> extra)
          {
            std::string msg = "Error: More than two entries in a line in" + basisName;
            utils::throwException(true, msg);
          }

          char lChar = nlString.back();

          int n = std::stoi(nlString.substr(0, nlString.size() - 1));
          // normalization constant for the radial part of Slater function
          const double term1 = pow(2.0*alpha, n + 1.0/2.0);
          const double term2 = pow(factorial(2*n), 1.0/2.0);
          const double normConst = term1/term2;

          int l;
          try
          {
            l = lStringToIntMap.at(std::string(1, lChar));
          }
          catch (const std::out_of_range &e)
          {
            std::string msg = "Character doesn't exist in the lStringToIntMap in "
              "SlaterBasis.cpp: " + std::string(1, lChar);
            utils::throwException(true, msg);
          }
          std::vector<int> mList;
          if (l == 1)
          {
            // Special ordering for p orbitals to be compatible with quantum chemistry codes like QChem
            mList = {1, -1, 0}; 
          }
          else
          {
            for (int m = -l; m <= l; ++m)
            {
              mList.push_back(m);
            }
          }
          for (int m : mList)
          {
            SlaterPrimitive *sp = new SlaterPrimitive{n, l, m, alpha, normConst};
            atomToSlaterPrimitivePtr[atomSymbol].push_back(sp);
          }
        }
      }
      else
      {
        std::string msg = "Unable to open file: " + basisName;
        utils::throwException(true, msg);
      }
    }

		inline double Dm(const int m)
		{
			if(m == 0)
				return 1.0/sqrt(2*M_PI);
			else  
				return 1.0/sqrt(M_PI);
		}

		inline double Clm(const int l, const int m)
		{
			return sqrt(((2.0*l + 1)*boost::math::factorial<double>(l-m))/(2.0*boost::math::factorial<double>(l+m)));
		}

		double Qm(const int m, const double phi)
		{
			if(m > 0)
				return cos(m*phi);
			else if(m == 0)
				return 1.0;
			else //if(m < 0)
				return sin(std::abs(m)*phi);
		}

		double dQmDPhi(const int m, const double phi)
		{
			if(m > 0)
				return -m*sin(m*phi);
			else if(m == 0)
				return 0.0;
			else //(m < 0)
				return std::abs(m)*cos(std::abs(m)*phi);
		}

		double Plm(const int l, const int m, const double x)
		{
			if(std::abs(m) > l)
				return 0.0;
			else
				//
				// NOTE: Multiplies by {-1}^m to remove the 
				// implicit Condon-Shortley factor in the associated legendre 
				// polynomial implementation of boost
				// This is done to be consistent with the QChem's implementation
				return pow(-1.0,m)*boost::math::legendre_p(l, m, x);
		}

		double dPlmDTheta(const int l, const int m, const double theta)
		{
			const double cosTheta = cos(theta);
			if(std::abs(m) > l)
				return 0.0;

			else if(l == 0)
				return 0.0;

			else if(m < 0)
			{
				const int modM = std::abs(m);
				const double factor = pow(-1,m)*boost::math::factorial<double>(l-modM)/boost::math::factorial<double>(l+modM);
				return factor*dPlmDTheta(l, modM, theta);
			}

			else if(m == 0)
			{

				return -1.0*Plm(l,1,cosTheta);
			}

			else if(m == l)
				return l*Plm(l,l-1, cosTheta);

			else
			{
				const double term1  = (l+m)*(l-m+1)*Plm(l, m-1, cosTheta);
				const double term2 = Plm(l, m+1, cosTheta);
				return 0.5*(term1-term2);
			}
		}


		inline double d2PlmDTheta2(const int l, const int m, const double theta)
		{
			const double cosTheta = cos(theta);
			if(std::abs(m) > l)
				return 0.0;

			else if(l == 0)
				return 0.0;

			else if(m < 0)
			{
				const int modM = std::abs(m);
				const double factor = pow(-1,m)*boost::math::factorial<double>(l-modM)/boost::math::factorial<double>(l+modM);
				return factor*d2PlmDTheta2(l, modM, theta);
			}

			else if(m == 0)
				return -1.0*dPlmDTheta(l, 1, theta);
			
			else if(m == l)
				return l*dPlmDTheta(l,l-1, theta);
			
			else
			{
				double term1 = (l+m)*(l-m+1)*dPlmDTheta(l, m-1, theta);
				double term2 = dPlmDTheta(l, m+1, theta);
				return 0.5*(term1-term2);
			}
		}

		inline double slaterRadialPart(const double r, const int n, const double alpha)
		{
			if(n==1)
				return exp(-alpha*r);
			else
				return pow(r,n-1)*exp(-alpha*r);
		}

		inline double slaterRadialPartDerivative(const double r, 
                                             const double alpha, 
                                             const int n, 
                                             const int derOrder)
		{
			if(derOrder == 0 && n >= 1)
				return slaterRadialPart(r, n, alpha); 
			else if(derOrder == 0 && n < 1)
				return 0.0;
			else
				return (n-1)*slaterRadialPartDerivative(r,alpha,n-1,derOrder-1) - alpha*slaterRadialPartDerivative(r,alpha,n,derOrder-1);
		}

    double getSlaterValue(const std::vector<double> & x, 
                     const int n, 
                     const int l, 
                     const int m, 
                     const double alpha, 
                     const double angleTol)
		{
			double r, theta, phi;
			convertCartesianToSpherical(x, r, theta, phi, angleTol);
			const int modM = std::abs(m);
			const double C = Clm(l,modM)*Dm(m);
			const double cosTheta = cos(theta);
			const double R = slaterRadialPart(r, n, alpha);
			const double P = Plm(l,modM,cosTheta);
			const double Q = Qm(m,phi);
			const double returnValue = C*R*P*Q;
			return returnValue;
		}

		std::vector<double> getSlaterGradientAtOrigin(const int n,
				const int l,
				const int m,
				const double alpha)
		{
			std::vector<double> returnValue(3);
			const int modM = std::abs(m);
			const double C = Clm(l,modM)*Dm(m);
			if(n==1)
			{
				std::string message("Gradient of slater orbital at atomic position is undefined for n=1");
        utils::throwException(true, message);

			}

			if(n==2)
			{

				if(l==0)
				{
					std::string message("Gradient of slater orbital at atomic position is undefined for n=2 and l=0");
          utils::throwException(true, message);
				}

				if(l==1)
				{

					if(m==-1)
					{
						returnValue[0] = 0.0;
						returnValue[1] = C;
						returnValue[2] = 0.0;
					}

					if(m==0)
					{
						returnValue[0] = 0.0;
						returnValue[1] = 0.0;
						returnValue[2] = C;
					}

					if(m==1)
					{
						returnValue[0] = C;
						returnValue[1] = 0.0;
						returnValue[2] = 0.0;
					}
				}

			}

			else
			{
				returnValue[0] = 0.0;
				returnValue[1] = 0.0;
				returnValue[2] = 0.0;
			}

			return returnValue;
		}

		std::vector<double> getSlaterGradientAtPoles(const double r,
				const double theta,
				const int n,
				const int l,
				const int m,
				const double alpha,
        const double angleTol)
		{
			const double R = slaterRadialPart(r,n,alpha);
			const double dRDr = slaterRadialPartDerivative(r, alpha, n, 1);
			const int modM = std::abs(m);
			const double C = Clm(l,modM)*Dm(m);
			std::vector<double> returnValue(3);
			if(std::fabs(theta-0.0) < angleTol)
			{
				if(m == 0)
				{
					returnValue[0] = 0.0;
					returnValue[1] = 0.0;
					returnValue[2] = C*dRDr;
				}

				else if(m == 1)
				{
					returnValue[0] = C*(R/r)*l*(l+1)/2.0;
					returnValue[1] = 0.0;
					returnValue[2] = 0.0;
				}

				else if(m == -1)
				{
					returnValue[0] = 0.0;
					returnValue[1] = C*(R/r)*l*(l+1)/2.0;
					returnValue[2] = 0.0;
				}

				else
				{
					returnValue[0] = 0.0;
					returnValue[1] = 0.0;
					returnValue[2] = 0.0;
				}
			}

			else // the other possibility is std::fabs(theta-M_PI) < angleTol 
			{
				if(m == 0)
				{
					returnValue[0] = 0.0;
					returnValue[1] = 0.0;
					returnValue[2] = C*dRDr*pow(-1,l+1);
				}

				else if(m == 1)
				{
					returnValue[0] = C*(R/r)*l*(l+1)/2.0*pow(-1,l+1);
					returnValue[1] = 0.0;
					returnValue[2] = 0.0;
				}

				else if(m == -1)
				{
					returnValue[0] = 0.0;
					returnValue[1] = C*(R/r)*l*(l+1)/2.0*pow(-1,l+1);
					returnValue[2] = 0.0;
				}

				else
				{
					returnValue[0] = 0.0;
					returnValue[1] = 0.0;
					returnValue[2] = 0.0;
				}
			}

			return returnValue;
		}

		std::vector<double> getSlaterGradient(const std::vector<double> & x,
				const int n,
				const int l,
				const int m,
				const double alpha,
        const double rTol,
        const double angleTol)
		{
			double r, theta, phi;
			convertCartesianToSpherical(x, r, theta, phi, angleTol);
			const int modM = std::abs(m);
			const double C = Clm(l,modM)*Dm(m);

			std::vector<double> returnValue(3);
			if(r < rTol)
			{
				returnValue = getSlaterGradientAtOrigin(n,l,m,alpha);
			}

			else if(std::fabs(theta-0.0) < angleTol|| std::fabs(theta-M_PI) < angleTol)
			{
				returnValue = getSlaterGradientAtPoles(r, theta, n, l, m, alpha, angleTol);	
			}

			else
			{
				const double R = slaterRadialPart(r,n,alpha);
				const double dRDr = slaterRadialPartDerivative(r, alpha, n, 1);
				const double cosTheta = cos(theta);
				const double P = Plm(l, modM, cosTheta);
				const double dPDTheta = dPlmDTheta(l, modM, theta);
				const double Q = Qm(m,phi);
				const double dQDPhi = dQmDPhi(m,phi);      
				double jacobianInverse[3][3];
				jacobianInverse[0][0] = sin(theta)*cos(phi); jacobianInverse[0][1] = cos(theta)*cos(phi)/r; jacobianInverse[0][2] = -1.0*sin(phi)/(r*sin(theta));
				jacobianInverse[1][0] = sin(theta)*sin(phi); jacobianInverse[1][1] = cos(theta)*sin(phi)/r; jacobianInverse[1][2] = cos(phi)/(r*sin(theta));
				jacobianInverse[2][0] = cos(theta); 	   jacobianInverse[2][1] = -1.0*sin(theta)/r;     jacobianInverse[2][2] = 0.0;

				double partialDerivatives[3];
				partialDerivatives[0] = dRDr*P*Q;
				partialDerivatives[1] = R*dPDTheta*Q;
				partialDerivatives[2] = R*P*dQDPhi;
				for(unsigned int i = 0; i < 3; ++i)
				{
					returnValue[i] = C*(jacobianInverse[i][0]*partialDerivatives[0]+
							jacobianInverse[i][1]*partialDerivatives[1]+
							jacobianInverse[i][2]*partialDerivatives[2]);
				}

			}

			return returnValue;
		}

		double getSlaterLaplacianAtOrigin(const int n,
				const int l,
				const int m,
				const double alpha)
		{
			if(n==1 || n==2)
			{
				std::string message("Laplacian of slater function is undefined at atomic position for n=1 and n=2.");
        utils::throwException(true, message);
			}

			else if(n==3)
			{
				if(l==0)
				{
					const int modM = std::abs(m);
					const double C = Clm(l,modM)*Dm(m);
					return 6.0*C;
				}

				else if(l==1)
				{

					std::string message("Laplacian of slater function is undefined at atomic position for n=3, l=1.");
          utils::throwException(true, message);
					return 0.0;
				}

				else if(l==2)
				{
					return 0.0;
				}

				else // l >= 3
					return 0.0;
			}

			else
			{
				return 0.0;
			}
		}

		double getSlaterLaplacianAtPoles(const double r,
				const double theta,
				const int n,
				const int l,
				const int m,
				const double alpha,
        const double angleTol)
		{

			double returnValue = 0.0;
			if(m == 0)
			{
				const int modM = std::abs(m);
				const double C = Clm(l,modM)*Dm(m);
				const double R = slaterRadialPart(r, n , alpha);
				const double dRdr = slaterRadialPartDerivative(r, alpha, n, 1);
				const double d2Rdr2 = slaterRadialPartDerivative(r, alpha, n, 2);
				if(std::fabs(theta-0.0) < angleTol)
				{
					const double term1 = C * (2.0*dRdr/r + d2Rdr2);
					const double term2 = C * (R/(r*r)) * (-l*(l+1));
					returnValue = term1 + term2;
				}

				else // the other possibility is std::fabs(theta-M_PI) < angleTol 
				{
					const double term1 = C * (2.0*dRdr/r + d2Rdr2) * pow(-1,l);
					const double term2 = C * (R/(r*r)) * (-l*(l+1)) * pow(-1,l);
					returnValue = term1 + term2;
				}
			}

			else
				returnValue = 0.0;

			return returnValue;
		}

		double getSlaterLaplacian(const std::vector<double> & x,
				const int n,
				const int l,
				const int m,
				const double alpha,
        const double rTol,
        const double angleTol)
		{
			double r, theta, phi;
			convertCartesianToSpherical(x, r, theta, phi, angleTol);
			double returnValue = 0.0;
			if(r < rTol)
			{
				returnValue = getSlaterLaplacianAtOrigin(n, l, m, alpha);
			}

			else if(std::fabs(theta-0.0) < angleTol || std::fabs(theta-M_PI) < angleTol)
			{
				returnValue = getSlaterLaplacianAtPoles(r, theta, n, l, m, alpha, angleTol);
			}

			else
			{
				const int modM = std::abs(m);
				const double C = Clm(l,modM)*Dm(m);
				const double cosTheta = cos(theta);
				const double sinTheta = sin(theta);
				const double R = slaterRadialPart(r, n , alpha);
				const double dRdr = slaterRadialPartDerivative(r, alpha, n, 1);
				const double d2Rdr2 = slaterRadialPartDerivative(r, alpha, n, 2);
				const double P = Plm(l,modM,cosTheta);
				const double Q = Qm(m,phi);
				const double term1 = C*P*Q*(2.0*dRdr/r + d2Rdr2);
				const double a = dPlmDTheta(l, modM, theta);
				const double b = d2PlmDTheta2(l, modM, theta);
				const double term2 = C * (R/(r*r)) * Q * 
					((cosTheta/sinTheta)*a + b);
				const double term3 = -C*m*m*(R/(r*r))*Q*P/(sinTheta*sinTheta);
				returnValue = term1 + term2 + term3;
			}

			return returnValue;
		}

		

	} // end of local namespace

	SlaterBasis::SlaterBasis(const double rTol /*=1e-10*/,
                                 const double angleTol /*=1e-10*/):
    d_rTol(rTol),
    d_angleTol(angleTol)
	{
		
	}

	SlaterBasis::~SlaterBasis()
	{
		// deallocate SlaterPrimitive pointers stored in the map
		for (auto &pair : d_atomToSlaterPrimitivePtr)
		{
			std::vector<SlaterPrimitive *> &primitives = pair.second;
			for (SlaterPrimitive *sp : primitives)
			{
				if (sp != nullptr)
				{
					delete sp;
				}
			}
			primitives.clear();
		}
	}


	void
		SlaterBasis::constructBasisSet(
				const std::vector<std::pair<std::string, std::vector<double>>> &atomCoords,
				const std::string &auxBasisFileName)
		{
			d_atomSymbolsAndCoords = atomCoords;
			unsigned int natoms = d_atomSymbolsAndCoords.size();
			const auto atomToSlaterBasisName =
				readAtomToSlaterBasisName(auxBasisFileName);

			for (const auto &pair : atomToSlaterBasisName)
			{
				const std::string &atomSymbol = pair.first;
				const std::string &basisName  = pair.second;
        d_atomToSlaterPrimitivePtr.clear();
				getSlaterPrimitivesFromBasisFile(atomSymbol, basisName, d_atomToSlaterPrimitivePtr);
			}

			for (unsigned int i = 0; i < natoms; ++i)
			{
				const std::string &        atomSymbol = d_atomSymbolsAndCoords[i].first;
        const std::vector<double> & atomCenter = d_atomSymbolsAndCoords[i].second;
        unsigned int nprimitives = d_atomToSlaterPrimitivePtr[atomSymbol].size();
        for (unsigned int j = 0; j < nprimitives; ++j)
        {
          SlaterBasisInfo info;
          info.symbol = &atomSymbol;
          info.center = atomCenter.data();
          info.sp = d_atomToSlaterPrimitivePtr[atomSymbol][j];
          d_slaterBasisInfo.push_back(info);
        }
      }
    }

  const std::vector<SlaterBasisInfo> &
    SlaterBasis::getSlaterBasisInfo() const
    {
      return d_slaterBasisInfo;
    }

  int
    SlaterBasis::getSlaterBasisSize() const
    {
      return static_cast<int>(d_slaterBasisInfo.size());
    }


  double SlaterBasis::getBasisValue(const unsigned int basisId, 
      const std::vector<double> & x) const
  {
    const SlaterBasisInfo & info = d_slaterBasisInfo[basisId];
    const double * x0 = info.center;
    const SlaterPrimitive * sp = info.sp;
    const double alpha = sp->alpha;
    const int n = sp->n;
    const int l = sp->l;
    const int m = sp->m;
    const double normConst = sp->normConst;
    std::vector<double> dx(3);
    for(unsigned int i = 0; i < 3; ++i)
      dx[i] = x[i] - x0[i];

    double returnValue = normConst*getSlaterValue(dx, n, l, m, alpha, d_angleTol);
    return returnValue;
  }

  std::vector<double> 
    SlaterBasis::getBasisGradient(const unsigned int basisId, 
      const std::vector<double> & x) const
  {
    const SlaterBasisInfo & info = d_slaterBasisInfo[basisId];
    const double * x0 = info.center;
    const SlaterPrimitive * sp = info.sp;
    const double alpha = sp->alpha;
    const int n = sp->n;
    const int l = sp->l;
    const int m = sp->m;
    const double normConst = sp->normConst;
    std::vector<double> dx(3);
    for(unsigned int i = 0; i < 3; ++i)
      dx[i] = x[i] - x0[i];

    std::vector<double> returnValue = getSlaterGradient(dx, n, l, m, alpha, d_rTol, d_angleTol);
    for(unsigned int i = 0; i < returnValue.size(); ++i)
      returnValue[i] *= normConst;

    return returnValue;
  }

  double SlaterBasis::getBasisLaplacian(const unsigned int basisId, 
      const std::vector<double> & x) const
  {
    const SlaterBasisInfo & info = d_slaterBasisInfo[basisId];
    const double * x0 = info.center;
    const SlaterPrimitive * sp = info.sp;
    const double alpha = sp->alpha;
    const int n = sp->n;
    const int l = sp->l;
    const int m = sp->m;
    const double normConst = sp->normConst;
    std::vector<double> dx(3);
    for(unsigned int i = 0; i < 3; ++i)
      dx[i] = x[i] - x0[i];

    double returnValue = normConst*getSlaterLaplacian(dx, n, l, m, alpha, d_rTol, d_angleTol);
    return returnValue;
  }
} // namespace dftfe
