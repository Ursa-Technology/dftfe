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
// @author Bikash Kanungo 
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
#include "GaussianBasis.h"
#include "StringOperations.h"

		/**
		 * For the definition of the associated Legendre polynomials i.e. Plm() and their derivatives 
		 * (as used for evaluating the real form of spherical harmonics and their derivatives) refer:
		 * @article{bosch2000computation,
		 *  	   title={On the computation of derivatives of Legendre functions},
		 *     	   author={Bosch, W},
		 *         journal={Physics and Chemistry of the Earth, Part A: Solid Earth and Geodesy},
		 *         volume={25},
		 *         number={9-11},
		 *         pages={655--659},
		 *         year={2000},
		 *         publisher={Elsevier}
		 *        }
		 */
namespace dftfe
{
	  // local namespace
		namespace
		{
			double doubleFactorial(int n)
			{
				if (n == 0 || n==-1)
					return 1.0;
				return n*doubleFactorial(n-2);
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

			std::vector<double> getNormConsts(const std::vector<double> & alpha, const int l)
			{
				int L = alpha.size();
        std::vector<double> returnValue(L);
				for(unsigned int i = 0; i < L; ++i)
				{
					const double term1 = doubleFactorial(2*l+1)*sqrt(M_PI);
					const double term2 = pow(2.0,2*l+3.5)*pow(alpha[i],l+1.5);
					const double overlapIntegral = term1/term2;
					returnValue[i] = 1.0/sqrt(overlapIntegral);
				}
				return returnValue;
			}

			double getDistance(const double * x, const double * y)
			{
				double r = 0.0;
				for(unsigned int i = 0; i < 3; ++i)
					r += pow(x[i]-y[i],2.0);
				return sqrt(r);
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
				assert(m >= 0);
				assert(std::abs(m) <= l);
				//const int modM = std::abs(m);
				return sqrt(((2.0*l + 1)*boost::math::factorial<double>(l-m))/(2.0*boost::math::factorial<double>(l+m)));
			}

			double Qm(const int m, const double phi)
			{
				if(m > 0)
					return cos(m*phi);
				else if(m == 0)
					return 1.0;
				else //(m < 0)
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

			double d2PlmDTheta2(const int l, const int m, const double theta)
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

			double gaussianRadialPart(const double r, const int l, const double alpha)
			{
				return pow(r,l)*exp(-alpha*r*r);
			}


			double gaussianRadialPartDerivative(const double r, const double alpha, const int l, const int derOrder)
			{
				if(derOrder == 0 && l >= 0)
					return pow(r,l)*exp(-alpha*r*r);
				else if(derOrder == 0 && l < 0)
					return 0.0;
				else
					return l*gaussianRadialPartDerivative(r,alpha,l-1,derOrder-1) - 2*alpha*gaussianRadialPartDerivative(r,alpha,l+1,derOrder-1);
			}

			double getLimitingValueLaplacian(const int l, const int m, const double theta)
			{
				double returnValue = 0.0;
				if(std::fabs(theta-0.0) < POLAR_ANGLE_TOL)
				{
					if(m == 0)
						returnValue = -0.5*l*(l+1);
					if(m == 2)
						returnValue = 0.25*(l-1)*l*(l+1)*(l+2);
				}

				if(std::fabs(theta-M_PI) < POLAR_ANGLE_TOL)
				{

					if(m == 0)
						returnValue = -0.5*l*(l+1)*pow(-1.0,l);
					if(m == 2)
						returnValue = 0.25*(l-1)*l*(l+1)*(l+2)*pow(-1.0,l);;
				}

				return returnValue;
			}


			double getContractedGaussianValue(const ContractedGaussian * cg, 
          const std::vector<double> & x, 
          const double angleTol)
			{
				const int nC = cg->nC;
				const int l = cg->l;
				const int m = cg->m;
				double r, theta, phi;
				convertCartesianToSpherical(x, r, theta, phi, angleTol);
				double returnValue = 0.0;
				for(unsigned int i = 0; i < nC; ++i)
				{
					const double alphaVal = cg->alpha[i];
					const double cVal = cg->c[i];
					const double norm = cg->normConsts[i];
					returnValue += cVal*norm*gaussianRadialPart(r, l, alphaVal);
				}

				const int modM = std::abs(m);
				const double C = Clm(l,modM)*Dm(m);
				const double cosTheta = cos(theta);
				const double P = Plm(l,modM,cosTheta);
				const double Q = Qm(m,phi);
				returnValue *= C*P*Q;
				return returnValue;
			}

			std::vector<double> getContractedGaussianGradient(
          const ContractedGaussian * cg, 
          const std::vector<double> & x, 
          double rTol, 
          double angleTol)
			{
				const int nC = info->nC;
				const int l = cg->l;
				const int m = cg->m;
				double r, theta, phi;
				convertCartesianToSpherical(x, r, theta, phi, angleTol);
				std::vector<double> returnValue(3);
				double R = 0.0;
        double dRdr = 0.0;
        double T = 0.0;
        for(unsigned int i = 0; i < nC; ++i)
				{
					const double alphaVal = b->alpha[i];
					const double cVal = b->c[i];
					const double norm = b->normConsts[i];
					R += cVal*norm*gaussianRadialPart(r, l, alphaVal);
				  dRdr += cVal*norm*gaussianradialPartDerivative(r, alphaVal, l, 1);
					T += cVal*norm;
        }
				
        const int modM = std::abs(m);
				const double C = Clm(l,modM)*Dm(m);
				const double cosTheta = cos(theta);
				const double P = Plm(l, modM, cosTheta);
				const double dPDTheta = dPlmDTheta(l, modM, theta);
				const double Q = Qm(m,phi);
				const double dQDPhi = dQmDPhi(m,phi);      
				if(r < rTol)
				{
					if(l==1)
					{
						if(m==-1)
						{
							returnValue[0] = 0.0;
							returnValue[1] = T;
							returnValue[2] = 0.0;
						}

						if(m==0)
						{
							returnValue[0] = 0.0;
							returnValue[1] = 0.0;
							returnValue[2] = T;
						}

						if(m==1)
						{
							returnValue[0] = T;
							returnValue[1] = 0.0;
							returnValue[2] = 0.0;
						}

					}

					else
					{
						returnValue[0] = 0.0;
						returnValue[1] = 0.0;
						returnValue[2] = 0.0;
					}
				}

				else if(std::fabs(theta-0.0) < angleTol)
				{
					if(m == 0)
					{
						returnValue[0] = 0.0;
						returnValue[1] = 0.0;
						returnValue[2] = dRDr*P*cosTheta;
					}

					else if(m == 1)
					{
						returnValue[0] = (R/r)*l*(l+1)/2.0;
						returnValue[1] = 0.0;
						returnValue[2] = 0.0;
					}

					else if(m == -1)
					{
						returnValue[0] = 0.0;
						returnValue[1] = (R/r)*l*(l+1)/2.0;
						returnValue[2] = 0.0;
					}

					else
					{
						returnValue[0] = 0.0;
						returnValue[1] = 0.0;
						returnValue[2] = 0.0;
					}
				}

				else if(std::fabs(theta-M_PI) < angleTol)
				{
					if(m == 0)
					{
						returnValue[0] = 0.0;
						returnValue[1] = 0.0;
						returnValue[2] = dRDr*P*cosTheta;
					}

					else if(m == 1)
					{
						returnValue[0] = (R/r)*l*(l+1)/2.0*pow(-1,l+1);
						returnValue[1] = 0.0;
						returnValue[2] = 0.0;
					}

					else if(m == -1)
					{
						returnValue[0] = 0.0;
						returnValue[1] = (R/r)*l*(l+1)/2.0*pow(-1,l+1);
						returnValue[2] = 0.0;
					}

					else
					{
						returnValue[0] = 0.0;
						returnValue[1] = 0.0;
						returnValue[2] = 0.0;
					}
				}

				else
				{
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
						returnValue[i] = (jacobianInverse[i][0]*partialDerivatives[0]+
								jacobianInverse[i][1]*partialDerivatives[1]+
								jacobianInverse[i][2]*partialDerivatives[2]);
					}
				}

        for(unsigned int i = 0; i < 3; ++i)
          returnValue[i] *= C;

				return returnValue;
			}


			double getContractedGaussianLaplacian(const ContractedGaussian * cg, const std::vector<double> & x, double rTol, double angleTol)
			{
				const int nC = cg->nC;
				const int l = cg->l;
				const int m = cg->m;
        double returnValue = 0.0;
				double r, theta, phi;
				convertCartesianToSpherical(x, r, theta, phi, angleTol);
				const int modM = std::abs(m);
				const double C = Clm(l,modM)*Dm(m);
				const double P = Plm(l,modM,cosTheta);
				const double Q = Qm(m,phi);
				double R = 0.0;
        double dRdr = 0.0;
        double dRdr2 = 0.0;
        double S = 0.0;
        for(unsigned int i = 0; i < nC; ++i)
				{
					const double alphaVal = b->alpha[i];
					const double cVal = b->c[i];
					const double norm = b->normConsts[i];
					R += cVal*norm*gaussianRadialPart(r, l, alphaVal);
				  dRdr += cVal*norm*gaussianradialPartDerivative(r, alphaVal, l, 1);
				  dRdr2 += cVal*norm*gaussianradialPartDerivative(r, alphaVal, l, 2);
					S += cVal*norm*alphaVal;
        }
				
				if(r < rTol)
				{
					if(l == 0)
						returnValue = -6.0*S;
					else
						returnValue = 0.0;
				}
				
        else
				{
					const double term1 = P*Q*(2.0*dRdr/r + d2Rdr2);
					if(std::fabs(theta-0.0) < angleTol || std::fabs(theta-M_PI) < angleTol)
					{
						const double limitingVal = getLimitingValueLaplacian(l, modM, theta);
						const double term2 = (R/(r*r))*Q*(limitingVal+limitingVal);
						const double term3 = -m*m*(R/(r*r))*Q*(limitingVal/2.0);
						returnValue = (term1 + term2 + term3);
					}

					else
					{
						const double a = dPlmDTheta(l, modM, theta);
						const double b = d2PlmDTheta2(l, modM, theta);
						const double term2 = (R/(r*r)) * Q * 
							((cosTheta/sinTheta)*a + b);
						const double term3 = -m*m*(R/(r*r))*Q*P/(sinTheta*sinTheta);
						returnValue = term1 + term2 + term3;
					}
				}

        returnValue *= C;
				return returnValue;
			}

			void
				getContractedGaussians(const std::string basisFileName
						std::vector<ContractedGaussian *> & contractedGaussians) 
				{
					// string to read line
					std::string readLine;
					std::ifstream readFile(basisFileName);
					if(readFile.fail())
					{
						std::string msg = "Unable to open file " + basisFileName;
						utils::throwException(false, msg);							
					}
					
					// ignore the first line
					std::getline(readFile, readLine);
					while(std::getline(readFile,readLine))
					{
						std::istringstream lineString(readLine);
            std::vector<std::string> words = utils::strOps::split(lineString.str(), "\t ");
            std::string msg = "Unable to read l character(s) and the number of contracted Gaussian in file " + basisFileName;
            utils::throwException(words.size() >= 2, msg);
            
            std::string lChars = words[0];
            // check if it's a valid string
            // i.e., it contains one of the following string:
            // "S", "SP", "SPD", SPDF" ...
            std::size_t pos = lChars.find_first_not_of("SPDFGHspdfgh");
            bool validStr = pos == std::string::npos;
            std::string msg = "Undefined L character(s) for the contracted Gaussian read in file " + basisFileName;
            utils::throwException(validStr, msg);
            const int numLChars = lChars.size();

            // read the number of contracted gaussians
            std::string strNContracted = words[1];
            int nContracted;
            bool isInt = utils::strOps:strToInt(strNContracted,nContracted);
            std::string msg =  "Undefined number of contracted Gaussian in file " + basisFileName;
            utils::throwException(isInt, msg);
            std::vector<double> alpha(nContracted,0.0);
            std::vector<std::vector<double>> c(nContracted, std::vector<double>(numLChars,0.0));
            for(unsigned int i = 0; i < nContracted; ++i)
            {
              if(std::getline(readFile,readLine))
              {
                if(readLine.empty())
                {
                  std::string msg = "Empty line found in Gaussian basis file " + basisFilename;
                  utils::throwException(false, msg);
                }

                std::istringstream lineContracted(readLine);
                std::vector<string> wordsAlphaCoeff = utils::strOps::split(lineContracted.str(), "\t ");
                std::string msg = "Unable to read the exponent and the coefficients of contracted Gaussian in file " + basisFileName;
                utils::throwException(wordsAlphaCoeff.size() >= 1 + numLChars);
                std::string alphaStr = wordsAlphaCoeff[0]
                std::string msg = "Undefined value " + alpaStr + 
                  " read for the Gaussian exponent in file " + basisFileName;
                bool isNumber = utils::strOps::strToDouble(alphaStr, alpha[i]);
                utils::throwException(isNumber, msg);
                for(unsigned int j = 0; j < numLChars; ++j)
                {
                  std::string coeffStr = wordsAlphaCoeff[1+j];
                  std::string msg= "Undefined value " + coeffStr + 
                    " read for the Gaussian coefficient in file " +
                    basisFileName;
                  bool isNumber = utils::strOps::strToDouble(coeffStr, c[i][j]);
                  utils::throwException(isNumber,msg)
                }
              }
              else
              {
                std::string msg = "Undefined row for the contracted Gaussian detected in file" + basisFileName;
                utils::throwException(false,msg)
              }
            }

            for(unsigned int j = 0; j < numLChars; ++j)
            {
              std::vector<int> mList;
              int l = lChars.at(j);
              if (l == 1)
              {
                // Special ordering for p orbitals to be compatible with quantum chemistry codes like QChem
                // In most quantum chemistry codes, eben for spherical Gaussians, for the p-orbitals (l=1),
                // the m-ordering {1, -1, 0} (i.e., px, py, pz) instead of {-1, 0, 1} (i.e., py, pz, px)
                mList = {1, -1, 0}; 
              }
              else
              {
                for (int m = -l; m <= l; ++m)
                {
                  mList.push_back(m);
                }
              }

              for(unsigned int k = 0; k < mList.size(); ++k)
              {
                ContractedGaussian * cg = new ContractedGaussian;
                cg->nG = nContracted;
                cg->l = l;
                cg->m = mList[k];
                cg->alpha = alpha;
                cg->norm = getNormConsts(alpha, l);
                cg->c = c[i];
                std::vector<double> normConsts = getNormConsts(alpha, l);
                atomicContractedGaussians.push_back(cg);
              }
            }
          }

          readFile.close();
        }
    }

    //
    // Constructor
    //
    GaussianBasis::GaussianBasis(double rTol /*=1e-10*/, double angleTol = /*=1e-10*/): 
      d_rTol(rTol),
      d_angleTol(angleTol)
  {

  }

    //
    //Destructor
    //
    GaussianBasis::~GaussianBasis()
    {
      for (auto &pair : d_atomToContractedGaussiansPtr)
      {
        std::vector<ContractedGaussian *> & contractedGaussians = pair.second;
        for (ContractedGaussian *cg : contractedGaussians)
        {
          if (cg != nullptr)
          {
            delete cg;
          }
        }
        contractedGaussians.clear();
      }
    }

    void GaussianBasis::constructBasisSet(
        const std::vector<std::pair<std::string, std::vector<double>>> &atomCoords,
        const std::vector<std::string, std::string> & atomBasisFileName)
    {
      d_atomSymbolsAndCoords = atomCoords;
      unsigned int natoms = d_atomSymbolsAndCoords.size();
      d_atomToContractedGaussiansPtr.clear();
      for (const auto &pair : atomBasisFileName)
      {
        const std::string &atomSymbol = pair.first;
        const std::string &basisFileName  = pair.second;
        d_atomToConstractedGaussiansPtr[atomSymbol] = std::vector<ContractedGaussians *>(0);
        getContractedGaussians(basisFileName, d_atomToContractedGaussiansPtr[atomSymbol]);
      }

      d_gaussianBasisInfo.resize(0);
      for (unsigned int i = 0; i < natoms; ++i)
      {
        const std::string &        atomSymbol = d_atomSymbolsAndCoords[i].first;
        const std::vector<double> & atomCenter = d_atomSymbolsAndCoords[i].second;
        unsigned int n = d_atomToContractedGaussiansPtr[atomSymbol].size();
        for (unsigned int j = 0; j < n; ++j)
        {
          GaussianBasisInfo info;
          info.symbol = &atomSymbol;
          info.center = atomCenter.data();
          info.cg = d_atomToContractedGaussiansPtr[atomSymbol][j];
          d_gaussianBasisInfo.push_back(info);
        }
      }
    }

    
    const std::vector<GaussianBasisInfo> & GaussianBasis::getGaussianBasisInfo() const
    {
      return d_gaussianBasisInfo;
    }

    
    int GaussianBasis::getNumBasis() const
    {
      d_gaussianBasisInfo.size();
    }

    
    double GaussianBasis::getBasisValue(const unsigned int basisId, 
        const std::vector<double> & x) const
    {
      const GaussianBasisInfo & info = d_gaussianBasisInfo[basisId];
      const double * x0 = info.center;
      const ContractedGaussian * cg = info.cg;
      std::vector<double> dx(3);
      for(unsigned int i = 0; i < 3; ++i)
        dx[i] = x[i] - x0[i];

      double returnValue = getContractedGaussianValue(cg, dx, d_rTol, d_angleTol);
      return returnValue;
    }

    
    std::vector<double> GaussianBasis::getBasisGradient(const unsigned int basisId, 
        const std::vector<double> & x) const
    {
      const GaussianBasisInfo & info = d_gaussianBasisInfo[basisId];
      const double * x0 = info.center;
      const ContractedGaussian * cg = info.cg;
      std::vector<double> dx(3);
      for(unsigned int i = 0; i < 3; ++i)
        dx[i] = x[i] - x0[i];

      std::vector<double> returnValue = getContractedGaussianValue(cg, dx, d_rTol, d_angleTol);
      return returnValue;
    }


    double GaussianBasis::getBasisLaplacian(const unsigned int basisId, 
        const std::vector<double> & x) const
    {
      const GaussianBasisInfo & info = d_gaussianBasisInfo[basisId];
      const double * x0 = info.center;
      const ContractedGaussian * cg = info.cg;
      std::vector<double> dx(3);
      for(unsigned int i = 0; i < 3; ++i)
        dx[i] = x[i] - x0[i];

      double returnValue = getContractedGaussianLaplacian(cg, dx, d_rTol, d_angleTol);
      return returnValue;
    }
} // namespace dftfe
