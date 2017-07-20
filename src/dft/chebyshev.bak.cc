#include <complex>
#include<vector>
#ifdef ENABLE_PERIODIC_BC
std::complex<double> dftClass::innerProduct(vectorType &  X,
					    vectorType &  Y)
{

  unsigned int dofs_per_proc = X.local_size()/2; 
  std::vector<double> xReal(dofs_per_proc), xImag(dofs_per_proc);
  std::vector<double> yReal(dofs_per_proc), yImag(dofs_per_proc);
  std::vector<std::complex<double> > xlocal(dofs_per_proc);
  std::vector<std::complex<double> > ylocal(dofs_per_proc);

   
  X.extract_subvector_to(local_dof_indicesReal.begin(), 
  			 local_dof_indicesReal.end(), 
  			 xReal.begin()); 

  X.extract_subvector_to(local_dof_indicesImag.begin(), 
			 local_dof_indicesImag.end(), 
			 xImag.begin());
  
  Y.extract_subvector_to(local_dof_indicesReal.begin(), 
			 local_dof_indicesReal.end(), 
			 yReal.begin()); 

  Y.extract_subvector_to(local_dof_indicesImag.begin(), 
			 local_dof_indicesImag.end(), 
			 yImag.begin());

  for(int i = 0; i < dofs_per_proc; ++i)
    {
      xlocal[i].real(xReal[i]);
      xlocal[i].imag(xImag[i]);
      ylocal[i].real(yReal[i]);
      ylocal[i].imag(yImag[i]);
    }

  int inc = 1;
  int n = dofs_per_proc;

  std::complex<double>  localInnerProduct;

  zdotc_(&localInnerProduct,
	 &n,
	 &xlocal[0],
	 &inc,
	 &ylocal[0],
	 &inc);

  std::complex<double> returnValue(0.0,0.0);

  MPI_Allreduce(&localInnerProduct,
		&returnValue,
		1,
		MPI_C_DOUBLE_COMPLEX,
		MPI_SUM,
		mpi_communicator);

  return returnValue; 
}

void dftClass::alphaTimesXPlusY(std::complex<double>   alpha,
				vectorType           & x,
				vectorType           & y)
{

  //
  //compute y = alpha*x + y
  //

  //
  //extract real and imaginary parts of x and y
  //
  unsigned int dofs_per_proc = x.local_size()/2; 
  std::vector<double> xReal(dofs_per_proc), xImag(dofs_per_proc);
  std::vector<double> yReal(dofs_per_proc), yImag(dofs_per_proc);
  std::vector<std::complex<double> > xlocal(dofs_per_proc);
  std::vector<std::complex<double> > ylocal(dofs_per_proc);

   
  x.extract_subvector_to(local_dof_indicesReal.begin(), 
  			 local_dof_indicesReal.end(), 
  			 xReal.begin()); 

  x.extract_subvector_to(local_dof_indicesImag.begin(), 
			 local_dof_indicesImag.end(), 
			 xImag.begin());
  
  y.extract_subvector_to(local_dof_indicesReal.begin(), 
			 local_dof_indicesReal.end(), 
			 yReal.begin()); 

  y.extract_subvector_to(local_dof_indicesImag.begin(), 
			 local_dof_indicesImag.end(), 
			 yImag.begin());

  for(int i = 0; i < dofs_per_proc; ++i)
    {
      xlocal[i].real(xReal[i]);
      xlocal[i].imag(xImag[i]);
      ylocal[i].real(yReal[i]);
      ylocal[i].imag(yImag[i]);
    }

  int n = dofs_per_proc;int inc = 1;

  //call blas function
  zaxpy_(&n,
	 &alpha,
	 &xlocal[0],
	 &inc,
	 &ylocal[0],
	 &inc);

  //
  //initialize y to zero before copying ylocal values to y
  //
  y = 0.0;
  for(unsigned int i = 0; i < dofs_per_proc; ++i)
    {
      y.local_element(2*i)   = ylocal[i].real();
      y.local_element(2*i+1) = ylocal[i].imag();
    }

  y.update_ghost_values();
}
#endif

//chebyshev solver
void dftClass::chebyshevSolver(){
  computing_timer.enter_section("Chebyshev solve"); 
  //compute upper bound of spectrum
  bUp = upperBound(); 
  char buffer[100];
  sprintf(buffer, "bUp: %18.10e\n", bUp);
  pcout << buffer;
  pcout << "bLow: " << bLow[d_kPointIndex] << std::endl;
  pcout << "a0: " << a0[d_kPointIndex] << std::endl;
  //filter
  for (unsigned int i=0; i<eigenVectors[0].size(); i++){
  sprintf(buffer, "%2u l2: %18.10e     linf: %18.10e \n", i, eigenVectors[0][i]->l2_norm(), eigenVectors[0][i]->linfty_norm());
  pcout << buffer; 
  }
  double t=MPI_Wtime();
  chebyshevFilter(eigenVectors[d_kPointIndex], chebyshevOrder, bLow[d_kPointIndex], bUp, a0[d_kPointIndex]);
  pcout << "Total time for only chebyshev filter: " << (MPI_Wtime()-t)/60.0 << "mins\n";
  for (unsigned int i=0; i<eigenVectors[0].size(); i++){
  sprintf(buffer, "%2u l2: %18.10e     linf: %18.10e \n", i, eigenVectors[0][i]->l2_norm(), eigenVectors[0][i]->linfty_norm());
  pcout << buffer; 
  }
  //Gram Schmidt orthonormalization
  gramSchmidt(eigenVectors[d_kPointIndex]);
  //Rayleigh Ritz step
  rayleighRitz(eigenVectors[d_kPointIndex]);
  pcout << "Total time for chebyshev filter: " << (MPI_Wtime()-t)/60.0 << "mins\n";
  computing_timer.exit_section("Chebyshev solve"); 
}

double dftClass::upperBound(){
  computing_timer.enter_section("Chebyshev upper bound"); 
  unsigned int lanczosIterations=10;
  double beta;

#ifdef ENABLE_PERIODIC_BC
  std::complex<double> alpha;
#else
  double alpha;
#endif

  //generate random vector v
  vChebyshev=0.0;
  std::srand(this_mpi_process);
  const unsigned int local_size = vChebyshev.local_size();
  std::vector<unsigned int> local_dof_indices(local_size);
  vChebyshev.locally_owned_elements().fill_index_vector(local_dof_indices);
  std::vector<double> local_values(local_size, 0.0);
  for (unsigned int i=0; i<local_size; i++) 
    {
      local_values[i]= ((double)std::rand())/((double)RAND_MAX);
    }

  /*for (unsigned int i=0; i<local_size/2; i++) 
    {
      local_values[2*i]= 1.0;//((double)std::rand())/((double)RAND_MAX);
      local_values[2*i+1] = 0.0;
      }*/
  constraintsNoneEigen.distribute_local_to_global(local_values, local_dof_indices, vChebyshev);
  //
  vChebyshev/=vChebyshev.l2_norm();
  vChebyshev.update_ghost_values();
  //
  std::vector<vectorType*> v,f; 
  v.push_back(&vChebyshev);
  f.push_back(&fChebyshev);
  eigen.HX(v,f);

  char buffer2[100];
  sprintf(buffer2, "v: %18.10e,  f: %18.10e\n", vChebyshev.l1_norm(), fChebyshev.l2_norm());
  //pcout << buffer2;
  //
#ifdef ENABLE_PERIODIC_BC
  //evaluate fChebyshev^{H}*vChebyshev
  alpha=innerProduct(fChebyshev,vChebyshev);
  alphaTimesXPlusY(-alpha,vChebyshev,fChebyshev);
  std::vector<std::complex<double> > T(lanczosIterations*lanczosIterations,0.0);
#else
  alpha=fChebyshev*vChebyshev;
  fChebyshev.add(-1.0*alpha,vChebyshev);
  std::vector<double> T(lanczosIterations*lanczosIterations,0.0); 
#endif
  
  
  T[0]=alpha;
  unsigned index=0;

  //filling only lower trangular part
  for (unsigned int j=1; j<lanczosIterations; j++)
    {
      beta=fChebyshev.l2_norm();
      char buffer1[100];
      sprintf(buffer1, "alpha: %18.10e,  beta: %18.10e\n", alpha, beta);
      v0Chebyshev=vChebyshev; vChebyshev.equ(1.0/beta,fChebyshev);
      eigen.HX(v,f); fChebyshev.add(-1.0*beta,v0Chebyshev);
#ifdef ENABLE_PERIODIC_BC
      alpha = innerProduct(fChebyshev,vChebyshev);
      alphaTimesXPlusY(-alpha,vChebyshev,fChebyshev);
#else      
      alpha = fChebyshev*vChebyshev;  
      fChebyshev.add(-1.0*alpha,vChebyshev);
#endif
     
      index+=1;
      T[index]=beta; 
      index+=lanczosIterations;
      T[index]=alpha;
      sprintf(buffer1, "alpha: %18.10e,  beta: %18.10e\n", alpha, beta);
      //pcout << buffer1;
    }

  //eigen decomposition to find max eigen value of T matrix
  std::vector<double> eigenValuesT(lanczosIterations);
  char jobz='N', uplo='L';
  int n = lanczosIterations, lda = lanczosIterations, info;
  int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n;
  std::vector<int> iwork(liwork, 0);
 
#ifdef ENABLE_PERIODIC_BC
  int lrwork = 1 + 5*n + 2*n*n;
  std::vector<double> rwork(lrwork,0.0); 
  std::vector<std::complex<double> > work(lwork);
  zheevd_(&jobz, &uplo, &n, &T[0], &lda, &eigenValuesT[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);
#else
  std::vector<double> work(lwork, 0.0);
  dsyevd_(&jobz, &uplo, &n, &T[0], &lda, &eigenValuesT[0], &work[0], &lwork, &iwork[0], &liwork, &info);
#endif

  //
  computing_timer.exit_section("Chebyshev upper bound");
  for (unsigned int i=0; i<eigenValuesT.size(); i++){eigenValuesT[i]=std::abs(eigenValuesT[i]);}
  std::sort(eigenValuesT.begin(),eigenValuesT.end()); 
  //
  char buffer[100];
  sprintf(buffer, "bUp1: %18.10e,  bUp2: %18.10e\n", eigenValuesT[lanczosIterations-1], fChebyshev.l2_norm());
  //pcout << buffer;
  
  return (eigenValuesT[lanczosIterations-1]+fChebyshev.l2_norm());
}

//Gram-Schmidt orthonormalization
void dftClass::gramSchmidt(std::vector<vectorType*>& X){
  computing_timer.enter_section("Chebyshev GS orthonormalization"); 
 
  

#ifdef ENABLE_PERIODIC_BC
  unsigned int localSize = vChebyshev.local_size()/2;
#else
  unsigned int localSize = vChebyshev.local_size();
#endif

  //copy to petsc vectors
  unsigned int numVectors = X.size();
  Vec vec;
  VecCreateMPI(PETSC_COMM_WORLD, localSize, PETSC_DETERMINE, &vec);
  VecSetFromOptions(vec);
  //
  Vec *petscColumnSpace;
  VecDuplicateVecs(vec, numVectors, &petscColumnSpace);
  VecDestroy(&vec);

  //
#ifdef ENABLE_PERIODIC_BC
  PetscScalar ** columnSpacePointer;
  VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  for (int i = 0; i < numVectors; ++i)
    {
      std::vector<std::complex<double> > localData(localSize);
      //std::vector<double> localData(localSize);
      std::vector<double> tempReal(localSize),tempImag(localSize);
      X[i]->extract_subvector_to(local_dof_indicesReal.begin(),
				 local_dof_indicesReal.end(),
				 tempReal.begin());

      X[i]->extract_subvector_to(local_dof_indicesImag.begin(),
				 local_dof_indicesImag.end(),
				 tempImag.begin());

      for(int j = 0; j < localSize; ++j)
	{
	  localData[j].real(tempReal[j]);
	  localData[j].imag(tempImag[j]);
	  //localData[j] = tempReal[j];
	}
      std::copy(localData.begin(),localData.end(), &(columnSpacePointer[i][0])); 
    }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#else
  PetscScalar ** columnSpacePointer;
  VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  for (int i = 0; i < numVectors; ++i){
    std::vector<double> localData(localSize);
    std::copy (X[i]->begin(),X[i]->end(),localData.begin());
    std::copy (localData.begin(),localData.end(), &(columnSpacePointer[i][0])); 
  }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#endif


  //
  BV slepcColumnSpace;
  BVCreate(PETSC_COMM_WORLD,&slepcColumnSpace);
  BVSetFromOptions(slepcColumnSpace);
  BVSetSizesFromVec(slepcColumnSpace,petscColumnSpace[0],numVectors);
  BVSetType(slepcColumnSpace,"vecs");
  int numVectors2=numVectors;
  BVInsertVecs(slepcColumnSpace,0, &numVectors2,petscColumnSpace,PETSC_FALSE);
  BVOrthogonalize(slepcColumnSpace,NULL);
  //
  for(int i = 0; i < numVectors; ++i){
    BVCopyVec(slepcColumnSpace,i,petscColumnSpace[i]);
  }
  BVDestroy(&slepcColumnSpace);
  //
  

#ifdef ENABLE_PERIODIC_BC
  VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  for (int i = 0; i < numVectors; ++i)
    {
      std::vector<std::complex<double> > localData(localSize);
      //std::vector<double> localData(localSize);
      std::copy(&(columnSpacePointer[i][0]),&(columnSpacePointer[i][localSize]), localData.begin()); 
      for(int j = 0; j < localSize; ++j)
	{
	  X[i]->local_element(2*j) = localData[j].real();
	  X[i]->local_element(2*j+1) = localData[j].imag();
	}
      X[i]->update_ghost_values();
    }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#else
  VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  for (int i = 0; i < numVectors; ++i)
    {
      std::vector<double> localData(localSize);
      std::copy(&(columnSpacePointer[i][0]),&(columnSpacePointer[i][localSize]), localData.begin()); 
      std::copy(localData.begin(), localData.end(), X[i]->begin());
      X[i]->update_ghost_values();
    }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#endif
  //
  VecDestroyVecs(numVectors, &petscColumnSpace);
  //
  computing_timer.exit_section("Chebyshev GS orthonormalization"); 
}

void dftClass::rayleighRitz(std::vector<vectorType*> &X){
  computing_timer.enter_section("Chebyshev Rayleigh Ritz"); 
  //Hbar=Psi^T*H*Psi
  eigen.XHX(X);  //Hbar is now available as a 1D array XHXValue 

  //compute the eigen decomposition of Hbar
  int n = X.size(), lda = X.size(), info;
  int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n;
   std::vector<int> iwork(liwork,0);
  char jobz='V', uplo='U';

#ifdef ENABLE_PERIODIC_BC
  int lrwork = 1 + 5*n + 2*n*n;
  std::vector<double> rwork(lrwork,0.0); 
  std::vector<std::complex<double> > work(lwork);
  zheevd_(&jobz, &uplo, &n, &eigen.XHXValue[0],&lda,&eigenValues[d_kPointIndex][0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);
#else
  std::vector<double> work(lwork);
  dsyevd_(&jobz, &uplo, &n, &eigen.XHXValue[0], &lda, &eigenValues[d_kPointIndex][0], &work[0], &lwork, &iwork[0], &liwork, &info);
#endif

  //print eigen values
  char buffer[100];
  for (unsigned int i=0; i< (unsigned int)n; i++){
    pcout << "kPoint: "<< d_kPointIndex<<std::endl;
    sprintf(buffer, "eigen value %2u: %18.16e\n", i, eigenValues[d_kPointIndex][i]);
    pcout << buffer;
    }


  //rotate the basis PSI=PSI*Q
  int m = X.size(); 
#ifdef ENABLE_PERIODIC_BC
  int n1 = X[0]->local_size()/2;
  std::vector<std::complex<double> > Xbar(n1*m), Xlocal(n1*m); //Xbar=Xlocal*Q
  std::vector<std::complex<double> >::iterator val = Xlocal.begin();
  for(std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x)
    {
      for (unsigned int i=0; i<(unsigned int)n1; i++)
	{
	  (*val).real((**x).local_element(2*i));
	  (*val).imag((**x).local_element(2*i+1));
	   val++;
	}
    }
#else
  int n1 = X[0]->local_size();
  std::vector<double> Xbar(n1*m), Xlocal(n1*m); //Xbar=Xlocal*Q
  std::vector<double>::iterator val = Xlocal.begin();
  for (std::vector<vectorType*>::iterator x = X.begin(); x < X.end(); ++x)
    {
      for (unsigned int i=0; i<(unsigned int)n1; i++)
	{
	  *val=(**x).local_element(i); 
	  val++;
	}
    }
#endif
  
char transA  = 'N', transB  = 'N';
lda=n1; int ldb=m, ldc=n1;

#ifdef ENABLE_PERIODIC_BC
 std::complex<double> alpha = 1.0, beta  = 0.0;
 zgemm_(&transA, &transB, &n1, &m, &m, &alpha, &Xlocal[0], &lda, &eigen.XHXValue[0], &ldb, &beta, &Xbar[0], &ldc);
#else
 double alpha = 1.0, beta  = 0.0;
 dgemm_(&transA, &transB, &n1, &m, &m, &alpha, &Xlocal[0], &lda, &eigen.XHXValue[0], &ldb, &beta, &Xbar[0], &ldc);
#endif

 
#ifdef ENABLE_PERIODIC_BC
 //copy back Xbar to X
  val=Xbar.begin();
  for (std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x)
    {
      **x=0.0;
      for (unsigned int i=0; i<(unsigned int)n1; i++){
	(**x).local_element(2*i)=(*val).real(); 
	(**x).local_element(2*i+1)=(*val).imag(); 
	val++;
      }
      (**x).update_ghost_values();
    }
#else
  //copy back Xbar to X
  val=Xbar.begin();
  for (std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x)
    {
      **x=0.0;
      for (unsigned int i=0; i<(unsigned int)n1; i++){
	(**x).local_element(i)=*val; val++;
      }
      (**x).update_ghost_values();
    }
#endif

  //set a0 and bLow
  a0[d_kPointIndex]=eigenValues[d_kPointIndex][0]; 
  bLow[d_kPointIndex]=eigenValues[d_kPointIndex].back(); 
  //
  computing_timer.exit_section("Chebyshev Rayleigh Ritz"); 
}

//chebyshev solver
//inputs: X - input wave functions, m-polynomial degree, a-lower bound of unwanted spectrum
//b-upper bound of the full spectrum, a0-lower bound of the wanted spectrum
void dftClass::chebyshevFilter(std::vector<vectorType*> & X, unsigned int m, double a, double b, double a0){
  computing_timer.enter_section("Chebyshev filtering"); 
  double e, c, sigma, sigma1, sigma2, gamma;
  e=(b-a)/2.0; c=(b+a)/2.0;
  sigma=e/(a0-c); sigma1=sigma; gamma=2.0/sigma1;
  
  //Y=alpha1*(HX+alpha2*X)
  double alpha1=sigma1/e, alpha2=-c;
  eigen.HX(X, PSI);
  for (std::vector<vectorType*>::iterator y=PSI.begin(), x=X.begin(); y<PSI.end(); ++y, ++x){  
    (**y).add(alpha2,**x);
    (**y)*=alpha1;
  } 
  //loop over polynomial order
  for (unsigned int i=2; i<m+1; i++){
    sigma2=1.0/(gamma-sigma);
    //Ynew=alpha1*(HY-cY)+alpha2*X
    alpha1=2.0*sigma2/e, alpha2=-(sigma*sigma2);
    eigen.HX(PSI, tempPSI);
    for (std::vector<vectorType*>::iterator ynew=tempPSI.begin(), y=PSI.begin(), x=X.begin(); ynew<tempPSI.end(); ++ynew, ++y, ++x){  
      (**ynew).add(-c,**y);
      (**ynew)*=alpha1;
      (**ynew).add(alpha2,**x);
      **x=**y;
      **y=**ynew;
    }
    sigma=sigma2;
  }
  
  //copy back PSI to eigenVectors
  for (std::vector<vectorType*>::iterator y=PSI.begin(), x=X.begin(); y<PSI.end(); ++y, ++x){  
    **x=**y;
  }   
  computing_timer.exit_section("Chebyshev filtering"); 
}
 
