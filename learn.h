/******************************************************
 This project aims to develop deep learning using ANN *
 and backpropagation algorithm.                       *
                                                      *
 Author: Golam Gause Jaman                            *
 email: jamagola@isu.edu                              *
 ******************************************************/
 
 /*
If you want to use an array, you'll have to allocate a new block, large enough for the whole new array, 
and copy the existing elements across before deleting the old array. Or you could use a more complicated 
data structure that doesn't store its elements contiguously.
Luckily, the standard library has containers to handle this automatically; including vector, a resizable array.

e.g. std::vector<float> X;
 
Source: https://stackoverflow.com/questions/29015546/allocate-more-memory-for-dynamically-allocated-array
*/

/*	
Followings are some key information:
Pattern by pattern learning when update occurs for each data pair.
Batch or mini-batch learning when update occurs after group of data pairs.
Layers include input, hidden and output. Minimum layers supported is 3.
Learning rate: Not adaptive (needs attention).
Bias: Single constant used across all nodes with variable weight.
Trained using same sequence a provided 
(needs attention, use of n-fold, n segmented data and use one for verification , rest for training).
Stopping criteria: Epoch only (needs attention, error index).
Error: Mean Square Error.
Output range from each node: 0 ~ 1
Activation function used: Sigmoid
*/

#ifndef LEARN
#define LEARN

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

int maxN(int*, int);
int sumN(int*, int);
int inOutN(int*, int, int);
int linkN(int*, int);
void randomizeW(double*, double*, int);
void copyW(double*, double*, int, int);
void dispW(double*, int*, int);
void initX(double*, double*, int*, int, int);
double evalX(double*, double*, double*, double*, double*, int*, int, int, double);
double sigmoid(double);
double addrW(double*, int*, int, int, int, int, int);
void evalDelta(double*, double*, double*, int*, int, double);
void updateW(double*, double*, double*, double*, int*, int, double, double);
double trainDeep(double*, double*, double*, double*, double*, double*, int*, int, int, double, int, int, double);
void evalOUT(double*, double*, double*, int*, int, double);
double deep(double*, double*, double*, double*, double*, double*, int*, int, int, double, int, int, double, double*);

// Maximum node count among layers
int maxN(int* N, int size)
{
	int result = 0;
	for(int i=0; i<size; i++) if(N[i]>result) result = N[i];
	return result;
}

//Total node count
int sumN(int* N, int size)
{
	int result = 0;
	for(int i=0; i<size; i++) result += N[i];
	return result;
}

//Total outputs across training dataset
int inOutN(int* N, int L, int P)
{
	return sumN(N,L)*P;
}

//Total links associated with weights
int linkN(int* N, int L)
{
	int result = 0;
	for(int l=0; l<L-1; l++) result += N[l]*N[l+1]+N[l+1];
	return result;
}

// Copy elements
void copyW(double* W, double* W_, int size, int batch)
{
	for(int i=0; i<size; i++) {
		W[i] = W_[i]/batch;
		W_[i]=0;
	}
}

//Randomization
void randomizeW(double* W, double* W_, int size)
{
	srand(time(NULL));
	for(int i=0; i<size; i++) {
		W[i] = (rand()%100)/100.00;
		W_[i] = 0;
	}
}

//Display weights
void dispW(double* W, int* N, int L)
{    
    int temp = 0;
	cout << endl << "<-----Weights----->" << endl;
	for(int i=0; i<L-1; i++)
	{
		for(int j=0; j<N[i+1]; j++)
		{
			int k=0;
			for(; k<N[i]; k++)
			{
				cout << W[k+temp] << " "; 
			}
			cout << W[k+temp] << endl; //Bias
			temp = k+temp+1;
		}
		cout << endl;
	}	
}

//Calculate output of each node
void initX(double* X, double* input, int* N, int L, int P)
{
	int n=0;
	int s=inOutN(N, L, P); 
	int t=sumN(N,L);
	while(n<s)
	{
		for(int p=0; p<P; p++)
		{
			for(int l=0; l<L; l++)
			{
				for(int i=0; i<N[l]; i++)
				{
					if(l==0) {
						X[n]=input[i+N[0]*p];
					}
					else {
						X[n] = 0;
					}
					n++;
				}
			}
		}
		
	}
}

//Activation function
double sigmoid(double x)
{
	return exp(x)/(1+exp(x));
}

//Evaluatate
double evalX(double* W, double* X_, double* delta, double* input, double* output, int* N, int p, int L, double bias)
{
	int n=0;
	int m=0;
	int q=N[0];
	int t=0;
	double temp=0;
	double err=0;
	
	for(int i=0; i<N[0]; i++) {
		X_[i]=input[i+p*N[0]];
	}
	for(int l=1; l<L; l++)
	{
		for(int j=0; j<N[l]; j++)
		{
			for(int i=0; i<N[l-1]; i++, m++)
			{
		 		temp += X_[i+n]*W[m];	
			}
			temp += bias*W[m];
			X_[q] = sigmoid(temp);
			temp=0;
			if(l==L-1)
			{
				err = err+ pow((output[N[l]*p+t] - X_[q]),2);
				delta[q]=X_[q]*(1-X_[q])*(output[N[l]*p+t] - X_[q]);
				t++;
			}
			q++; 
			m++;
		}
		n+=N[l-1];
	}
	return err/N[L-1];
}

//Weight Address
double addrW(double* W, int* N, int L, int fromL, int toL, int fromN, int toN)
{
	return W[linkN(N,toL)+toN*(N[fromL]+1)+fromN];
}

//Evaluate error term using back-propagation
void evalDelta(double* X_, double* W, double* delta, int* N, int L, double bias)
{
	int q=0;
	int t=0;
	double temp=0.0;
	
	for(int l=L-2; l>0; l--)
	{
		q=sumN(N,l);
		for(int j=0; j<N[l]; j++, t++)
		{
			for(int i=0; i<N[l+1]; i++)
			{
				temp += delta[sumN(N,l+1)+i]*addrW(W,N,L,l,l+1,j,i);	
			}
			delta[q+t] = X_[q+t]*(1 - X_[q+t])*temp;
			temp = 0;
		}
		t=0;
	}
}


//Update weights
void updateW(double* W, double* W_, double* X_, double* delta, int* N, int L, double eta, double bias)
{
	int m=0;
	int b=N[0]+1;

	for(int l=0; l<L-1;l++)
	{
		for(int j=0; j<N[l+1]; j++, b++)
		{
			for(int i=0; i<N[l]; i++, m++)
			{
				W_[m]=W_[m]+W[m]+eta*delta[sumN(N,l+1)+j]*X_[sumN(N,l)+i];
			}
			W_[m]=W_[m]+W[m]+eta*delta[sumN(N,l+1)+j]*bias;
			m++;
		}
    }
}

//Training process : batch preferred to be factor of P
double trainDeep(double* W, double* W_, double* X_, double* delta, double* input, double* output, int* N, int L, int P, double bias, int batch, int epoch, double eta )
{
	int progress=0;
	double erro=0;
	double errorP=0;
	//Step 1: Randomize weights
	randomizeW(W,W_,linkN(N,L));
	
	for(int e=0; e<epoch; e++)
	{
		for(int p=0; p<P; p++)
		{
			//Step 2: Evaluate firing values
			erro+=evalX(W,X_,delta,input,output,N,p,L,bias);
			//Step 3: Evaluate error terms
			evalDelta(X_,W,delta,N,L,bias);
			//Step 4: Evaluate weights
			updateW(W,W_,X_,delta,N,L,eta,bias);
			progress++;
			
			if(progress==batch)
			{
				copyW(W,W_,linkN(N,L),batch);
				progress=0;
			}
		}
		errorP=erro/P;
		erro=0;
	}
	return errorP;
}

//Verification
void evalOUT(double* W, double* X_, double* in, int* N, int L, double bias)
{
	int n=0;
	int m=0;
	int q=N[0];
	double temp=0;
	
	for(int i=0; i<N[0]; i++) {
		X_[i]=in[i];
	}
	for(int l=1; l<L; l++)
	{
		for(int j=0; j<N[l]; j++)
		{
			for(int i=0; i<N[l-1]; i++, m++)
			{
		 		temp += X_[i+n]*W[m];	
			}
			temp += bias*W[m];
			X_[q] = sigmoid(temp);
			temp=0;
			q++; 
			m++;
		}
		n+=N[l-1];
	}
}

//Evaluate network
double deep(double* W, double* W_, double* X_, double* delta, double* input, double* output, int* N, int L, int P, double bias, int batch, int epoch, double eta, double* in)
{
	double error;
	//Training process
	error = trainDeep(W,W_,X_,delta,input,output,N,L,P,bias,batch,epoch,eta);
	evalOUT(W,X_,in,N,L,bias);
	return error;
}

#endif
