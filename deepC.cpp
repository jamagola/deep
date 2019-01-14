/******************************************************
 This project aims to develop deep learning using ANN *
 and backpropagation algorithm.                       *
                                                      *
 Author: Golam Gause Jaman                            *
 email: jamagola@isu.edu                              *
 ******************************************************/

#include "learn.h"

using namespace std;

/* Run this program using the console pauser or add your own getch, system("pause") or input loop */
int main() {
	
	int P = 8; 							//Training data pair count
	int L = 4; 							//Total layers (hidden and others) including input and output
	int epoch = 5000;					//Epoch run count
	int batch = 1;						//Batch process/update step
	//int *N; 							//Nuron/nodes count in each layer
	//N = new int[L];
	int N[4]={2,3,3,1}; 
	double eta = 0.9;					//Learning step/rate
	double bias = 1; 					//Bias
	double error = 0;					//Mean Square Error
	
	double *X; 							//Inputs or Outputs
	X = new double[inOutN(N,L,P)];
	double *delta; 						//Error terms
	delta = new double[sumN(N,L)];
	double *W;							//Weights
	W = new double[linkN(N, L)];
	double *W_;							//Weights back up
	W_ = new double[linkN(N, L)];
	double *X_; 						//Inputs-Hidden-Outputs
	X_ = new double[inOutN(N,L,1)];
	
	//double *input;
	//input = new double[N[0]*P];
	//double *output;
	//output = new double[N[L-1]*P];
	double input[16]={0,0,0,1,1,0,1,1,0,1,0,0,1,1,1,0};
	double output[8]={0,1,1,0,1,0,0,1};
	
	double in[2]={1,1}; //Validation
	error = deep(W,W_,X_,delta,input,output,N,L,P,bias,batch,epoch,eta,in);
	
	//Display All nodes output and error
	for(int i=0; i<sumN(N,L); i++) cout<<"Node "<<i<<"\t:"<<X_[i]<<endl;
	cout << endl << endl <<"Error(MSE): "<< error << endl;
	
	//Free memory
	delete [] X;
	delete [] X_;
	delete [] W;
	delete [] W_;
	delete [] delta;
	//delete [] input;
	//delete [] output;
	//delete [] N; 
	
	system("pause");
	return 0;
}
