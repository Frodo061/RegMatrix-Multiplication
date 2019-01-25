#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <mm_malloc.h>
#include <stdlib.h>

using namespace std;
cudaEvent_t start, stop;

void startStopWatch () {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
}

void stopStopWatch () {
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	cout << time << " ms." << endl;
}

__global__
void multMat(float *a,float *b, float *c, int N){
    int linha=blockIdx.y*blockDim.y+threadIdx.y; 
    int coluna=blockIdx.x*blockDim.x+threadIdx.x;
    float sum=0;
    if(coluna<N&&linha<N){
        for(int i=0;i<N;i++)sum+=a[linha*N+i]*b[i*N+coluna];
        c[linha*N+coluna]=sum;
    }
}

void stencil(float *a, float *b, float *c, int N){
    float *devA,*devB, *devC;
    int NQ = N*N;
    cudaMalloc((void**) &devA, NQ * sizeof(float));
    cudaMalloc((void**) &devB, NQ * sizeof(float));
    cudaMalloc((void**) &devC, NQ * sizeof(float));
    
    startStopWatch();
	cudaMemcpy(devA,a,NQ*sizeof(float),cudaMemcpyHostToDevice);	
	cudaMemcpy(devB,b,NQ*sizeof(float),cudaMemcpyHostToDevice);	
    stopStopWatch();
    dim3 dimGrid(N,N);
    dim3 dimBlock(1,1);
    startStopWatch();
    multMat<<<dimGrid,dimBlock>>>(devA,devB,devC,N);
    stopStopWatch();
    startStopWatch();
    cudaMemcpy(c,devC,NQ*sizeof(float),cudaMemcpyDeviceToHost);
    stopStopWatch();
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

void newMatrices(float **a, float **b, float **c, int N){
    int i;
    int NQ = N*N;
    *a = (float *)_mm_malloc(NQ * sizeof(float), 32);
    *b = (float *)_mm_malloc(NQ * sizeof(float), 32);
    *c = (float *)_mm_malloc(NQ * sizeof(float), 32);
    for (i = 0; i < NQ; i++){
        (*b)[i] = 1;
        (*a)[i] = ((float)rand()) / ((float)RAND_MAX);
    }
}

int main (int argc, char** argv) {
  	int N = atoi(argv[1]);
    srand(0);
	float *a,*b,*c;
    newMatrices(&a,&b,&c,N);
    stencil(a,b,c,N);
	return 0;
}
