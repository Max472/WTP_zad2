
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <Windows.h>
/*
using namespace std;

const unsigned int N  = 16*1024*1024;
const unsigned int Nt = 512;

float tab1[N];
float tab2[N];
float tab3[N];

inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		cout << "CUDA Runtime Error: " << cudaGetErrorString(result) << endl;
	}
	return result;
}

__global__ void kernel(float* t1, float* t2, float* t3)
{
	float v0, v1, v2;
	unsigned int idx;
	idx = threadIdx.x + blockDim.x * blockIdx.x;
	v0 = t1[idx];
	v1 = t2[idx];
	v2 = v0 + v1;
	t3[idx] = v2;
}

void gpu_dodaj()
{
	unsigned int Nb;
	unsigned int bytes = N * sizeof(float);
	float *d_tab1, *d_tab2, *d_tab3;
	Nb = N / Nt;
	cout << "Nb: " << Nb << "\n";
	checkCuda(cudaMalloc((float**)&d_tab1, bytes));
	checkCuda(cudaMalloc((float**)&d_tab2, bytes));
	checkCuda(cudaMalloc((float**)&d_tab3, bytes));
	checkCuda(cudaMemcpy((void*)(d_tab1), (void*)(tab1), bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy((void*)(d_tab2), (void*)(tab2), bytes, cudaMemcpyHostToDevice));	
	kernel <<<Nb, Nt>>>(d_tab1, d_tab2, d_tab3);
	checkCuda(cudaMemcpy((void*)(tab3), (void*)(d_tab3), bytes, cudaMemcpyDeviceToHost));
	cudaFree(d_tab1);
	cudaFree(d_tab2);
	cudaFree(d_tab3);
	cudaDeviceReset();
}

void cpu_dodaj()
{
	double tt;
	LARGE_INTEGER tb, te, tf;
	QueryPerformanceFrequency(&tf);
	QueryPerformanceCounter(&tb);
	for (unsigned int i = 0; i < N; i++)
	{
		tab3[i] = tab2[i] + tab1[i];
	}
	QueryPerformanceCounter(&te);
	tt = 1000.0*(double(te.QuadPart - tb.QuadPart)) / double(tf.QuadPart);
	cout << "CPU time: " << tt << " ms\n";
}

int main()
{
	for (unsigned int i = 0; i < N; i++)
	{
		tab1[i] = float(i);
		tab2[i] = float(N - i);
	}
	cpu_dodaj();
	gpu_dodaj();

	int i;
	cin >> i;
	return 0;
}
*/

#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__
void kernel(float *vec, float *mat, float *out, const int N, const int M){
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
        float sum=0;
    if(tid<M){
        for(int i=0; i<N; i++)
            sum += vec[i]*mat[(i*M)+tid];
        out[tid]=sum;
    }
}

// debuging functions
void init_array(float *a, const int N);
void init_mat(float *a, const int N, const int M);
void print_array(float *a, const int N, char *d);
void print_mat(float *a, const int N, const int M, char *d);

int main (void) {
        srand( time(NULL) );

    float *a, *b, *c;
        float *dev_a, *dev_b, *dev_c;

    int N=4;
    a=(float*)malloc(sizeof(float)*N);
    b=(float*)malloc(sizeof(float)*N*N);
    c=(float*)malloc(sizeof(float)*N);
        init_array(a, N);
        init_mat(b, N, N);
        init_array(c, N);

    printf("<<<<<<<<<< initial data:\n");
        print_array(a, N, "in-vector");
        print_mat(b, N, N, "matrix");
        print_array(c, N, "out-vector");

        cudaMalloc((void**)&dev_a, sizeof(float)*N);
        cudaMalloc((void**)&dev_b, sizeof(float)*N*N);
        cudaMalloc((void**)&dev_c, sizeof(float)*N);

        cudaMemcpy(dev_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    printf("\n\nRunning Kernel...\n\n");
        kernel<<<N/256+1, 256>>>(dev_a, dev_b, dev_c, N, N);
        //printf("error code: %s\n",cudaGetErrorString(cudaGetLastError()));

        cudaMemcpy(c, dev_c, sizeof(float)*N, cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

    printf(">>>>>>>>>> final data:\n");
	printf(">>>>>>>>>> final data:\n");
        print_array(c, N, "out-vector");

		int i;
		std::cin >> i;

        return 0;
};

void init_array(float *a, const int N) {
        int i;
        for(i=0; i<N; i++)
                a[i] = rand() % 4 + 1;
}
void init_mat(float *a, const int N, const int M) {
        int i, j;
        for(i=0; i<N; i++)
            for(j=0; j<N; j++)
                    a[i*N+j] = rand() % 4 + 1;
}
void print_array(float *a, const int N, char *d) {
        int i;
        for(i=0; i<N; i++)
                printf("\n%s[%d]: %f",d, i, a[i]);
    printf("\n");
}
void print_mat(float *a, const int N, const int M, char *d) {
        int i, j;
        for(i=0; i<N; i++){
        printf("\n%s[%d]:", d, i);
        for (j=0; j<N; j++)
                    printf("\t%6.4f", a[i*N+j]);
    }
    printf("\n");
}