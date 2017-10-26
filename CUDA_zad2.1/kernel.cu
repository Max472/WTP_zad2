#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include <Windows.h>

// EXERCISE DESCRIPTION
/*
Æwiczenie 2 - Obliczenia ogólnego przeznaczenia na procesorach graficznych. Termin: na 4 zajêcia.
Celem æwiczenia jest zaprojektowanie i implementacja masowo-równoleg³ego algorytmu do obliczania operacji mno¿enia macierzy przez wektor.
Algorytm powinien dzia³aæ poprawnie dla macierzy kwadratowych o N na N elementach, gdzie N jest ca³kowit¹ potêg¹ dwóch.
Podczas projektowania algorytmu nale¿y przyj¹æ:
- nieskoñczon¹ liczbê równoleg³ych procesorów,
- mo¿liwoœæ synchronizacji w¹tków obliczeniowych.
*/
// EXAMPLE APPLICATION
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
// C IMPLEMENTATION
/*
float *a, *b, *c;
float *dev_a, *dev_b, *dev_c;

int N = 4;
a=(float*)malloc(sizeof(float)*N);
b=(float*)malloc(sizeof(float)*N*N);
c=(float*)malloc(sizeof(float)*N);

init_array(a, N);
init_mat(b, N, N);
init_array(c, N);

printf("<<<<<<<<<< initial data:\n");
print_array(a, N, "in-vector");
print_mat(b, N, N, "matrix");
print_array(c, N, "out-");

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
print_array(c, N, "out-vector");
*/

#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__
void kernel(float *vec, float *mat, float *out, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<N)
	{
		float sum = 0.0;
        for(int j = 0; j < N; j++)
            sum += vec[j]*mat[j+i*N];
        out[i] = sum;
    }
}

// debuging functions
void init_array(float *a, const int N);
void init_mat(float *a, const int N, const int M);
void print_array(float *a, const int N, char *d);
void print_mat(float *a, const int N, const int M, char *d);

// helper functions
void fill(float * a, unsigned int n)
{
	for (unsigned int i = 0; i < n; i++)
	{
		a[i] = i+1;
	}
}
void print(float * a, unsigned int n, unsigned int m, char * s)
{
	for (unsigned int i = 0; i < n; i++)
	{
		if (i%m == 0)
		{
			std::cout << "\n" << s;
		}
		std::cout << "\t" << a[i];
	}
}

// test functions
void gpu(float * vector, float * matrix, float * result, unsigned int N)
{
	kernel << <N / 256 + 1, 256 >> >(vector, matrix, result, N);
}
void cpu(float * vector, float * matrix, float * result, unsigned int N)
{
	for (unsigned int i = 0; i < N; i++)
	{
		result[i] = 0.0;
		for (unsigned int j = 0; j < N; j++)
		{
			result[i] += matrix[j+i*N] * vector[j];
		}
	}
}

int main (void)
{
	std::ios_base::sync_with_stdio(false);
	srand(time(NULL));

	unsigned int Ns[] = { 0, 1, 2, 3, 4 , 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
	unsigned int N;
	unsigned int tests = 16;
	unsigned int repeats = 10;
	float *vector, *matrix, *result_gpu, *result_cpu;
	float *dev_vector, *dev_matrix, *dev_result;

	bool verify = true;

	if(!verify) std::cout << "N\tcpu\tgpu\treal\n";
	for (int t = 0; t < tests ; ++t)
	{
		double time_cpu = 0.0;
		double time_prep = 0.0;
		double time_gpu = 0.0;
		for (int r = 0; r<repeats; r++)
		{
			N = 1 << Ns[t];
			// cpu memory allocation
			vector = new float[N];
			matrix = new float[N*N];
			result_cpu = new float[N];

			// gpu memory allocation
			cudaMalloc((void**)&dev_vector, sizeof(float)*N);
			cudaMalloc((void**)&dev_matrix, sizeof(float)*N*N);
			cudaMalloc((void**)&dev_result, sizeof(float)*N);

			// data initialization
			fill(vector, N);
			fill(matrix, N*N);

			clock_t start_cpu = clock();
			cpu(vector, matrix, result_cpu, N);
			time_cpu += double(clock() - start_cpu) / CLOCKS_PER_SEC;
			
			clock_t start_prep = clock();
			cudaMemcpy(dev_vector, vector, sizeof(float)*N, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_matrix, matrix, sizeof(float)*N*N, cudaMemcpyHostToDevice);
			time_prep += double(clock() - start_prep) / CLOCKS_PER_SEC;

			clock_t start_gpu = clock();
			gpu(dev_vector, dev_matrix, dev_result, N);
			time_gpu += double(clock() - start_gpu) / CLOCKS_PER_SEC;

			if (verify)
			{
				print(vector, N, N, "VEC: ");
				std::cout << "\n";
				print(matrix, N*N, N, "MAT: ");
				std::cout << "\n";
				result_gpu = new float[N];
				cudaMemcpy(result_gpu, dev_result, sizeof(float)*N, cudaMemcpyDeviceToHost);
				print(result_gpu, N, N, "GPU: ");
				std::cout << "\n";
				delete[] result_gpu;
				print(result_cpu, N, N, "CPU: ");
				std::cout << "\n";
			}

			// cpu memory release
			delete[] vector;
			delete[] matrix;
			delete[] result_cpu;

			// gpu memory release
			cudaFree(dev_vector);
			cudaFree(dev_matrix);
			cudaFree(dev_result);
			cudaDeviceReset();
		}
		time_cpu /= repeats;
		time_gpu /= repeats;
		time_prep /= repeats;

		if (!verify) std::cout << N << "\t" << time_cpu << "\t" << time_gpu << "\t" << time_gpu + time_prep << "\n";
	}

	if(verify) std::cin >> N;
    return 0;
};

void init_array(float * a, const int N)
{
    for(int i=0; i<N; i++)
		a[i] = rand() % 20;
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

void print_mat(float *a, const int N, const int M, char *d)
{
	int i, j;
    for(i=0; i<N; i++)
	{
		printf("\n%s[%d]:", d, i);
		for (j=0; j<N; j++)
			printf("\t%6.4f", a[(i*N)+j]);
    }
    printf("\n");
}