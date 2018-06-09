/*lab-3 of ECE 285 GPU Programming
 Student: Wenyu Zhang
 PID: A53238371
 Email: wez078@ucsd.edu*/

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <numeric>
// CPU library for Half float
#include "./half-1.12.0/half.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "tmm.h"
using namespace std;
using namespace tmm;
// k128-w32
#include <curand.h>
#include <curand_kernel.h>
#include <tuple>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <sstream>
#include <math.h>
#if defined USEOMP
#include <omp.h>
#endif

//func declarations
__global__ void tmm_kernel(Parameter para, tmm_int a_seg, tmm_int b_seg, tmm_int k, half *gpuHalfp, half *gpuHalfq, half *gpuR);
__global__ void transform_half(half *gpu_half_feature, float *gpu_float_feature, long long vec_size);
void tmm_update_k128(Parameter para, tmm_model *model, tmm_problem *prob);
tmm_problem read_problem(string path);
void grid_problem(tmm_problem* prob, Parameter para);
tmm_model* tmm_load_model(char const *path);
void init_model(tmm_model*model, tmm_problem*prob, Parameter para);
void tmm_destroy_model(tmm_model **model);
void transform_R(short *halfR, float *floatR, tmm_problem *prob, Parameter para);
tmm_float look_up_floatR(tmm_int u, tmm_int v, tmm_model *model, tmm_problem *prob);
tmm_double calc_rmse(tmm_problem *prob, tmm_model *model);

//------------------------------------------------------------Kernel Functions-----------------------------------------------

//-------------------------kernel----------------------------
__global__ void tmm_kernel(Parameter para, tmm_int a_seg, tmm_int b_seg, tmm_int k, half *gpuHalfp, half *gpuHalfq, half *gpuR)
{
    //from GMEM to SMEM
    int bx = blockDim.x, by = blockDim.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int I = blockIdx.y*by + ty, J = blockIdx.x*bx + tx; // row and col
    int t = para.tile;
	__shared__ half aTile[32][32];
	__shared__ half bTile[32][32];
    float c = 0.0f;     
	for (unsigned int k1=0; k1 < (k+t-1)/t; k1++)
	{
		if (I < a_seg && k1*t+tx < k){
			aTile[ty][tx] = __ldg(&gpuHalfp[I*k + k1*t + tx]);
		}
		if (J < b_seg && k1*t+ty < k){
			bTile[ty][tx] = __ldg(&gpuHalfq[J*k + k1*t + ty]);
		}
		__syncthreads(); // Synchronizes all threads in a block	
		for (unsigned int k2=0; k2< t; k2++)
            c += __half2float(aTile[ty][k2])*__half2float(bTile[k2][tx]);
        __syncthreads(); // Avoids memory hazards
	}
	if (I < a_seg && J < b_seg)
        gpuR[I*a_seg + J] = __float2half(c);
}
//--------------------------------------------------------------------
__global__ void transform_half(half *gpu_half_feature, float *gpu_float_feature, long long vec_size)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int number_threads = gridDim.x*blockDim.x;

	for (long long i = tid; i < vec_size; i += number_threads)
	{
		gpu_float_feature[i] = __half2float(gpu_half_feature[i]);
	}
}
//---------------------------------------------------------------------
void transform_R(short *halfR, float *floatR, tmm_problem *prob, Parameter para)
{

	half *gpu_half_R;
	float *gpu_float_R;

	cudaMalloc(&gpu_half_R, sizeof(half)*prob->gridSizeM*prob->gridSizeN);
	cudaMalloc(&gpu_float_R, sizeof(float)*prob->gridSizeM*prob->gridSizeN);
	gpuErr(cudaPeekAtLastError());

	for (int i = 0; i < para.rowScale*para.colScale; i++)
	{
		cudaMemcpy(gpu_half_R, halfR + i*prob->gridSizeM*prob->gridSizeN, sizeof(half)*prob->gridSizeM*prob->gridSizeN, cudaMemcpyHostToDevice);
		gpuErr(cudaPeekAtLastError());

		int num_blocks = (prob->gridSizeM*prob->gridSizeN + 255) / 256;
		if (num_blocks > 8 * 24)num_blocks = 8 * 24;

		transform_half <<<num_blocks, 256 >>>(gpu_half_R, gpu_float_R, prob->gridSizeM*prob->gridSizeN);

		gpuErr(cudaPeekAtLastError());
		cudaMemcpy(floatR + i*prob->gridSizeM*prob->gridSizeN, gpu_float_R, sizeof(float)*prob->gridSizeM*prob->gridSizeN, cudaMemcpyDeviceToHost);
		gpuErr(cudaPeekAtLastError());
	}

	cudaFree(gpu_half_R);
	cudaFree(gpu_float_R);
	gpuErr(cudaPeekAtLastError());
}

//--------------------------core---------------------------------------

void tmm_update_k128(Parameter para, tmm_model *model, tmm_problem *prob)
{   using half_float::half;
    printf("calling tmm_update_k128()...\n");
	printf("gridSizeM: %d\n", model->gridSizeM);
	printf("gridSizeN: %d\n", model->gridSizeN);

	printf("k: %d\n", model->k);
    //malloc
    cudaMalloc(reinterpret_cast<void**>(&model->gpuHalfp), sizeof(half)*model->gridSizeM*model->k);
    cudaMalloc(reinterpret_cast<void**>(&model->gpuHalfq), sizeof(half)*model->gridSizeN*model->k);
	cudaMalloc(reinterpret_cast<void**>(&prob->gpuR), sizeof(half)*model->gridSizeM*model->gridSizeN);
	
	//gpuErr(cudaPeekAtLastError());

	for (int rowTile = 0; rowTile < para.rowScale; rowTile++)
	{
		for (int colTile = 0; colTile < para.colScale; colTile++)
		{
			short *p_tmp = model->halfp + model->gridSizeM*model->k*rowTile;
            short *q_tmp = model->halfq + model->gridSizeN*model->k*colTile;
			assert(p_tmp);
			assert(q_tmp); 
			
			// Copy from CPU to GMEM
			cudaMemcpy(model->gpuHalfp, p_tmp, sizeof(half)*model->gridSizeM*model->k, cudaMemcpyHostToDevice);
            cudaMemcpy(model->gpuHalfq, q_tmp, sizeof(half)*model->gridSizeN*model->k, cudaMemcpyHostToDevice);
			
			// Dim Configuration
            dim3 block(32, 32);
			dim3 grid((model->gridSizeN+block.x-1)/block.x, (model->gridSizeM+block.y-1)/block.y);
			// Kernel
			tmm_kernel<<<grid,block>>>(para, model->gridSizeM, model->gridSizeN, model->k, model->gpuHalfp, model->gpuHalfq, prob->gpuR);
			
			// Copy from GMEM to CPU
			short *R_tmp = prob->halfR + model->gridSizeM*model->gridSizeN*(para.colScale*rowTile+colTile);
			cudaMemcpy(R_tmp, prob->gpuR, (long long)(sizeof(half))*model->gridSizeM*model->gridSizeN , cudaMemcpyDeviceToHost);
		}
	}
    cudaDeviceSynchronize();
    //transform_R(prob->halfR, prob->floatR, prob, para);
    cudaFree(model->gpuHalfp);
    cudaFree(model->gpuHalfq);
	cudaFree(prob->gpuR);
}
//------------------------------------------------------------Host Functions-------------------------------------------------
//----------------------------------------------------------
tmm_problem read_problem(string path)  //load matrix R(m n nnz R)
{
	//A simple function that reads the sparse matrix in COO manner.
	printf("read_problem:%s\n", path.c_str());
	tmm_problem prob;
	prob.m = 1;
	prob.n = 1;
	prob.nnz = 0;
	prob.R = nullptr;


	if (path.empty())
		return prob;

	FILE*fptr = fopen(path.c_str(), "rb");
	if (fptr == NULL) {
		printf("error file open %s\n", path.c_str());
		return prob;
	}


	unsigned int tmp;
	fread(&prob.m, sizeof(unsigned int), 1, fptr);
	fread(&prob.n, sizeof(unsigned int), 1, fptr);
	fread(&tmp, sizeof(unsigned int), 1, fptr);
	prob.nnz = tmp;

	tmm_node *R = new tmm_node[prob.nnz];


	long long idx = 0;
	while (true)
	{
		int flag = 0;
		int u, v;
		float r;

		flag += fread(&u, sizeof(int), 1, fptr);
		flag += fread(&v, sizeof(int), 1, fptr);
		flag += fread(&r, sizeof(float), 1, fptr);

		if (flag != 3)break;

		R[idx].u = u;
		R[idx].v = v;
		R[idx].r = r;
		idx++;
	}
	prob.R = R;

	fclose(fptr);
	printf("m:%lld, n:%lld, nnz:%lld\n\n", prob.m, prob.n, prob.nnz);
	return prob;
}
//-----------------------------------------------------------------
void grid_problem(tmm_problem* prob, Parameter para) //grid matrix R(gridSizeM, gridSizeN, halfR, floatR)
{
	clock_t start;

	printf("grid problem ...\n\n\n");
	fflush(stdout);

	//grid the problem into several grids
	//In our homework, ux = vy = 2, u_seg = m/2, v_seg = n/2
	long long u_seg, v_seg;
	u_seg = (long long)ceil((double)prob->m / para.rowScale);
	v_seg = (long long)ceil((double)prob->n / para.colScale);

	prob->gridSizeM = u_seg;
	prob->gridSizeN = v_seg;

	prob->halfR = (short*)malloc((long long)(sizeof(short)) * prob->m * prob->n);
    prob->floatR = (short*)malloc((long long)(sizeof(float)) * prob->m * prob->n);
}

//-------------------------------------------
tmm_model* tmm_load_model(char const *path)  // load feature matrix P, Q(m, n, k, P, Q)
{
	printf("tmm_load_model called\n");

	FILE* fptr = fopen(path, "rb");
	if (fptr == NULL)
	{
		printf("%s open failed\n", path);
		exit(0);
	}
	clock_t start = clock();

	tmm_model *model = new tmm_model;
	model->P = nullptr;
	model->Q = nullptr;

	int count;

	int tmp_f, tmp_m, tmp_n, tmp_k;

	count = fread(&tmp_m, sizeof(int), 1, fptr);
	count = fread(&tmp_n, sizeof(int), 1, fptr);
	count = fread(&tmp_k, sizeof(int), 1, fptr);

	model->m = tmp_m;
	model->n = tmp_n;
	model->k = tmp_k;

	printf("m:   %lld\n", model->m);
	printf("n:   %lld\n", model->n);
	printf("k:   %lld\n", model->k);

	printf("p_size:%lld\n", ((long long)model->m)*model->k);

	try
	{
		model->P = malloc_aligned_float<short>((tmm_long)model->m*model->k);
		model->Q = malloc_aligned_float<short>((tmm_long)model->n*model->k);
	}
	catch (bad_alloc const &e)
	{
		cerr << e.what() << endl;
		tmm_destroy_model(&model);
		return nullptr;
	}

	auto read = [&](short *ptr, tmm_int size)
	{
		for (tmm_int i = 0; i < size; i++)
		{
			short *ptr1 = ptr + (tmm_long)i*model->k;
			count = fread(ptr1, sizeof(short), model->k, fptr);
			if (i % 100000000 == 0)printf("progress:%%%.3f\n", ((double)100.0)*i / size);
		}
	};


	printf("loading feature p m:%lld ...\n", model->m);
	read(model->P, model->m);
	printf("loading feature q n:%lld ...\n", model->n);
	read(model->Q, model->n);

	printf("time elapsed:%.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);

	return model;
}
//-----------------------------------------------------------------
void init_model(tmm_model*model, tmm_problem*prob, Parameter para)//malloc(gridSizeM, gridSizeN, halfpq)
{

    printf("init model ...\n");
    clock_t start = clock();

    //tmm_model *model = new tmm_model;
    //model->fun = 0;
    model->gridSizeM = prob->gridSizeM;
    model->gridSizeN = prob->gridSizeN;

    //allocate memory
    cudaMallocHost(&model->halfp, sizeof(short)*model->m*model->k);
    cudaMallocHost(&model->halfq, sizeof(short)*model->n*model->k);
	cudaMemcpy(model->halfp, model->P, sizeof(short)*model->m*model->k, cudaMemcpyHostToHost);
	cudaMemcpy(model->halfq, model->Q, sizeof(short)*model->n*model->k, cudaMemcpyHostToHost);

    printf("time elapsed:%.8lfs\n",(clock() - start)/(double)CLOCKS_PER_SEC);
    printf("\n\n\n");
}

//-----------------------------------------
void tmm_destroy_model(tmm_model **model)
{
	if (model == nullptr || *model == nullptr)
		return;
#ifdef _WIN32
	_aligned_free((*model)->P);
	_aligned_free((*model)->Q);
#else
	free((*model)->P);
	free((*model)->Q);
#endif
	delete *model;
	*model = nullptr;
}

//-----------------------------------------------------------------
tmm_float look_up_floatR(tmm_int u, tmm_int v, tmm_model *model, tmm_problem *prob)
{
	int partx = u/model->gridSizeM;
	int gridm = u%model->gridSizeM;
	int party = v/model->gridSizeN;
	int gridn = v%model->gridSizeN;
	int partn = 256 * partx + party;
	int id = gridm * model->gridSizeN + gridn;
	return prob->floatR[partn*model->gridSizeM*model->gridSizeN + id];

}
//-------------------------------------------------------
tmm_double calc_rmse(tmm_problem *prob, tmm_model *model)
{
	printf("calculating rmse ...\n");
	if (prob->nnz == 0)
		return 0;
	tmm_double loss = 0;

	for (tmm_long i = 0; i < prob->nnz; i++)
	{
		tmm_node &N = prob->R[i];
		N.rp = look_up_floatR(N.u, N.v, model, prob);
		tmm_float e = N.r - look_up_floatR(N.u, N.v, model, prob);

		loss += e*e;

		if (i % 100000000 == 0 && i > 0)printf("progress: %%%.3lf, est_RMSE: %.4lf\n", ((double)100.0)*i / prob->nnz, sqrt(loss / (i + 1)));
	}
	return sqrt(loss / prob->nnz);
}
//----------------multiplication-----------------------------
void multiplication(string test_path, const char* model_path)
{
    Parameter para;
    tmm_problem prob = read_problem(test_path);  //"netflix_mme.bin" "netflix_mm.bin" 
    grid_problem(&prob, para);
    tmm_model *model = tmm_load_model(model_path); //"pqmodel_hf.bin"
    if(model == nullptr)
        throw runtime_error("cannot load model from " + string(model_path));
    init_model(model, &prob, para);

    //core
    tmm_update_k128(para, model, &prob);

	auto rmse = calc_rmse(&prob, model);
	cout << fixed << setprecision(4) << "RMSE = " << rmse << endl;

    tmm_destroy_model(&model);
}
//----------------------------------------------------------
#define MByte (1024∗1024) 

int main()
{
	string test_path = "C:/Users/wez078/lab3/Src/lab3/netflix_mme.bin";
	const char* model_path = "C:/Users/wez078/lab3/Src/lab3/pqmodel_hf.bin";

    multiplication(test_path, model_path);

    return 0;
}
