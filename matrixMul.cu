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
void tmm_update_k128(Parameter para, tmm_model *model, tmm_problem *prob, float **Rp);
tmm_problem read_problem(string path);
void grid_problem(tmm_problem* prob, Parameter para);
tmm_model* tmm_load_model(char const *path);
void init_model(tmm_model*model, tmm_problem*prob, Parameter para);
void tmm_destroy_model(tmm_model **model);
void transform_R(short *halfR, float *floatR, tmm_problem *prob, Parameter para);
tmm_float look_up_Rp(float **Rp, tmm_int u, tmm_int v, tmm_model *model);
tmm_double calc_rmse(tmm_problem *prob, tmm_model *model, float **Rp);

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

		transform_half << <num_blocks, 256 >> >(gpu_half_R gpu_float_R, prob->gridSizeM*prob->gridSizeN);

		gpuErr(cudaPeekAtLastError());
		cudaMemcpy(float_feature + i*prob->gridSizeM*prob->gridSizeN, gpu_float_R, sizeof(float)*prob->gridSizeM*prob->gridSizeN, cudaMemcpyDeviceToHost);
		gpuErr(cudaPeekAtLastError());
	}

	cudaFree(gpu_half_feature);
	cudaFree(gpu_float_feature);
	gpuErr(cudaPeekAtLastError());
}

//--------------------------core---------------------------------------

void tmm_update_k128(Parameter para, tmm_model *model, tmm_problem *prob, float **Rp)
{   using half_float::half;
    printf("calling tmm_update_k128()...\n");

    //malloc
    cudaMalloc(&model->gpuHalfp, sizeof(half)*model->gridSizeM*model->k);
    cudaMalloc(&model->gpuHalfq, sizeof(half)*model->gridSizeN*model->k);
	cudaMalloc(&prob->gpuR, sizeof(half)*model->gridSizeM*model->gridSizeN);
    //checkCudaErrors("Error allocating device memory arrays");	
	for (int rowTile = 0; rowTile < para.rowScale; rowTile++)
	{
		for (int colTile = 0; colTile< para.colScale; colTile++)
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
			tmm_kernel<<<grid,block>>>(para, model->gridSizeM, model->gridSizeN, model->k, model->gpuHalfp, model->gpuHalfq, prob->gpuR);
			// Copy from GMEM to CPU
			short *R_tmp = prob->halfR + model->gridSizeM*model->gridSizeN*(para.colScale*rowTile+colTile);
			//shortR = malloc(sizeof(short)*model->gridSizeM*model->gridSizeN);
			cudaMemcpy(R_tmp, prob->gpuR, sizeof(half)*model->gridSizeM*model->gridSizeN, cudaMemcpyDeviceToHost);
			//checkCudaErrors("Unable to retrieve result from device");
			// printf("load a R partition and update\n");
	  //       long long idx = 0;
	  //       int partNum = 2 * rowTile + colTile;
	  //       for (long long i = 0; i < model->gridSizeM; i++){
		 //        for (long long j = 0; j < model->gridSizeN; j++){
			//         Rp[partNum][idx] = (float)(shortR[idx]);
			//         idx++;
		 //        }
   //          }

		}
	}
    cudaDeviceSynchronize();
    transform_R(prob->halfR, prob->floatR, prob, para);
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
	printf("m:%lld, n:%lld, nnz:%lld\n", prob.m, prob.n, prob.nnz);
	return prob;
}
//-----------------------------------------------------------------
void grid_problem(tmm_problem* prob, Parameter para) //grid matrix R(gridSizeM, gridSizeN, halfR, floatR)
{
	clock_t start;

	printf("grid problem ...\n");
	fflush(stdout);

	//grid the problem into several grids
	//In our homework, ux = vy = 2, u_seg = m/2, v_seg = n/2
	long long u_seg, v_seg;
	u_seg = (long long)ceil((double)prob->m / para.rowScale);
	v_seg = (long long)ceil((double)prob->n / para.colScale);

	prob->gridSizeM = u_seg;
	prob->gridSizeN = v_seg;

    cudaMallocHost(&prob->halfR, sizeof(short)*prob->m*prob->n);
    cudaMallocHost(&prob->floatR, sizeof(float)*prob->m*prob->n);
	// auto get_grid_id = [=](int u, int v)   // allocate each node of R into the 4 grids
	// {
	// 	return ((u / u_seg) * 2 + v / v_seg);     //return the grid ID
	// };

	// //count the size of each grid
	// long long *gridSize = new long long[para.rowScale*para.colScale]();

	// start = clock();

	// for (long long i = 0; i < prob->nnz; i++)
	// {
	// 	int tmp_u = prob->R[i].u;
	// 	int tmp_v = prob->R[i].v;
	// 	gridSize[get_grid_id(tmp_u, tmp_v)] ++;
	// }
	// prob->gridSize = gridSize;

	// //get max of grid size
	// long long max_grid_size = 0;
	// for (int i = 0; i < para.rowScale*para.colScale; i++)
	// {
	// 	//printf("gridSize[%d]:%lld\n",i,prob->gridSize[i]);
	// 	if (max_grid_size < gridSize[i])max_grid_size = gridSize[i];
	// }
	// prob->maxGridSize = max_grid_size;

	// //generate the pointer to each grid.
	// tmm_node**R2D = new tmm_node*[para.rowScale*para.colScale + 1];
	// tmm_node* R = prob->R;
	// R2D[0] = R;
	// for (int grid = 0; grid < para.rowScale*para.colScale; grid++)R2D[grid + 1] = R2D[grid] + gridSize[grid];

	// prob->R2D = R2D;

	// //swap
	// printf("swapping ...\n");
	// start = clock();
	// tmm_node**pivots = new tmm_node*[4];
	// for (int i = 0; i < 4; i++)pivots[i] = R2D[i];

	// for (int grid = 0; grid < 4; grid++)
	// {
	// 	for (tmm_node*pivot = pivots[grid]; pivot != R2D[grid + 1];)
	// 	{
	// 		int corre_grid = get_grid_id(pivot->u, pivot->v);
	// 		if (corre_grid == grid)
	// 		{
	// 			pivot++;
	// 			continue;
	// 		}
	// 		tmm_node *next = pivots[corre_grid];
	// 		swap(*pivot, *next);
	// 		pivots[corre_grid] ++;
	// 	}
	// }

	// printf("grid swap time:%.8lfs\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	// printf("\n\n");
	// fflush(stdout);
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
    // cudaMallocHost(&model->floatp, sizeof(float)*para.rowScale*model->gridSizeM*model->k);
    // cudaMallocHost(&model->floatq, sizeof(float)*para.colScale*model->gridSizeN*model->k);

    cudaMallocHost(&model->halfp, sizeof(short)*para.rowScale*model->gridSizeM*model->k);
    cudaMallocHost(&model->halfq, sizeof(short)*para.colScale*model->gridSizeN*model->k);

    gpuErr(cudaPeekAtLastError());

    //random init
    // init_feature(model->halfp, model->ux, model->u_seg, model->k);
    // init_feature(model->halfq, model->vy, model->v_seg, model->k);

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
tmm_float look_up_Rp(float **Rp, tmm_int u, tmm_int v, tmm_model *model)
{
	int partx = u/model->gridSizeM;
	int gridm = u%model->gridSizeM;
	int party = v/model->gridSizeN;
	int gridn = v%model->gridSizeN;
	int partn = 2 * partx + party;
	int id = gridm * model->gridSizeN + gridn;
	return Rp[partn][id];

}
//-------------------------------------------------------
tmm_double calc_rmse(tmm_problem *prob, tmm_model *model, float **Rp)
{
	printf("calculating rmse ...\n");
	if (prob->nnz == 0)
		return 0;
	tmm_double loss = 0;

	for (tmm_long i = 0; i < prob->nnz; i++)
	{
		tmm_node &N = prob->R[i];
		N.rp = look_up_Rp(Rp, N.u, N.v, model);
		tmm_float e = N.r - look_up_Rp(Rp, N.u, N.v, model);

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
    float **Rp;
	//Rp = malloc(sizeof(float)*para.rowScale*para.colScale*model->gridSizeM*model->gridSizeN);

    tmm_update_k128(para, model, &prob, Rp);

	auto rmse = calc_rmse(&prob, model, Rp);
	cout << fixed << setprecision(4) << "RMSE = " << rmse << endl;
    

    tmm_destroy_model(&model);
}
//----------------------------------------------------------
#define MByte (1024∗1024) 

int main()
{
	string test_path = "C:/Users/wez078/lab3/Src/lab3/netflix_mme.bin";
	const char* model_path = "C:/Users/wez078/lab3/Src/lab3/pqmodel_hf.bin";
	//try
    //{
        multiplication(test_path, model_path);
    //}
    /*catch(runtime_error &e)
    {
        cout << e.what() << endl;
        return 1;
    }*/
    return 0;
}
