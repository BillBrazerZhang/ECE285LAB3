#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>


#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



namespace tmm
{
	int const kALIGNByte = 4;
    int const kALIGN = kALIGNByte/sizeof(float);
	

	typedef float tmm_float;
	typedef double tmm_double;
	typedef long long tmm_int;
	typedef long long tmm_long;
    
    struct Parameter
    {
    int tile;
	int rowScale;
	int colScale;
    Parameter():tile(32), rowScale(10), colScale(2){}
    };

	struct tmm_node
	{
		tmm_int u;
		tmm_int v;
		tmm_float r;
		tmm_float rp;
	};

	struct tmm_model
	{
		tmm_int m;
		tmm_int n;
		tmm_int k;
	    short *P;
		short *Q;
		//tmm_float b;
		long long gridSizeM;
		long long gridSizeN;
        short *halfp;
        short *halfq;
        half *gpuHalfp;
        half *gpuHalfq;
	};

	struct tmm_problem
	{
		tmm_int m;
		tmm_int n;
		tmm_long nnz;
		long long gridSizeM;
		long long gridSizeN;
        short *halfR;
        float *floatR;
		struct tmm_node *R;	
		// struct mf_node **R2D;
        // long long *gridSize;
        // long long maxGridSize;
		half *gpuR;
	};


    template <typename T> T* malloc_aligned_float(tmm_long size)
    {
	    tmm_int const kALIGNByte = 32;
	    tmm_int const kALIGN = kALIGNByte / sizeof(T);

	    void *ptr;
    #ifdef _WIN32
	    ptr = _aligned_malloc(size * sizeof(T), kALIGNByte);
	    if (ptr == nullptr)
		    throw bad_alloc();
    #else
	    int status = posix_memalign(&ptr, kALIGNByte, size * sizeof(T));
	    if (status != 0)
		    throw bad_alloc();
    #endif

	    return (T*)ptr;
    }

}