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
#endif


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
    Parameter():tile(32), rowScale(4), colScale(2){}
    };

	struct tmm_node
	{
		tmm_int u;
		tmm_int v;
		tmm_float r;
	};

	struct tmm_model
	{
		tmm_int fun;
		tmm_int m;
		tmm_int n;
		tmm_int k;
		tmm_float b;
		float *floatp;
        float *floatq;
        short *halfp;
        short *halfq;
		short *P;
		short *Q;
		long long u_seg, v_seg;
        half *gpuHalfp;
        half *gpuHalfq;
        int cur_u_id;
        int cur_v_id;

        //half *gpuHalfPptrs[2];// allocate P for GPU
        //half *gpuHalfQptrs[2];// allocate Q for GPU

        //int cur_global_x_id[2];//-1
        //int cur_global_y_id[2];//-1
	};

	struct tmm_problem
	{
		tmm_int m;
		tmm_int n;
		tmm_long nnz;
		struct tmm_node *R;
		struct tmm_node **R2D;
		long long u_seg, v_seg;
        long long *gridSize;
        long long maxGridSize;	
        struct tmm_node *gpuR;
        int cur_u_id;
        int cur_v_id;

        //struct tmm_node *gpuRptrs[2];
        //int cur_global_x_id[2];
        //int cur_global_y_id[2];		
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