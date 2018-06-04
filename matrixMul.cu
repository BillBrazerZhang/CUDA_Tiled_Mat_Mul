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
#include "matrixMul.h"
using namespace std;
using namespace tmm;


//------------------------------------------------------------Host Functions-------------------------------------------------
//----------------------------------------------------------
tmm_problem load_R(half *gpuR, tmm_node *R, int rowTile, int colTile, tmm_model *model)  //load matrix R
{
	//A simple function that reads the sparse matrix in COO manner.
	printf("load and update output R\n");
	for (int i = 0; i < model->gridSizeM; i++)
		for (int j = 0; j < model->gridSizeN; j++){
			R[]rowTile*model->gridSizeM
	
}
//-----------------------------------------
tmm_model* tmm_load_model(char const *path)  // load feature matrix P, Q
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
//-------------------------------------------
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
tmm_float tmm_predict(tmm_model const *model, tmm_int u, tmm_int v)
{
	using half_float::half;

	if (u < 0 || u >= model->m || v < 0 || v >= model->n)
		return model->b;

	half *ph = (half*)model->P + ((tmm_long)u)*model->k;
	half *qh = (half*)model->Q + ((tmm_long)v)*model->k;

	float p, q;
	tmm_float z = 0.0f;
	for (int w = 0; w < model->k; w++) {
		p = (float)(*ph);
		q = (float)(*qh);
		z += p*q;
		ph++;
		qh++;
	}

	if (isnan(z))
		z = model->b;

	if (model->fun == P_L2_tmmC &&
		model->fun == P_L1_tmmC &&
		model->fun == P_LR_tmmC)
		z = z > 0.0f ? 1.0f : -1.0f;

	return z;
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
		tmm_float e = N.r - tmm_predict(model, N.u, N.v);

		loss += e*e;

		if (i % 100000000 == 0 && i > 0)printf("progress: %%%.3lf, est_RMSE: %.4lf\n", ((double)100.0)*i / prob->nnz, sqrt(loss / (i + 1)));
	}
	return sqrt(loss / prob->nnz);
}
//----------------multiplication-----------------------------
void multiplication(string test_path, const char* model_path)
{
    tmm_problem prob = read_problem(test_path);  //"netflix_mme.bin" "netflix_mm.bin" 
    tmm_model *model = tmm_load_model(model_path); //"pqmodel_hf.bin"
    if(model == nullptr)
        throw runtime_error("cannot load model from " + string(model_path));

    //core
    tmm(model);

	auto rmse = calc_rmse(&prob, model);
	cout << fixed << setprecision(4) << "RMSE = " << rmse << endl;
    

    tmm_destroy_model(&model);
}
//----------------------------------------------------------
#define MByte (1024∗1024) 
void checkGpuSize()
{
	size_t remainMem, totalMem;
	cudaMemGetInfo(&remainMem, &totalMem);
	printf("GPU total memoroy: %d MB, remaining memory: %d MB\n", totalMem/MByte, remainMem/MByte);
}

int main()
{
	string test_path = "C:/Users/wez078/lab2/pj1/class_labs/Src/lab2/netflix_mme.bin";
	const char* model_path = "C:/Users/wez078/lab2/pj1/class_labs/Src/lab2/init_pqmodel_hf.bin";
	try
    {
        multiplication(test_path, model_path);
    }
    catch(runtime_error &e)
    {
        cout << e.what() << endl;
        return 1;
    }
    return 0;
}
