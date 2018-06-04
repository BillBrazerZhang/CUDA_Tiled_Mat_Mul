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
tmm_problem read_problem(string path)  //load matrix R
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
		tmm_float e = N.r - look_up_Rp(Rp, N.u, N.v, model);

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
