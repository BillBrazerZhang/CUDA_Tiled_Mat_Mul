//------------------------------------------------------------Kernel Functions-----------------------------------------------
void tmm_update_k128(Parameter para, tmm_model *model, half *gpuR, short *R, float *Rp)
{   using half_float::half;
    printf("calling tmm_update_k128()...\n");

    //malloc
    cudaMalloc(&model->gpuHalfp, sizeof(half)*model->gridSizeM*model->k);
    cudaMalloc(&model->gpuHalfq, sizeof(half)*model->gridSizeN*model->k);
	cudaMalloc(&gpuR, sizeof(half)*model->gridSizeM*model->gridSizeN);
    checkCUDAError("Error allocating device memory arrays");	
	model->gridSizeM = model->m/para.rowScale + 1;
	model->gridSizeN = model->n/para.colScale + 1;
	for (int rowTile = 0; rowTile < para.rowScale; rowTile++)
	{
		for (int colTile = 0; colTile< para.colScale; colTile++)
		{
			short *p_tmp = model->halfp + model->gridSizeM*model->k*model->rowTile;
            short *q_tmp = model->halfq + model->gridSizeN*model->k*model->colTile;
			assert(p_tmp);
			assert(q_tmp); 
			// Copy from CPU to GMEM
			cudaMemcpy(&model->gpuHalfp, p_tmp, sizeof(half)*model->gridSizeM*model->k, cudaMemcpyHostToDevice);
            cudaMemcpy(&model->gpuHalfq, q_tmp, sizeof(half)*model->gridSizeN*model->k, cudaMemcpyHostToDevice);
			
			// Dim Configuration
            dim3 block(32, 32);
			dim3 grid((model->gridSizeN+block.x-1)/block.x, (model->gridSizeM+block.y-1)/block.y);
			tmm_kernel<<<grid,block>>>(para, model->gridSizeM, model->gridSizeN, model->k, gpuHalfp, gpuHalfq, gpuR);
			// Copy from GMEM to CPU
			short *R;
			cudaMemcpy(R, gpuR, sizeof(half)*model->gridSizeM*model->gridSizeN, cudaMemcpyDeviceToHost);
			checkCUDAError("Unable to retrieve result from device");
			printf("load a R partition and update\n");
	        long long idx = 0;
	        int partNum = 2 * rowTile + colTile;
	        for (int i = 0; i < model->gridSizeM; i++){
		        for (int j = 0; j < model->gridSizeN; j++){
			        Rp[partNum][idx] = (float)(R[idx]);
			        idx++;
		        }
            }
			cudaFree(gpuHalfp);
			cudaFree(gpuHalfq);
			cudaFree(gpuR);
		}
	}

}

void update_Rp(half *R, float **Rp, int rowTile, int colTile, tmm_model *model)  //load matrix R
{
	
}

__global__ void tmm_kernel(Parameter para, tmm_int a_seg, tmm_int b_seg, tmm_int k, half *gpuHalfp, half *gpuHalfq, half *gpuR)
{
    //from GMEM to SMEM
    const unsigned int bx = blockDim.x, by = blockDim.y;
    const unsigned int tx = threadIdx.x, ty = threadIdx.y;
    const unsigned int I = blockIdx.y*by + ty, J = blockIdx.x*bx + tx; // row and col
    const unsigned int t = para.tile;
    __shared__ half aTile[t][t], bTile[t][t];
    float c = 0.0f;     
	for (unsigned int k1=0; k1 < (k+t-1)/t; k1++)
	{
		if (I < a_seg && k1*t+tx < k){
			aTile[ty][tx] = __lgd(&gpuHalfp[I*k + k1*t + tx]);
		}
		if (J < b_seg && k1*t+ty < k){
			bTile[ty][tx] = __lgd(&gpuHalfq[J*k + k1*t + ty]);
		}
		__syncthreads(); // Synchronizes all threads in a block	
		for (unsigned int k2=0; k2< t; k2++)
            c += __half2float(aTile[ty][k2])*__half2float(bTile[k2][tx]);
        __syncthreads(); // Avoids memory hazards
	}
	if (I < a_seg && J < b_seg)
        gpuR[I*a_seg + J] = __float2half(c);
}
