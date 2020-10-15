# CUDA_cyclic_reduction
CUDA_cyclic_reduction


__global__ void CRM_forward(double *a, double *b, double *c, double *f, double *x, int size, int stepSize, int i) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int threadid = index_y * grid_width + index_x;
	int index1, index2, offset;
	double k1, k2;
	if (threadid >= stepSize) return;
	int j = pow(2.0, i + 1)*(threadid + 1) - 1;
	offset = pow(2.0, i);
	index1 = j - offset;
	index2 = j + offset;
	k1 = a[j] / b[index1];
	k2 = c[j] / b[index2];
	if (j == size - 1) {
		k1 = a[j] / b[j - offset];
		b[j] = b[j] - c[j - offset] * k1;
		f[j] = f[j] - f[j - offset] * k1;
		a[j] = -a[j - offset] * k1;
		c[j] = 0.0;
	}
	else {
		k1 = a[j] / b[j - offset];
		k2 = c[j] / b[j + offset];
		b[j] = b[j] - c[j - offset] * k1 - a[j + offset] * k2;
		f[j] = f[j] - f[j - offset] * k1 - f[j + offset] * k2;
		a[j] = -a[j - offset] * k1;
		c[j] = -c[j + offset] * k2;
	}
}
__global__ void CRM_backward(double *a, double *b, double *c, double *f, double *x, int size, int stepSize, int i) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int threadid = index_y * grid_width + index_x;

	int index1, index2, offset;
	if (threadid >= stepSize) return;

	int j = pow(2.0, i + 1)*(threadid + 1) - 1;



	offset = pow(2.0, i);
	index1 = j - offset;
	index2 = j + offset;

	if (j != index1) {
		if (index1 - offset < 0) x[index1] = (f[index1] - c[index1] * x[index1 + offset]) / b[index1];
		else x[index1] = (f[index1] - a[index1] * x[index1 - offset] - c[index1] * x[index1 + offset]) / b[index1];
	}
	if (j != index2) {
		if (index2 + offset >= size) x[index2] = (f[index2] - a[index2] * x[index2 - offset]) / b[index2];
		else x[index2] = (f[index2] - a[index2] * x[index2 - offset] - c[index2] * x[index2 + offset]) / b[index2];
	}
}
__global__ void cr_div(double *b, double *f, double *x, int index) {
	x[index] = f[index] / b[index];
}
//calculate the block size according to size
void calc_dim(int size, dim3 *block, dim3 *grid) {
	if (size<4) { block->x = 1;block->y = 1; }
	else if (size<16) { block->x = 2;block->y = 2; }
	else if (size<64) { block->x = 4;block->y = 4; }
	else if (size<256) { block->x = 8;block->y = 8; }
	else { block->x = 16;block->y = 16; }
	grid->x = (unsigned int)ceil(sqrt((double)size / block->x));
	grid->y = (unsigned int)ceil(sqrt((double)size / block->y));
}
void cyclic_reduction_CUDA(double *a, double *b, double *c, double *f, double *y, int N)
{
	dim3 dimBlock, dimGrid;
	int i;
	double *aa_d = new  double[N];
	double *cc_d = new  double[N];
	double *bb_d = new  double[N];
	double *ff_d = new  double[N];
	double *uu_d = new  double[N];
	// allocate device memory
	cudaMalloc(&aa_d, sizeof(double)*N);
	cudaMalloc(&bb_d, sizeof(double)*N);
	cudaMalloc(&cc_d, sizeof(double)*N);
	cudaMalloc(&ff_d, sizeof(double)*N);
	cudaMalloc(&uu_d, sizeof(double)*N);
	//memory transfers
	cudaMemcpy(aa_d, a, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(bb_d, b, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(cc_d, c, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(ff_d, f, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(uu_d, y, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaFuncSetCacheConfig(CRM_forward, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(CRM_backward, cudaFuncCachePreferL1);

	int step_size;


	for (i = 0;i < log2(N + 1) - 1;i++) {
		step_size = (N - pow(2.0, i + 1)) / pow(2.0, i + 1) + 1;
		calc_dim(step_size, &dimBlock, &dimGrid);

		CRM_forward << <dimGrid, dimBlock >> >(aa_d, bb_d, cc_d, ff_d, uu_d, N, step_size, i);
	}

	cr_div << <1, 1 >> >(bb_d, ff_d, uu_d, (N - 1) / 2);
	cudaDeviceSynchronize();

	int step_size2;
	for (i = log2(N + 1) - 2;i >= 0;i--) {
		step_size2 = (N - pow(2.0, i + 1)) / pow(2.0, i + 1) + 1;
		calc_dim(step_size2, &dimBlock, &dimGrid);
		CRM_backward << <dimGrid, dimBlock >> > (aa_d, bb_d, cc_d, ff_d, uu_d, N, step_size2, i);
	}

	cudaMemcpy(y, uu_d, sizeof(double)*N, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}

