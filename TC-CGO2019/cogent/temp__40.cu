// created by tc_code_include() in tc_code_include.py
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <locale.h>
#include <algorithm>
using namespace std;

// created by tc_gen_definition_new()
#define SIZE_SLICE_1_E 16
#define SIZE_SLICE_1_A 16
#define SIZE_SLICE_1_B 4
#define SIZE_SLICE_1_C 1
#define SIZE_SLICE_1_D1 8
#define SIZE_SLICE_1_D2 8

#define SIZE_INT_UNIT_1 SIZE_SLICE_1_E

#define SIZE_TB_1_X 	SIZE_SLICE_1_A * SIZE_SLICE_1_C
#define SIZE_TB_1_Y 	SIZE_SLICE_1_D1
#define SIZE_REG_1_X 	SIZE_SLICE_1_B
#define SIZE_REG_1_Y 	SIZE_SLICE_1_D2

#define NUM_INDEX 		5
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

// created by tc_gen_code_Kernel()
__global__ void kernel__1_1(float* dev_t3, 
float* dev_t2, 
float* dev_v2, 
int size_a, int size_d1, int size_d2, int size_b, int size_c, int size_e, 
int numBlk_a, int numBlk_d1, int numBlk_d2, int numBlk_b, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ float sm_a[16][64];
	__shared__ float sm_b[16][64];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_c = threadIdx.x / SIZE_SLICE_1_A;
	int idx_d1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_d2 * numBlk_d1 * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_d2 * numBlk_d1 * numBlk_a);

	int blk_idx_b = tmp_blkIdx / (numBlk_d2 * numBlk_d1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d2 * numBlk_d1 * numBlk_a);

	int blk_idx_d2 = tmp_blkIdx / (numBlk_d1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d1 * numBlk_a);

	int blk_idx_d1 = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_d1 * SIZE_SLICE_1_D1 + idx_d1 + (blk_idx_d2 * SIZE_SLICE_1_D2 + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + idx_c) * size_b) * size_d2) * size_d1) * size_a;


	float temp_av;
	float temp_bv[8];
	float reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'b', 'c', 'e']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['e', 'd1', 'd2']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 4; ll++)
		{
			// ['a', 'b', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_t2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			// Exception: Full-Full
			sm_a[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 8; ll++)
		{
			// ['e', 'd1', 'd2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_d1 < rng_d1
			sm_b[threadIdx.x][threadIdx.y + ll * 8] = dev_v2[(blk_idx_d1 * SIZE_SLICE_1_D1 + idx_d1 + (blk_idx_d2 * SIZE_SLICE_1_D2 + ll) * size_d1) * size_e + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d1 + 0];
			temp_bv[1] = sm_b[ll][idx_d1 + 8];
			temp_bv[2] = sm_b[ll][idx_d1 + 16];
			temp_bv[3] = sm_b[ll][idx_d1 + 24];
			temp_bv[4] = sm_b[ll][idx_d1 + 32];
			temp_bv[5] = sm_b[ll][idx_d1 + 40];
			temp_bv[6] = sm_b[ll][idx_d1 + 48];
			temp_bv[7] = sm_b[ll][idx_d1 + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (idx_c) * SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
				reg_tile[6][xx] += temp_av * temp_bv[6];
				reg_tile[7][xx] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 8
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel__2_1(float* dev_t3, 
float* dev_t2, 
float* dev_v2, 
int size_a, int size_d1, int size_d2, int size_b, int size_c, int size_e, 
int numBlk_a, int numBlk_d1, int numBlk_d2, int numBlk_b, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ float sm_a[16][64];
	__shared__ float sm_b[16][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_c = threadIdx.x / SIZE_SLICE_1_A;
	int idx_d1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_d2 * numBlk_d1 * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_d2 * numBlk_d1 * numBlk_a);

	int blk_idx_b = tmp_blkIdx / (numBlk_d2 * numBlk_d1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d2 * numBlk_d1 * numBlk_a);

	int blk_idx_d2 = tmp_blkIdx / (numBlk_d1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d1 * numBlk_a);

	int blk_idx_d1 = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_d1 * SIZE_SLICE_1_D1 + idx_d1 + (blk_idx_d2 * SIZE_SLICE_1_D2 + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + idx_c) * size_b) * size_d2) * size_d1) * size_a;


	float temp_av;
	float temp_bv[8];
	float reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'b', 'c', 'e']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['e', 'd1', 'd2']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			// ['a', 'b', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_t2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal) 
			sm_a[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 8; ll++)
		{
			// ['e', 'd1', 'd2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_d1 < rng_d1
			sm_b[threadIdx.x][threadIdx.y + ll * 8] = dev_v2[(blk_idx_d1 * SIZE_SLICE_1_D1 + idx_d1 + (blk_idx_d2 * SIZE_SLICE_1_D2 + ll) * size_d1) * size_e + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d1 + 0];
			temp_bv[1] = sm_b[ll][idx_d1 + 8];
			temp_bv[2] = sm_b[ll][idx_d1 + 16];
			temp_bv[3] = sm_b[ll][idx_d1 + 24];
			temp_bv[4] = sm_b[ll][idx_d1 + 32];
			temp_bv[5] = sm_b[ll][idx_d1 + 40];
			temp_bv[6] = sm_b[ll][idx_d1 + 48];
			temp_bv[7] = sm_b[ll][idx_d1 + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (idx_c) * SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
				reg_tile[6][xx] += temp_av * temp_bv[6];
				reg_tile[7][xx] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 8
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel__3_1(float* dev_t3, 
float* dev_t2, 
float* dev_v2, 
int size_a, int size_d1, int size_d2, int size_b, int size_c, int size_e, 
int numBlk_a, int numBlk_d1, int numBlk_d2, int numBlk_b, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ float sm_a[16][64];
	__shared__ float sm_b[16][64];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_c = threadIdx.x / SIZE_SLICE_1_A;
	int idx_d1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_d2 * numBlk_d1 * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_d2 * numBlk_d1 * numBlk_a);

	int blk_idx_b = tmp_blkIdx / (numBlk_d2 * numBlk_d1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d2 * numBlk_d1 * numBlk_a);

	int blk_idx_d2 = tmp_blkIdx / (numBlk_d1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d1 * numBlk_a);

	int blk_idx_d1 = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_d1 * SIZE_SLICE_1_D1 + idx_d1 + (blk_idx_d2 * SIZE_SLICE_1_D2 + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + idx_c) * size_b) * size_d2) * size_d1) * size_a;

	// need to support partial tiles
	int rng_a, rng_d1, rng_d2, rng_b, rng_c;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_d1 - (blk_idx_d1 * SIZE_SLICE_1_D1)) >= SIZE_SLICE_1_D1)
	{
		rng_d1 = SIZE_SLICE_1_D1;
	}
	else
	{
		rng_d1 = size_d1 % SIZE_SLICE_1_D1;
	}
	if ((size_d2 - (blk_idx_d2 * SIZE_SLICE_1_D2)) >= SIZE_SLICE_1_D2)
	{
		rng_d2 = SIZE_SLICE_1_D2;
	}
	else
	{
		rng_d2 = size_d2 % SIZE_SLICE_1_D2;
	}
	if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B)
	{
		rng_b = SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % SIZE_SLICE_1_B;
	}
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}

	float temp_av;
	float temp_bv[8];
	float reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'b', 'c', 'e']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['e', 'd1', 'd2']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && 0 < rng_c)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_t2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (idx_a < rng_a) 
			sm_a[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_d1 < rng_d1)
		for (int ll = 0; ll < rng_d2; ll++)
		{
			// ['e', 'd1', 'd2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_d1 < rng_d1
			sm_b[threadIdx.x][threadIdx.y + ll * 8] = dev_v2[(blk_idx_d1 * SIZE_SLICE_1_D1 + idx_d1 + (blk_idx_d2 * SIZE_SLICE_1_D2 + ll) * size_d1) * size_e + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d1 + 0];
			temp_bv[1] = sm_b[ll][idx_d1 + 8];
			temp_bv[2] = sm_b[ll][idx_d1 + 16];
			temp_bv[3] = sm_b[ll][idx_d1 + 24];
			temp_bv[4] = sm_b[ll][idx_d1 + 32];
			temp_bv[5] = sm_b[ll][idx_d1 + 40];
			temp_bv[6] = sm_b[ll][idx_d1 + 48];
			temp_bv[7] = sm_b[ll][idx_d1 + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (idx_c) * SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
				reg_tile[6][xx] += temp_av * temp_bv[6];
				reg_tile[7][xx] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_c < rng_c && idx_d1 < rng_d1)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_d2 && j < rng_b)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel__4_1(float* dev_t3, 
float* dev_t2, 
float* dev_v2, 
int size_a, int size_d1, int size_d2, int size_b, int size_c, int size_e, 
int numBlk_a, int numBlk_d1, int numBlk_d2, int numBlk_b, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ float sm_a[16][64];
	__shared__ float sm_b[16][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_c = threadIdx.x / SIZE_SLICE_1_A;
	int idx_d1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_d2 * numBlk_d1 * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_d2 * numBlk_d1 * numBlk_a);

	int blk_idx_b = tmp_blkIdx / (numBlk_d2 * numBlk_d1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d2 * numBlk_d1 * numBlk_a);

	int blk_idx_d2 = tmp_blkIdx / (numBlk_d1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d1 * numBlk_a);

	int blk_idx_d1 = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_d1 * SIZE_SLICE_1_D1 + idx_d1 + (blk_idx_d2 * SIZE_SLICE_1_D2 + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + idx_c) * size_b) * size_d2) * size_d1) * size_a;

	// need to support partial tiles
	int rng_a, rng_d1, rng_d2, rng_b, rng_c;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_d1 - (blk_idx_d1 * SIZE_SLICE_1_D1)) >= SIZE_SLICE_1_D1)
	{
		rng_d1 = SIZE_SLICE_1_D1;
	}
	else
	{
		rng_d1 = size_d1 % SIZE_SLICE_1_D1;
	}
	if ((size_d2 - (blk_idx_d2 * SIZE_SLICE_1_D2)) >= SIZE_SLICE_1_D2)
	{
		rng_d2 = SIZE_SLICE_1_D2;
	}
	else
	{
		rng_d2 = size_d2 % SIZE_SLICE_1_D2;
	}
	if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B)
	{
		rng_b = SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % SIZE_SLICE_1_B;
	}
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}

	float temp_av;
	float temp_bv[8];
	float reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'b', 'c', 'e']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['e', 'd1', 'd2']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && 0 < rng_c && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'b', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_a + (threadIdx.y + l + 0) * stride_int_t2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal && idx_a < rng_a) 
			sm_a[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_a + (threadIdx.y + l + 8) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_d1 < rng_d1 && threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_d2; ll++)
		{
			// ['e', 'd1', 'd2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_d1 < rng_d1
			sm_b[threadIdx.x][threadIdx.y + ll * 8] = dev_v2[(blk_idx_d1 * SIZE_SLICE_1_D1 + idx_d1 + (blk_idx_d2 * SIZE_SLICE_1_D2 + ll) * size_d1) * size_e + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d1 + 0];
			temp_bv[1] = sm_b[ll][idx_d1 + 8];
			temp_bv[2] = sm_b[ll][idx_d1 + 16];
			temp_bv[3] = sm_b[ll][idx_d1 + 24];
			temp_bv[4] = sm_b[ll][idx_d1 + 32];
			temp_bv[5] = sm_b[ll][idx_d1 + 40];
			temp_bv[6] = sm_b[ll][idx_d1 + 48];
			temp_bv[7] = sm_b[ll][idx_d1 + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (idx_c) * SIZE_SLICE_1_A + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
				reg_tile[6][xx] += temp_av * temp_bv[6];
				reg_tile[7][xx] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_c < rng_c && idx_d1 < rng_d1)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_d2 && j < rng_b)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void sd_t_d2_fusion(int size_a, int size_d1, int size_d2, int size_b, int size_c, int size_e, float* t3, float* host_t2, float* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	float* dev_t3;
	float* dev_t2;
	float* dev_v2;


	num_thread_blocks_kernel_1 = CEIL(size_a, SIZE_SLICE_1_A) * CEIL(size_d1, SIZE_SLICE_1_D1) * CEIL(size_d2, SIZE_SLICE_1_D2) * CEIL(size_b, SIZE_SLICE_1_B) * CEIL(size_c, SIZE_SLICE_1_C);
	// cudaMalloc()
	cudaMalloc((void**) &dev_t3, sizeof(float) * size_a * size_d1 * size_d2 * size_b * size_c);
	cudaMalloc((void**) &dev_t2, sizeof(float) * size_e * size_c * size_b * size_a);
	cudaMalloc((void**) &dev_v2, sizeof(float) * size_d2 * size_d1 * size_e);

	// cudaMemcpy()
	cudaMemcpy(dev_t3, t3, sizeof(float) * size_a * size_d1 * size_d2 * size_b * size_c, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(float) * size_e * size_c * size_b * size_a, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(float) * size_d2 * size_d1 * size_e, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
	long long int tmp_operations = (long long int)(size_a * size_d1 * size_d2 * size_b * size_c) * size_e;
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", SIZE_TB_1_X, SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", SIZE_REG_1_X, SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", SIZE_TB_1_X * SIZE_REG_1_X, SIZE_TB_1_Y * SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
	printf ("====================================================================================================\n");
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(SIZE_TB_1_X, SIZE_TB_1_Y);

	int stride_output_a = 1;
	int stride_output_d1 = stride_output_a * size_a;
	int stride_output_d2 = stride_output_d1 * size_d1;
	int stride_output_b = stride_output_d2 * size_d2;
	int stride_output_c = stride_output_b * size_b;

	int stride_reg_x_1 = stride_output_b;
	int stride_reg_y_1 = stride_output_d2;

	int size_internal = size_e;

	int stride_int_t2 = size_a * size_b * size_c;
	int stride_int_v2 = 1;

	for (int i = 0; i < 100; i++) {
	// Decision Tree for Kernel Types
	// No Chance to Utilize the Register Transpose
	if (size_a % SIZE_SLICE_1_A == 0 && size_d1 % SIZE_SLICE_1_D1 == 0 && size_d2 % SIZE_SLICE_1_D2 == 0 && size_b % SIZE_SLICE_1_B == 0 && size_c % SIZE_SLICE_1_C == 0)
	{
		// [2] Extenral Index: Full
		if (size_e % SIZE_SLICE_1_E == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Full && Internal: Full
			printf ("External: Full, Internal: Full\n");
			kernel__1_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_d1, size_d2, size_b, size_c, size_e, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_d1, SIZE_SLICE_1_D1), CEIL(size_d2, SIZE_SLICE_1_D2), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Full && Internal: Partial
			printf ("External: Full, Internal: Partial\n");
			kernel__2_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_d1, size_d2, size_b, size_c, size_e, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_d1, SIZE_SLICE_1_D1), CEIL(size_d2, SIZE_SLICE_1_D2), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
	}
	else
	{
		// [2] Extenral Index: Partial
		if (size_e % SIZE_SLICE_1_E == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Partial && Internal: Full
			printf ("External: Partial, Internal: Full\n");
			kernel__3_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_d1, size_d2, size_b, size_c, size_e, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_d1, SIZE_SLICE_1_D1), CEIL(size_d2, SIZE_SLICE_1_D2), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Partial && Internal: Partial
			printf ("External: Partial, Internal: Partial\n");
			kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_d1, size_d2, size_b, size_c, size_e, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_d1, SIZE_SLICE_1_D1), CEIL(size_d2, SIZE_SLICE_1_D2), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
	}

	}
	// Copy the Result from Device to Host
	cudaMemcpy(t3, dev_t3, sizeof(float) * (size_a * size_d1 * size_d2 * size_b * size_c), cudaMemcpyDeviceToHost);

	// cudaFree()
	cudaFree(dev_t3);	cudaFree(dev_t2);	cudaFree(dev_v2);

	// Shoule be Fixed
	// HostFree

}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void sd_t_d2_fusion_(int size_a, int size_d, int size_b, int size_c, int size_e, float* t3, float* t2, float* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Pre-Processing for Split
	// Based on Tile-Sizes and Problem-Size
	// Currently, one index can be split into two indices
	int size_d1;
	int size_d2;

	if (size_d % 8 == 0)
	{
		//
		size_d1 = 8;
		size_d2 = size_d / 8;
	}
	else
	{
		//
		size_d1 = size_d;
		size_d2 = 1;
	}

	// Call An Application
	sd_t_d2_fusion(size_a, size_d1, size_d2, size_b, size_c, size_e, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}
