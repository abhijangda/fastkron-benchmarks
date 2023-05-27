// created by tc_code_include() in tc_code_include.py
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <locale.h>
#include <algorithm>
using namespace std;

// created by tc_gen_definition_new()
#define SIZE_SLICE_1_D 16
#define SIZE_SLICE_1_A 16
#define SIZE_SLICE_1_C 4
#define SIZE_SLICE_1_B1 8
#define SIZE_SLICE_1_B2 8

#define SIZE_INT_UNIT_1 SIZE_SLICE_1_D

#define SIZE_TB_1_X 	SIZE_SLICE_1_A
#define SIZE_TB_1_Y 	SIZE_SLICE_1_B1
#define SIZE_REG_1_X 	SIZE_SLICE_1_C
#define SIZE_REG_1_Y 	SIZE_SLICE_1_B2

#define NUM_INDEX 		4
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

// created by tc_gen_code_Kernel()
__global__ void kernel__1_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a, int size_b1, int size_b2, int size_c, int size_d, 
int numBlk_a, int numBlk_b1, int numBlk_b2, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_b1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b2 * numBlk_b1 * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b2 * numBlk_b1 * numBlk_a);

	int blk_idx_b2 = tmp_blkIdx / (numBlk_b1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b1 * numBlk_a);

	int blk_idx_b1 = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b1 * SIZE_SLICE_1_B1 + idx_b1 + (blk_idx_b2 * SIZE_SLICE_1_B2 + (blk_idx_c * SIZE_SLICE_1_C) * size_b2) * size_b1) * size_a;


	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'c', 'd']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'b1', 'b2']], '+=']
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
			// ['a', 'c', 'd']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_c * SIZE_SLICE_1_C + ll) * size_a + (threadIdx.y + l + 0) * stride_int_t2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			// Exception: Full-Full
			sm_a[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_c * SIZE_SLICE_1_C + ll) * size_a + (threadIdx.y + l + 8) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 8; ll++)
		{
			// ['d', 'b1', 'b2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_b1 < rng_b1
			sm_b[threadIdx.x][threadIdx.y + ll * 8] = dev_v2[(blk_idx_b1 * SIZE_SLICE_1_B1 + idx_b1 + (blk_idx_b2 * SIZE_SLICE_1_B2 + ll) * size_b1) * size_d + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_b1 + 0];
			temp_bv[1] = sm_b[ll][idx_b1 + 8];
			temp_bv[2] = sm_b[ll][idx_b1 + 16];
			temp_bv[3] = sm_b[ll][idx_b1 + 24];
			temp_bv[4] = sm_b[ll][idx_b1 + 32];
			temp_bv[5] = sm_b[ll][idx_b1 + 40];
			temp_bv[6] = sm_b[ll][idx_b1 + 48];
			temp_bv[7] = sm_b[ll][idx_b1 + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

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
__global__ void kernel__2_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a, int size_b1, int size_b2, int size_c, int size_d, 
int numBlk_a, int numBlk_b1, int numBlk_b2, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_b1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b2 * numBlk_b1 * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b2 * numBlk_b1 * numBlk_a);

	int blk_idx_b2 = tmp_blkIdx / (numBlk_b1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b1 * numBlk_a);

	int blk_idx_b1 = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b1 * SIZE_SLICE_1_B1 + idx_b1 + (blk_idx_b2 * SIZE_SLICE_1_B2 + (blk_idx_c * SIZE_SLICE_1_C) * size_b2) * size_b1) * size_a;


	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'c', 'd']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'b1', 'b2']], '+=']
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
			// ['a', 'c', 'd']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_c * SIZE_SLICE_1_C + ll) * size_a + (threadIdx.y + l + 0) * stride_int_t2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal) 
			sm_a[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_c * SIZE_SLICE_1_C + ll) * size_a + (threadIdx.y + l + 8) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 8; ll++)
		{
			// ['d', 'b1', 'b2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_b1 < rng_b1
			sm_b[threadIdx.x][threadIdx.y + ll * 8] = dev_v2[(blk_idx_b1 * SIZE_SLICE_1_B1 + idx_b1 + (blk_idx_b2 * SIZE_SLICE_1_B2 + ll) * size_b1) * size_d + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_b1 + 0];
			temp_bv[1] = sm_b[ll][idx_b1 + 8];
			temp_bv[2] = sm_b[ll][idx_b1 + 16];
			temp_bv[3] = sm_b[ll][idx_b1 + 24];
			temp_bv[4] = sm_b[ll][idx_b1 + 32];
			temp_bv[5] = sm_b[ll][idx_b1 + 40];
			temp_bv[6] = sm_b[ll][idx_b1 + 48];
			temp_bv[7] = sm_b[ll][idx_b1 + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

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
__global__ void kernel__3_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a, int size_b1, int size_b2, int size_c, int size_d, 
int numBlk_a, int numBlk_b1, int numBlk_b2, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_b1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b2 * numBlk_b1 * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b2 * numBlk_b1 * numBlk_a);

	int blk_idx_b2 = tmp_blkIdx / (numBlk_b1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b1 * numBlk_a);

	int blk_idx_b1 = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b1 * SIZE_SLICE_1_B1 + idx_b1 + (blk_idx_b2 * SIZE_SLICE_1_B2 + (blk_idx_c * SIZE_SLICE_1_C) * size_b2) * size_b1) * size_a;

	// need to support partial tiles
	int rng_a, rng_b1, rng_b2, rng_c;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_b1 - (blk_idx_b1 * SIZE_SLICE_1_B1)) >= SIZE_SLICE_1_B1)
	{
		rng_b1 = SIZE_SLICE_1_B1;
	}
	else
	{
		rng_b1 = size_b1 % SIZE_SLICE_1_B1;
	}
	if ((size_b2 - (blk_idx_b2 * SIZE_SLICE_1_B2)) >= SIZE_SLICE_1_B2)
	{
		rng_b2 = SIZE_SLICE_1_B2;
	}
	else
	{
		rng_b2 = size_b2 % SIZE_SLICE_1_B2;
	}
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'c', 'd']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'b1', 'b2']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a)
		for (int ll = 0; ll < rng_c; ll++)
		{
			// ['a', 'c', 'd']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_c * SIZE_SLICE_1_C + ll) * size_a + (threadIdx.y + l + 0) * stride_int_t2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (idx_a < rng_a) 
			sm_a[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_c * SIZE_SLICE_1_C + ll) * size_a + (threadIdx.y + l + 8) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_b1 < rng_b1)
		for (int ll = 0; ll < rng_b2; ll++)
		{
			// ['d', 'b1', 'b2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_b1 < rng_b1
			sm_b[threadIdx.x][threadIdx.y + ll * 8] = dev_v2[(blk_idx_b1 * SIZE_SLICE_1_B1 + idx_b1 + (blk_idx_b2 * SIZE_SLICE_1_B2 + ll) * size_b1) * size_d + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_b1 + 0];
			temp_bv[1] = sm_b[ll][idx_b1 + 8];
			temp_bv[2] = sm_b[ll][idx_b1 + 16];
			temp_bv[3] = sm_b[ll][idx_b1 + 24];
			temp_bv[4] = sm_b[ll][idx_b1 + 32];
			temp_bv[5] = sm_b[ll][idx_b1 + 40];
			temp_bv[6] = sm_b[ll][idx_b1 + 48];
			temp_bv[7] = sm_b[ll][idx_b1 + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

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
	if (idx_a < rng_a && idx_b1 < rng_b1)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_b2 && j < rng_c)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel__4_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a, int size_b1, int size_b2, int size_c, int size_d, 
int numBlk_a, int numBlk_b1, int numBlk_b2, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][64];
	__shared__ double sm_b[16][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_b1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b2 * numBlk_b1 * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_b2 * numBlk_b1 * numBlk_a);

	int blk_idx_b2 = tmp_blkIdx / (numBlk_b1 * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b1 * numBlk_a);

	int blk_idx_b1 = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b1 * SIZE_SLICE_1_B1 + idx_b1 + (blk_idx_b2 * SIZE_SLICE_1_B2 + (blk_idx_c * SIZE_SLICE_1_C) * size_b2) * size_b1) * size_a;

	// need to support partial tiles
	int rng_a, rng_b1, rng_b2, rng_c;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_b1 - (blk_idx_b1 * SIZE_SLICE_1_B1)) >= SIZE_SLICE_1_B1)
	{
		rng_b1 = SIZE_SLICE_1_B1;
	}
	else
	{
		rng_b1 = size_b1 % SIZE_SLICE_1_B1;
	}
	if ((size_b2 - (blk_idx_b2 * SIZE_SLICE_1_B2)) >= SIZE_SLICE_1_B2)
	{
		rng_b2 = SIZE_SLICE_1_B2;
	}
	else
	{
		rng_b2 = size_b2 % SIZE_SLICE_1_B2;
	}
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[8][4];

	for (int i = 0; i < 8; i++)
	for (int j = 0; j < 4; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'c', 'd']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'b1', 'b2']], '+=']
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
		if (idx_a < rng_a && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_c; ll++)
		{
			// ['a', 'c', 'd']
			// Exception: Temp. version!: threadIdx.y + l + 0
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y + 0][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_c * SIZE_SLICE_1_C + ll) * size_a + (threadIdx.y + l + 0) * stride_int_t2];
			// Exception: Temp. version!: threadIdx.y + l + 8
			// Exception: Temp. version!: idx_a < rng_a
			if (threadIdx.y + l + 8 < size_internal && idx_a < rng_a) 
			sm_a[threadIdx.y + 8][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_c * SIZE_SLICE_1_C + ll) * size_a + (threadIdx.y + l + 8) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_b1 < rng_b1 && threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b2; ll++)
		{
			// ['d', 'b1', 'b2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_b1 < rng_b1
			sm_b[threadIdx.x][threadIdx.y + ll * 8] = dev_v2[(blk_idx_b1 * SIZE_SLICE_1_B1 + idx_b1 + (blk_idx_b2 * SIZE_SLICE_1_B2 + ll) * size_b1) * size_d + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_b1 + 0];
			temp_bv[1] = sm_b[ll][idx_b1 + 8];
			temp_bv[2] = sm_b[ll][idx_b1 + 16];
			temp_bv[3] = sm_b[ll][idx_b1 + 24];
			temp_bv[4] = sm_b[ll][idx_b1 + 32];
			temp_bv[5] = sm_b[ll][idx_b1 + 40];
			temp_bv[6] = sm_b[ll][idx_b1 + 48];
			temp_bv[7] = sm_b[ll][idx_b1 + 56];

			for (int xx = 0; xx < 4; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

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
	if (idx_a < rng_a && idx_b1 < rng_b1)
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if(i < rng_b2 && j < rng_c)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void sd_t_d2_fusion(int size_a, int size_b1, int size_b2, int size_c, int size_d, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;


	num_thread_blocks_kernel_1 = CEIL(size_a, SIZE_SLICE_1_A) * CEIL(size_b1, SIZE_SLICE_1_B1) * CEIL(size_b2, SIZE_SLICE_1_B2) * CEIL(size_c, SIZE_SLICE_1_C);
	// cudaMalloc()
	cudaMalloc((void**) &dev_t3, sizeof(double) * size_a * size_b1 * size_b2 * size_c);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_d * size_c * size_a);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_b2 * size_b1 * size_d);

	// cudaMemcpy()
	cudaMemcpy(dev_t3, t3, sizeof(double) * size_a * size_b1 * size_b2 * size_c, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_d * size_c * size_a, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_b2 * size_b1 * size_d, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
	long long int tmp_operations = 2 * (long long int)(size_a * size_b1 * size_b2 * size_c) * size_d;
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
	int stride_output_b1 = stride_output_a * size_a;
	int stride_output_b2 = stride_output_b1 * size_b1;
	int stride_output_c = stride_output_b2 * size_b2;

	int stride_reg_x_1 = stride_output_c;
	int stride_reg_y_1 = stride_output_b2;

	int size_internal = size_d;

	int stride_int_t2 = size_a * size_c;
	int stride_int_v2 = 1;

	// Decision Tree for Kernel Types
	// No Chance to Utilize the Register Transpose
	if (size_a % SIZE_SLICE_1_A == 0 && size_b1 % SIZE_SLICE_1_B1 == 0 && size_b2 % SIZE_SLICE_1_B2 == 0 && size_c % SIZE_SLICE_1_C == 0)
	{
		// [2] Extenral Index: Full
		if (size_d % SIZE_SLICE_1_D == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Full && Internal: Full
			printf ("External: Full, Internal: Full\n");
			kernel__1_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b1, size_b2, size_c, size_d, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b1, SIZE_SLICE_1_B1), CEIL(size_b2, SIZE_SLICE_1_B2), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Full && Internal: Partial
			printf ("External: Full, Internal: Partial\n");
			kernel__2_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b1, size_b2, size_c, size_d, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b1, SIZE_SLICE_1_B1), CEIL(size_b2, SIZE_SLICE_1_B2), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
	}
	else
	{
		// [2] Extenral Index: Partial
		if (size_d % SIZE_SLICE_1_D == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Partial && Internal: Full
			printf ("External: Partial, Internal: Full\n");
			kernel__3_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b1, size_b2, size_c, size_d, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b1, SIZE_SLICE_1_B1), CEIL(size_b2, SIZE_SLICE_1_B2), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Partial && Internal: Partial
			printf ("External: Partial, Internal: Partial\n");
			kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b1, size_b2, size_c, size_d, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b1, SIZE_SLICE_1_B1), CEIL(size_b2, SIZE_SLICE_1_B2), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
	}

	// Copy the Result from Device to Host
	cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a * size_b1 * size_b2 * size_c), cudaMemcpyDeviceToHost);

	// cudaFree()
	cudaFree(dev_t3);	cudaFree(dev_t2);	cudaFree(dev_v2);

	// Shoule be Fixed
	// HostFree

}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void sd_t_d2_fusion_(int size_a, int size_b, int size_c, int size_d, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Pre-Processing for Split
	// Based on Tile-Sizes and Problem-Size
	// Currently, one index can be split into two indices
	int size_b1;
	int size_b2;

	if (size_b % 37 == 0)
	{
		//
		size_b1 = 37;
		size_b2 = size_b / 37;
	}
	else
	{
		//
		size_b1 = size_b;
		size_b2 = 1;
	}

	// Call An Application
	sd_t_d2_fusion(size_a, size_b1, size_b2, size_c, size_d, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}
