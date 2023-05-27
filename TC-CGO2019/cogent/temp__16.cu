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
#define SIZE_SLICE_1_A1 16
#define SIZE_SLICE_1_A2 6
#define SIZE_SLICE_1_B 16
#define SIZE_SLICE_1_C 6

#define SIZE_INT_UNIT_1 SIZE_SLICE_1_D

#define SIZE_TB_1_X 	SIZE_SLICE_1_A1
#define SIZE_TB_1_Y 	SIZE_SLICE_1_B
#define SIZE_REG_1_X 	SIZE_SLICE_1_A2
#define SIZE_REG_1_Y 	SIZE_SLICE_1_C

#define NUM_INDEX 		4
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

// created by tc_gen_code_Kernel()
__global__ void kernel__1_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a1, int size_a2, int size_b, int size_c, int size_d, 
int numBlk_a1, int numBlk_a2, int numBlk_b, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][96];
	__shared__ double sm_b[16][96];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a1 = threadIdx.x;
	int idx_b = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a2 * numBlk_a1);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a2 * numBlk_a1);

	int blk_idx_b = tmp_blkIdx / (numBlk_a2 * numBlk_a1);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a2 * numBlk_a1);

	int blk_idx_a2 = tmp_blkIdx / numBlk_a1;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a1);

	int  blk_idx_a1 = tmp_blkIdx;

	int t3_base_thread = blk_idx_a1 * SIZE_SLICE_1_A1 + idx_a1 + (blk_idx_a2 * SIZE_SLICE_1_A2 + (blk_idx_b * SIZE_SLICE_1_B + idx_b + (blk_idx_c * SIZE_SLICE_1_C) * size_b) * size_a2) * size_a1;


	double temp_av;
	double temp_bv[6];
	double reg_tile[6][6];

	for (int i = 0; i < 6; i++)
	for (int j = 0; j < 6; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a1', 'a2', 'd']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['b', 'd', 'c']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 6; ll++)
		{
			// ['a1', 'a2', 'd']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a1 < rng_a1
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a1 * SIZE_SLICE_1_A1 + idx_a1 + (blk_idx_a2 * SIZE_SLICE_1_A2 + ll) * size_a1 + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 6; ll++)
		{
			// ['b', 'd', 'c']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a1 < rng_b
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_b * SIZE_SLICE_1_B + idx_a1 + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_d) * size_b + (threadIdx.y + l) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_b + 0];
			temp_bv[1] = sm_b[ll][idx_b + 16];
			temp_bv[2] = sm_b[ll][idx_b + 32];
			temp_bv[3] = sm_b[ll][idx_b + 48];
			temp_bv[4] = sm_b[ll][idx_b + 64];
			temp_bv[5] = sm_b[ll][idx_b + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a1 + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 6
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel__2_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a1, int size_a2, int size_b, int size_c, int size_d, 
int numBlk_a1, int numBlk_a2, int numBlk_b, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][96];
	__shared__ double sm_b[16][96];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a1 = threadIdx.x;
	int idx_b = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a2 * numBlk_a1);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a2 * numBlk_a1);

	int blk_idx_b = tmp_blkIdx / (numBlk_a2 * numBlk_a1);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a2 * numBlk_a1);

	int blk_idx_a2 = tmp_blkIdx / numBlk_a1;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a1);

	int  blk_idx_a1 = tmp_blkIdx;

	int t3_base_thread = blk_idx_a1 * SIZE_SLICE_1_A1 + idx_a1 + (blk_idx_a2 * SIZE_SLICE_1_A2 + (blk_idx_b * SIZE_SLICE_1_B + idx_b + (blk_idx_c * SIZE_SLICE_1_C) * size_b) * size_a2) * size_a1;


	double temp_av;
	double temp_bv[6];
	double reg_tile[6][6];

	for (int i = 0; i < 6; i++)
	for (int j = 0; j < 6; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a1', 'a2', 'd']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['b', 'd', 'c']], '+=']
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
		for (int ll = 0; ll < 6; ll++)
		{
			// ['a1', 'a2', 'd']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a1 < rng_a1
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a1 * SIZE_SLICE_1_A1 + idx_a1 + (blk_idx_a2 * SIZE_SLICE_1_A2 + ll) * size_a1 + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 6; ll++)
		{
			// ['b', 'd', 'c']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a1 < rng_b
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_b * SIZE_SLICE_1_B + idx_a1 + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_d) * size_b + (threadIdx.y + l) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_b + 0];
			temp_bv[1] = sm_b[ll][idx_b + 16];
			temp_bv[2] = sm_b[ll][idx_b + 32];
			temp_bv[3] = sm_b[ll][idx_b + 48];
			temp_bv[4] = sm_b[ll][idx_b + 64];
			temp_bv[5] = sm_b[ll][idx_b + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a1 + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 6
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel__3_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a1, int size_a2, int size_b, int size_c, int size_d, 
int numBlk_a1, int numBlk_a2, int numBlk_b, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][96];
	__shared__ double sm_b[16][96];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a1 = threadIdx.x;
	int idx_b = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a2 * numBlk_a1);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a2 * numBlk_a1);

	int blk_idx_b = tmp_blkIdx / (numBlk_a2 * numBlk_a1);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a2 * numBlk_a1);

	int blk_idx_a2 = tmp_blkIdx / numBlk_a1;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a1);

	int  blk_idx_a1 = tmp_blkIdx;

	int t3_base_thread = blk_idx_a1 * SIZE_SLICE_1_A1 + idx_a1 + (blk_idx_a2 * SIZE_SLICE_1_A2 + (blk_idx_b * SIZE_SLICE_1_B + idx_b + (blk_idx_c * SIZE_SLICE_1_C) * size_b) * size_a2) * size_a1;

	// need to support partial tiles
	int rng_a1, rng_a2, rng_b, rng_c;
	if ((size_a1 - (blk_idx_a1 * SIZE_SLICE_1_A1)) >= SIZE_SLICE_1_A1)
	{
		rng_a1 = SIZE_SLICE_1_A1;
	}
	else
	{
		rng_a1 = size_a1 % SIZE_SLICE_1_A1;
	}
	if ((size_a2 - (blk_idx_a2 * SIZE_SLICE_1_A2)) >= SIZE_SLICE_1_A2)
	{
		rng_a2 = SIZE_SLICE_1_A2;
	}
	else
	{
		rng_a2 = size_a2 % SIZE_SLICE_1_A2;
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

	double temp_av;
	double temp_bv[6];
	double reg_tile[6][6];

	for (int i = 0; i < 6; i++)
	for (int j = 0; j < 6; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a1', 'a2', 'd']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['b', 'd', 'c']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a1 < rng_a1)
		for (int ll = 0; ll < rng_a2; ll++)
		{
			// ['a1', 'a2', 'd']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a1 < rng_a1
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a1 * SIZE_SLICE_1_A1 + idx_a1 + (blk_idx_a2 * SIZE_SLICE_1_A2 + ll) * size_a1 + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a1 < rng_b)
		for (int ll = 0; ll < rng_c; ll++)
		{
			// ['b', 'd', 'c']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a1 < rng_b
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_b * SIZE_SLICE_1_B + idx_a1 + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_d) * size_b + (threadIdx.y + l) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_b + 0];
			temp_bv[1] = sm_b[ll][idx_b + 16];
			temp_bv[2] = sm_b[ll][idx_b + 32];
			temp_bv[3] = sm_b[ll][idx_b + 48];
			temp_bv[4] = sm_b[ll][idx_b + 64];
			temp_bv[5] = sm_b[ll][idx_b + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a1 + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a1 < rng_a1 && idx_b < rng_b)
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			if(i < rng_c && j < rng_a2)
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
int size_a1, int size_a2, int size_b, int size_c, int size_d, 
int numBlk_a1, int numBlk_a2, int numBlk_b, int numBlk_c, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][96];
	__shared__ double sm_b[16][96];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a1 = threadIdx.x;
	int idx_b = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_c = blockIdx.x / (numBlk_b * numBlk_a2 * numBlk_a1);
	tmp_blkIdx = blockIdx.x % (numBlk_b * numBlk_a2 * numBlk_a1);

	int blk_idx_b = tmp_blkIdx / (numBlk_a2 * numBlk_a1);
	tmp_blkIdx = tmp_blkIdx % (numBlk_a2 * numBlk_a1);

	int blk_idx_a2 = tmp_blkIdx / numBlk_a1;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a1);

	int  blk_idx_a1 = tmp_blkIdx;

	int t3_base_thread = blk_idx_a1 * SIZE_SLICE_1_A1 + idx_a1 + (blk_idx_a2 * SIZE_SLICE_1_A2 + (blk_idx_b * SIZE_SLICE_1_B + idx_b + (blk_idx_c * SIZE_SLICE_1_C) * size_b) * size_a2) * size_a1;

	// need to support partial tiles
	int rng_a1, rng_a2, rng_b, rng_c;
	if ((size_a1 - (blk_idx_a1 * SIZE_SLICE_1_A1)) >= SIZE_SLICE_1_A1)
	{
		rng_a1 = SIZE_SLICE_1_A1;
	}
	else
	{
		rng_a1 = size_a1 % SIZE_SLICE_1_A1;
	}
	if ((size_a2 - (blk_idx_a2 * SIZE_SLICE_1_A2)) >= SIZE_SLICE_1_A2)
	{
		rng_a2 = SIZE_SLICE_1_A2;
	}
	else
	{
		rng_a2 = size_a2 % SIZE_SLICE_1_A2;
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

	double temp_av;
	double temp_bv[6];
	double reg_tile[6][6];

	for (int i = 0; i < 6; i++)
	for (int j = 0; j < 6; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a1', 'a2', 'd']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['b', 'd', 'c']], '+=']
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
		if (idx_a1 < rng_a1 && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_a2; ll++)
		{
			// ['a1', 'a2', 'd']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a1 < rng_a1
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a1 * SIZE_SLICE_1_A1 + idx_a1 + (blk_idx_a2 * SIZE_SLICE_1_A2 + ll) * size_a1 + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a1 < rng_b && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_c; ll++)
		{
			// ['b', 'd', 'c']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a1 < rng_b
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_b * SIZE_SLICE_1_B + idx_a1 + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_d) * size_b + (threadIdx.y + l) * stride_int_v2];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_b + 0];
			temp_bv[1] = sm_b[ll][idx_b + 16];
			temp_bv[2] = sm_b[ll][idx_b + 32];
			temp_bv[3] = sm_b[ll][idx_b + 48];
			temp_bv[4] = sm_b[ll][idx_b + 64];
			temp_bv[5] = sm_b[ll][idx_b + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a1 + (xx * 16)];

				reg_tile[0][xx] += temp_av * temp_bv[0];
				reg_tile[1][xx] += temp_av * temp_bv[1];
				reg_tile[2][xx] += temp_av * temp_bv[2];
				reg_tile[3][xx] += temp_av * temp_bv[3];
				reg_tile[4][xx] += temp_av * temp_bv[4];
				reg_tile[5][xx] += temp_av * temp_bv[5];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a1 < rng_a1 && idx_b < rng_b)
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			if(i < rng_c && j < rng_a2)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void sd_t_d2_fusion(int size_a1, int size_a2, int size_b, int size_c, int size_d, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;


	num_thread_blocks_kernel_1 = CEIL(size_a1, SIZE_SLICE_1_A1) * CEIL(size_a2, SIZE_SLICE_1_A2) * CEIL(size_b, SIZE_SLICE_1_B) * CEIL(size_c, SIZE_SLICE_1_C);
	// cudaMalloc()
	cudaMalloc((void**) &dev_t3, sizeof(double) * size_a1 * size_a2 * size_b * size_c);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_d * size_a2 * size_a1);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_c * size_d * size_b);

	// cudaMemcpy()
	cudaMemcpy(dev_t3, t3, sizeof(double) * size_a1 * size_a2 * size_b * size_c, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_d * size_a2 * size_a1, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_c * size_d * size_b, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
	long long int tmp_operations = 2 * (long long int)(size_a1 * size_a2 * size_b * size_c) * size_d;
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", SIZE_TB_1_X, SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", SIZE_REG_1_X, SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", SIZE_TB_1_X * SIZE_REG_1_X, SIZE_TB_1_Y * SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
	printf ("====================================================================================================\n");
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(SIZE_TB_1_X, SIZE_TB_1_Y);

	int stride_output_a1 = 1;
	int stride_output_a2 = stride_output_a1 * size_a1;
	int stride_output_b = stride_output_a2 * size_a2;
	int stride_output_c = stride_output_b * size_b;

	int stride_reg_x_1 = stride_output_a2;
	int stride_reg_y_1 = stride_output_c;

	int size_internal = size_d;

	int stride_int_t2 = size_a1 * size_a2;
	int stride_int_v2 = size_b;

	// Decision Tree for Kernel Types
	// No Chance to Utilize the Register Transpose
	if (size_a1 % SIZE_SLICE_1_A1 == 0 && size_a2 % SIZE_SLICE_1_A2 == 0 && size_b % SIZE_SLICE_1_B == 0 && size_c % SIZE_SLICE_1_C == 0)
	{
		// [2] Extenral Index: Full
		if (size_d % SIZE_SLICE_1_D == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Full && Internal: Full
			printf ("External: Full, Internal: Full\n");
			kernel__1_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a1, size_a2, size_b, size_c, size_d, CEIL(size_a1, SIZE_SLICE_1_A1), CEIL(size_a2, SIZE_SLICE_1_A2), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Full && Internal: Partial
			printf ("External: Full, Internal: Partial\n");
			kernel__2_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a1, size_a2, size_b, size_c, size_d, CEIL(size_a1, SIZE_SLICE_1_A1), CEIL(size_a2, SIZE_SLICE_1_A2), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
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
			kernel__3_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a1, size_a2, size_b, size_c, size_d, CEIL(size_a1, SIZE_SLICE_1_A1), CEIL(size_a2, SIZE_SLICE_1_A2), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Partial && Internal: Partial
			printf ("External: Partial, Internal: Partial\n");
			kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a1, size_a2, size_b, size_c, size_d, CEIL(size_a1, SIZE_SLICE_1_A1), CEIL(size_a2, SIZE_SLICE_1_A2), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
	}

	// Copy the Result from Device to Host
	cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a1 * size_a2 * size_b * size_c), cudaMemcpyDeviceToHost);

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
	int size_a1;
	int size_a2;

	if (size_a % 52 == 0)
	{
		//
		size_a1 = 52;
		size_a2 = size_a / 52;
	}
	else
	{
		//
		size_a1 = size_a;
		size_a2 = 1;
	}

	// Call An Application
	sd_t_d2_fusion(size_a1, size_a2, size_b, size_c, size_d, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}
