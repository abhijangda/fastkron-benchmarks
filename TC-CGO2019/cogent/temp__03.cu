// created by tc_code_include() in tc_code_include.py
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <locale.h>
#include <algorithm>
using namespace std;

// created by tc_gen_definition_new()
#define SIZE_SLICE_1_E 8
#define SIZE_SLICE_1_A 8
#define SIZE_SLICE_1_D 8
#define SIZE_SLICE_1_B 1
#define SIZE_SLICE_1_C1 16
#define SIZE_SLICE_1_C2 4

#define SIZE_INT_UNIT_1 SIZE_SLICE_1_E

#define SIZE_TB_1_X 	SIZE_SLICE_1_A * SIZE_SLICE_1_B
#define SIZE_TB_1_Y 	SIZE_SLICE_1_C1
#define SIZE_REG_1_X 	SIZE_SLICE_1_D
#define SIZE_REG_1_Y 	SIZE_SLICE_1_C2

#define NUM_INDEX 		5
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

// created by tc_gen_code_Kernel()
__global__ void kernel__1_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a, int size_b, int size_c1, int size_c2, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c1, int numBlk_c2, int numBlk_d, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[8][64];
	__shared__ double sm_b[8][64];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_b = threadIdx.x / SIZE_SLICE_1_A;
	int idx_c1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c2 * numBlk_c1 * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c2 * numBlk_c1 * numBlk_b * numBlk_a);

	int blk_idx_c2 = tmp_blkIdx / (numBlk_c1 * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c1 * numBlk_b * numBlk_a);

	int blk_idx_c1 = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + idx_b + (blk_idx_c1 * SIZE_SLICE_1_C1 + idx_c1 + (blk_idx_c2 * SIZE_SLICE_1_C2 + (blk_idx_d * SIZE_SLICE_1_D) * size_c2) * size_c1) * size_b) * size_a;


	double temp_av;
	double temp_bv[8];
	double reg_tile[4][8];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 8; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['d', 'b', 'e', 'a']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['e', 'c1', 'c2']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.y < 8 && threadIdx.y < 8)
		for (int ll = 0; ll < 8; ll++)
		{
			// ['d', 'b', 'e', 'a']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: 0 < rng_b
			sm_a[threadIdx.y][threadIdx.x + ll * 8] = dev_t2[blk_idx_d * SIZE_SLICE_1_D + ll + (blk_idx_b * SIZE_SLICE_1_B + 0 + ((blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_e) * size_b) * size_d + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 4; ll++)
		{
			// ['e', 'c1', 'c2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_c1 < rng_c1
			sm_b[threadIdx.x][threadIdx.y + ll * 16] = dev_v2[(blk_idx_c1 * SIZE_SLICE_1_C1 + idx_c1 + (blk_idx_c2 * SIZE_SLICE_1_C2 + ll) * size_c1) * size_e + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 0];
			temp_bv[1] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 8];
			temp_bv[2] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 16];
			temp_bv[3] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 24];
			temp_bv[4] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 32];
			temp_bv[5] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 40];
			temp_bv[6] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 48];
			temp_bv[7] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 56];

			for (int yy = 0; yy < 4; yy++) // (2)
			{
				temp_av = sm_b[ll][idx_c1 + (yy * 16)];

				reg_tile[yy][0] += temp_av * temp_bv[0];
				reg_tile[yy][1] += temp_av * temp_bv[1];
				reg_tile[yy][2] += temp_av * temp_bv[2];
				reg_tile[yy][3] += temp_av * temp_bv[3];
				reg_tile[yy][4] += temp_av * temp_bv[4];
				reg_tile[yy][5] += temp_av * temp_bv[5];
				reg_tile[yy][6] += temp_av * temp_bv[6];
				reg_tile[yy][7] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel__2_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a, int size_b, int size_c1, int size_c2, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c1, int numBlk_c2, int numBlk_d, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[8][64];
	__shared__ double sm_b[8][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_b = threadIdx.x / SIZE_SLICE_1_A;
	int idx_c1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c2 * numBlk_c1 * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c2 * numBlk_c1 * numBlk_b * numBlk_a);

	int blk_idx_c2 = tmp_blkIdx / (numBlk_c1 * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c1 * numBlk_b * numBlk_a);

	int blk_idx_c1 = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + idx_b + (blk_idx_c1 * SIZE_SLICE_1_C1 + idx_c1 + (blk_idx_c2 * SIZE_SLICE_1_C2 + (blk_idx_d * SIZE_SLICE_1_D) * size_c2) * size_c1) * size_b) * size_a;


	double temp_av;
	double temp_bv[8];
	double reg_tile[4][8];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 8; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['d', 'b', 'e', 'a']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['e', 'c1', 'c2']], '+=']
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
		if (threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound && threadIdx.y < 8 && threadIdx.y < 8)
		for (int ll = 0; ll < 8; ll++)
		{
			// ['d', 'b', 'e', 'a']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: 0 < rng_b
			sm_a[threadIdx.y][threadIdx.x + ll * 8] = dev_t2[blk_idx_d * SIZE_SLICE_1_D + ll + (blk_idx_b * SIZE_SLICE_1_B + 0 + ((blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_e) * size_b) * size_d + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 4; ll++)
		{
			// ['e', 'c1', 'c2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_c1 < rng_c1
			sm_b[threadIdx.x][threadIdx.y + ll * 16] = dev_v2[(blk_idx_c1 * SIZE_SLICE_1_C1 + idx_c1 + (blk_idx_c2 * SIZE_SLICE_1_C2 + ll) * size_c1) * size_e + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 0];
			temp_bv[1] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 8];
			temp_bv[2] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 16];
			temp_bv[3] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 24];
			temp_bv[4] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 32];
			temp_bv[5] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 40];
			temp_bv[6] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 48];
			temp_bv[7] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 56];

			for (int yy = 0; yy < 4; yy++) // (2)
			{
				temp_av = sm_b[ll][idx_c1 + (yy * 16)];

				reg_tile[yy][0] += temp_av * temp_bv[0];
				reg_tile[yy][1] += temp_av * temp_bv[1];
				reg_tile[yy][2] += temp_av * temp_bv[2];
				reg_tile[yy][3] += temp_av * temp_bv[3];
				reg_tile[yy][4] += temp_av * temp_bv[4];
				reg_tile[yy][5] += temp_av * temp_bv[5];
				reg_tile[yy][6] += temp_av * temp_bv[6];
				reg_tile[yy][7] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// created by tc_gen_code_Kernel()
__global__ void kernel__3_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a, int size_b, int size_c1, int size_c2, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c1, int numBlk_c2, int numBlk_d, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[8][64];
	__shared__ double sm_b[8][64];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_b = threadIdx.x / SIZE_SLICE_1_A;
	int idx_c1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c2 * numBlk_c1 * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c2 * numBlk_c1 * numBlk_b * numBlk_a);

	int blk_idx_c2 = tmp_blkIdx / (numBlk_c1 * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c1 * numBlk_b * numBlk_a);

	int blk_idx_c1 = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + idx_b + (blk_idx_c1 * SIZE_SLICE_1_C1 + idx_c1 + (blk_idx_c2 * SIZE_SLICE_1_C2 + (blk_idx_d * SIZE_SLICE_1_D) * size_c2) * size_c1) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c1, rng_c2, rng_d;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B)
	{
		rng_b = SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % SIZE_SLICE_1_B;
	}
	if ((size_c1 - (blk_idx_c1 * SIZE_SLICE_1_C1)) >= SIZE_SLICE_1_C1)
	{
		rng_c1 = SIZE_SLICE_1_C1;
	}
	else
	{
		rng_c1 = size_c1 % SIZE_SLICE_1_C1;
	}
	if ((size_c2 - (blk_idx_c2 * SIZE_SLICE_1_C2)) >= SIZE_SLICE_1_C2)
	{
		rng_c2 = SIZE_SLICE_1_C2;
	}
	else
	{
		rng_c2 = size_c2 % SIZE_SLICE_1_C2;
	}
	if ((size_d - (blk_idx_d * SIZE_SLICE_1_D)) >= SIZE_SLICE_1_D)
	{
		rng_d = SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % SIZE_SLICE_1_D;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[4][8];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 8; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['d', 'b', 'e', 'a']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['e', 'c1', 'c2']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (0 < rng_b && idx_a < rng_a && threadIdx.y < 8 && threadIdx.y < 8)
		for (int ll = 0; ll < rng_d; ll++)
		{
			// ['d', 'b', 'e', 'a']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: 0 < rng_b
			sm_a[threadIdx.y][threadIdx.x + ll * 8] = dev_t2[blk_idx_d * SIZE_SLICE_1_D + ll + (blk_idx_b * SIZE_SLICE_1_B + 0 + ((blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_e) * size_b) * size_d + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_c1 < rng_c1)
		for (int ll = 0; ll < rng_c2; ll++)
		{
			// ['e', 'c1', 'c2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_c1 < rng_c1
			sm_b[threadIdx.x][threadIdx.y + ll * 16] = dev_v2[(blk_idx_c1 * SIZE_SLICE_1_C1 + idx_c1 + (blk_idx_c2 * SIZE_SLICE_1_C2 + ll) * size_c1) * size_e + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 0];
			temp_bv[1] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 8];
			temp_bv[2] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 16];
			temp_bv[3] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 24];
			temp_bv[4] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 32];
			temp_bv[5] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 40];
			temp_bv[6] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 48];
			temp_bv[7] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 56];

			for (int yy = 0; yy < 4; yy++) // (2)
			{
				temp_av = sm_b[ll][idx_c1 + (yy * 16)];

				reg_tile[yy][0] += temp_av * temp_bv[0];
				reg_tile[yy][1] += temp_av * temp_bv[1];
				reg_tile[yy][2] += temp_av * temp_bv[2];
				reg_tile[yy][3] += temp_av * temp_bv[3];
				reg_tile[yy][4] += temp_av * temp_bv[4];
				reg_tile[yy][5] += temp_av * temp_bv[5];
				reg_tile[yy][6] += temp_av * temp_bv[6];
				reg_tile[yy][7] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_b < rng_b && idx_c1 < rng_c1)
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if(i < rng_c2 && j < rng_d)
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
int size_a, int size_b, int size_c1, int size_c2, int size_d, int size_e, 
int numBlk_a, int numBlk_b, int numBlk_c1, int numBlk_c2, int numBlk_d, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[8][64];
	__shared__ double sm_b[8][64];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_b = threadIdx.x / SIZE_SLICE_1_A;
	int idx_c1 = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c2 * numBlk_c1 * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c2 * numBlk_c1 * numBlk_b * numBlk_a);

	int blk_idx_c2 = tmp_blkIdx / (numBlk_c1 * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c1 * numBlk_b * numBlk_a);

	int blk_idx_c1 = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + idx_b + (blk_idx_c1 * SIZE_SLICE_1_C1 + idx_c1 + (blk_idx_c2 * SIZE_SLICE_1_C2 + (blk_idx_d * SIZE_SLICE_1_D) * size_c2) * size_c1) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c1, rng_c2, rng_d;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B)
	{
		rng_b = SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % SIZE_SLICE_1_B;
	}
	if ((size_c1 - (blk_idx_c1 * SIZE_SLICE_1_C1)) >= SIZE_SLICE_1_C1)
	{
		rng_c1 = SIZE_SLICE_1_C1;
	}
	else
	{
		rng_c1 = size_c1 % SIZE_SLICE_1_C1;
	}
	if ((size_c2 - (blk_idx_c2 * SIZE_SLICE_1_C2)) >= SIZE_SLICE_1_C2)
	{
		rng_c2 = SIZE_SLICE_1_C2;
	}
	else
	{
		rng_c2 = size_c2 % SIZE_SLICE_1_C2;
	}
	if ((size_d - (blk_idx_d * SIZE_SLICE_1_D)) >= SIZE_SLICE_1_D)
	{
		rng_d = SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % SIZE_SLICE_1_D;
	}

	double temp_av;
	double temp_bv[8];
	double reg_tile[4][8];

	for (int i = 0; i < 4; i++)
	for (int j = 0; j < 8; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['d', 'b', 'e', 'a']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['e', 'c1', 'c2']], '+=']
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
		if (0 < rng_b && idx_a < rng_a && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound && threadIdx.y < 8 && threadIdx.y < 8)
		for (int ll = 0; ll < rng_d; ll++)
		{
			// ['d', 'b', 'e', 'a']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: 0 < rng_b
			sm_a[threadIdx.y][threadIdx.x + ll * 8] = dev_t2[blk_idx_d * SIZE_SLICE_1_D + ll + (blk_idx_b * SIZE_SLICE_1_B + 0 + ((blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_e) * size_b) * size_d + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_c1 < rng_c1 && threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_c2; ll++)
		{
			// ['e', 'c1', 'c2']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_c1 < rng_c1
			sm_b[threadIdx.x][threadIdx.y + ll * 16] = dev_v2[(blk_idx_c1 * SIZE_SLICE_1_C1 + idx_c1 + (blk_idx_c2 * SIZE_SLICE_1_C2 + ll) * size_c1) * size_e + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 0];
			temp_bv[1] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 8];
			temp_bv[2] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 16];
			temp_bv[3] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 24];
			temp_bv[4] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 32];
			temp_bv[5] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 40];
			temp_bv[6] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 48];
			temp_bv[7] = sm_a[ll][idx_a + (idx_b) * SIZE_SLICE_1_A + 56];

			for (int yy = 0; yy < 4; yy++) // (2)
			{
				temp_av = sm_b[ll][idx_c1 + (yy * 16)];

				reg_tile[yy][0] += temp_av * temp_bv[0];
				reg_tile[yy][1] += temp_av * temp_bv[1];
				reg_tile[yy][2] += temp_av * temp_bv[2];
				reg_tile[yy][3] += temp_av * temp_bv[3];
				reg_tile[yy][4] += temp_av * temp_bv[4];
				reg_tile[yy][5] += temp_av * temp_bv[5];
				reg_tile[yy][6] += temp_av * temp_bv[6];
				reg_tile[yy][7] += temp_av * temp_bv[7];
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_b < rng_b && idx_c1 < rng_c1)
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if(i < rng_c2 && j < rng_d)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void sd_t_d2_fusion(int size_a, int size_b, int size_c1, int size_c2, int size_d, int size_e, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;


	num_thread_blocks_kernel_1 = CEIL(size_a, SIZE_SLICE_1_A) * CEIL(size_b, SIZE_SLICE_1_B) * CEIL(size_c1, SIZE_SLICE_1_C1) * CEIL(size_c2, SIZE_SLICE_1_C2) * CEIL(size_d, SIZE_SLICE_1_D);
	// cudaMalloc()
	cudaMalloc((void**) &dev_t3, sizeof(double) * size_a * size_b * size_c1 * size_c2 * size_d);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_a * size_e * size_b * size_d);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_c2 * size_c1 * size_e);

	// cudaMemcpy()
	cudaMemcpy(dev_t3, t3, sizeof(double) * size_a * size_b * size_c1 * size_c2 * size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_a * size_e * size_b * size_d, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_c2 * size_c1 * size_e, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
	long long int tmp_operations = 2 * (long long int)(size_a * size_b * size_c1 * size_c2 * size_d) * size_e;
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
	int stride_output_b = stride_output_a * size_a;
	int stride_output_c1 = stride_output_b * size_b;
	int stride_output_c2 = stride_output_c1 * size_c1;
	int stride_output_d = stride_output_c2 * size_c2;

	int stride_reg_x_1 = stride_output_d;
	int stride_reg_y_1 = stride_output_c2;

	int size_internal = size_e;

	int stride_int_t2 = size_d * size_b;
	int stride_int_v2 = 1;

	// Decision Tree for Kernel Types
	// No Chance to Utilize the Register Transpose
	if (size_a % SIZE_SLICE_1_A == 0 && size_b % SIZE_SLICE_1_B == 0 && size_c1 % SIZE_SLICE_1_C1 == 0 && size_c2 % SIZE_SLICE_1_C2 == 0 && size_d % SIZE_SLICE_1_D == 0)
	{
		// [2] Extenral Index: Full
		if (size_e % SIZE_SLICE_1_E == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Full && Internal: Full
			printf ("External: Full, Internal: Full\n");
			kernel__1_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c1, size_c2, size_d, size_e, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c1, SIZE_SLICE_1_C1), CEIL(size_c2, SIZE_SLICE_1_C2), CEIL(size_d, SIZE_SLICE_1_D), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Full && Internal: Partial
			printf ("External: Full, Internal: Partial\n");
			kernel__2_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c1, size_c2, size_d, size_e, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c1, SIZE_SLICE_1_C1), CEIL(size_c2, SIZE_SLICE_1_C2), CEIL(size_d, SIZE_SLICE_1_D), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
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
			kernel__3_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c1, size_c2, size_d, size_e, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c1, SIZE_SLICE_1_C1), CEIL(size_c2, SIZE_SLICE_1_C2), CEIL(size_d, SIZE_SLICE_1_D), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Partial && Internal: Partial
			printf ("External: Partial, Internal: Partial\n");
			kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c1, size_c2, size_d, size_e, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c1, SIZE_SLICE_1_C1), CEIL(size_c2, SIZE_SLICE_1_C2), CEIL(size_d, SIZE_SLICE_1_D), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
	}

	// Copy the Result from Device to Host
	cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a * size_b * size_c1 * size_c2 * size_d), cudaMemcpyDeviceToHost);

	// cudaFree()
	cudaFree(dev_t3);	cudaFree(dev_t2);	cudaFree(dev_v2);

	// Shoule be Fixed
	// HostFree

}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void sd_t_d2_fusion_(int size_a, int size_b, int size_c, int size_d, int size_e, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Pre-Processing for Split
	// Based on Tile-Sizes and Problem-Size
	// Currently, one index can be split into two indices
	int size_c1;
	int size_c2;

	if (size_c % 12 == 0)
	{
		//
		size_c1 = 12;
		size_c2 = size_c / 12;
	}
	else
	{
		//
		size_c1 = size_c;
		size_c2 = 1;
	}

	// Call An Application
	sd_t_d2_fusion(size_a, size_b, size_c1, size_c2, size_d, size_e, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}
