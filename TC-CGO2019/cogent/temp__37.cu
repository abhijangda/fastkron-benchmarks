// created by tc_code_include() in tc_code_include.py
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <locale.h>
#include <algorithm>
using namespace std;

// created by tc_gen_definition_new()
#define SIZE_SLICE_1_G 16
#define SIZE_SLICE_1_A 16
#define SIZE_SLICE_1_E 6
#define SIZE_SLICE_1_F 1
#define SIZE_SLICE_1_D 16
#define SIZE_SLICE_1_B 6
#define SIZE_SLICE_1_C 1

#define SIZE_INT_UNIT_1 SIZE_SLICE_1_G

#define SIZE_TB_1_X 	SIZE_SLICE_1_A * SIZE_SLICE_1_F
#define SIZE_TB_1_Y 	SIZE_SLICE_1_D * SIZE_SLICE_1_C
#define SIZE_REG_1_X 	SIZE_SLICE_1_E
#define SIZE_REG_1_Y 	SIZE_SLICE_1_B

#define NUM_INDEX 		6
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

// created by tc_gen_code_Kernel()
__global__ void kernel__1_1(double* dev_t3, 
double* dev_t2, 
double* dev_v2, 
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, int numBlk_e, int numBlk_f, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][96];
	__shared__ double sm_b[16][96];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_f = threadIdx.x / SIZE_SLICE_1_A;
	int idx_d = threadIdx.y % SIZE_SLICE_1_D;
	int idx_c = threadIdx.y / SIZE_SLICE_1_D;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_d = tmp_blkIdx / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + idx_c + (blk_idx_d * SIZE_SLICE_1_D + idx_d + (blk_idx_e * SIZE_SLICE_1_E + (blk_idx_f * SIZE_SLICE_1_F + idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;


	double temp_av;
	double temp_bv[6];
	double reg_tile[6][6];

	for (int i = 0; i < 6; i++)
	for (int j = 0; j < 6; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['e', 'f', 'g', 'a']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['g', 'd', 'b', 'c']], '+=']
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
			// ['e', 'f', 'g', 'a']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: 0 < rng_f
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_e * SIZE_SLICE_1_E + ll + (blk_idx_f * SIZE_SLICE_1_F + 0 + ((blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_g) * size_f) * size_e + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 6; ll++)
		{
			// ['g', 'd', 'b', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_d < rng_d
			sm_b[threadIdx.x][threadIdx.y + ll * 16] = dev_v2[(blk_idx_d * SIZE_SLICE_1_D + idx_d + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_d) * size_g + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 0];
			temp_bv[1] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 16];
			temp_bv[2] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 32];
			temp_bv[3] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 48];
			temp_bv[4] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 64];
			temp_bv[5] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_f + (idx_a) * SIZE_SLICE_1_F + (xx * 16)];

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
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, int numBlk_e, int numBlk_f, 
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
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_f = threadIdx.x / SIZE_SLICE_1_A;
	int idx_d = threadIdx.y % SIZE_SLICE_1_D;
	int idx_c = threadIdx.y / SIZE_SLICE_1_D;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_d = tmp_blkIdx / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + idx_c + (blk_idx_d * SIZE_SLICE_1_D + idx_d + (blk_idx_e * SIZE_SLICE_1_E + (blk_idx_f * SIZE_SLICE_1_F + idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;


	double temp_av;
	double temp_bv[6];
	double reg_tile[6][6];

	for (int i = 0; i < 6; i++)
	for (int j = 0; j < 6; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['e', 'f', 'g', 'a']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['g', 'd', 'b', 'c']], '+=']
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
			// ['e', 'f', 'g', 'a']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: 0 < rng_f
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_e * SIZE_SLICE_1_E + ll + (blk_idx_f * SIZE_SLICE_1_F + 0 + ((blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_g) * size_f) * size_e + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 6; ll++)
		{
			// ['g', 'd', 'b', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_d < rng_d
			sm_b[threadIdx.x][threadIdx.y + ll * 16] = dev_v2[(blk_idx_d * SIZE_SLICE_1_D + idx_d + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_d) * size_g + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 0];
			temp_bv[1] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 16];
			temp_bv[2] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 32];
			temp_bv[3] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 48];
			temp_bv[4] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 64];
			temp_bv[5] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_f + (idx_a) * SIZE_SLICE_1_F + (xx * 16)];

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
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, int numBlk_e, int numBlk_f, 
int stride_int_t2, int stride_int_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ double sm_a[16][96];
	__shared__ double sm_b[16][96];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_f = threadIdx.x / SIZE_SLICE_1_A;
	int idx_d = threadIdx.y % SIZE_SLICE_1_D;
	int idx_c = threadIdx.y / SIZE_SLICE_1_D;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_d = tmp_blkIdx / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + idx_c + (blk_idx_d * SIZE_SLICE_1_D + idx_d + (blk_idx_e * SIZE_SLICE_1_E + (blk_idx_f * SIZE_SLICE_1_F + idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c, rng_d, rng_e, rng_f;
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
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}
	if ((size_d - (blk_idx_d * SIZE_SLICE_1_D)) >= SIZE_SLICE_1_D)
	{
		rng_d = SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % SIZE_SLICE_1_D;
	}
	if ((size_e - (blk_idx_e * SIZE_SLICE_1_E)) >= SIZE_SLICE_1_E)
	{
		rng_e = SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % SIZE_SLICE_1_E;
	}
	if ((size_f - (blk_idx_f * SIZE_SLICE_1_F)) >= SIZE_SLICE_1_F)
	{
		rng_f = SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % SIZE_SLICE_1_F;
	}

	double temp_av;
	double temp_bv[6];
	double reg_tile[6][6];

	for (int i = 0; i < 6; i++)
	for (int j = 0; j < 6; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['e', 'f', 'g', 'a']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['g', 'd', 'b', 'c']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (0 < rng_f && idx_a < rng_a)
		for (int ll = 0; ll < rng_e; ll++)
		{
			// ['e', 'f', 'g', 'a']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: 0 < rng_f
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_e * SIZE_SLICE_1_E + ll + (blk_idx_f * SIZE_SLICE_1_F + 0 + ((blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_g) * size_f) * size_e + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_d < rng_d && 0 < rng_c)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['g', 'd', 'b', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_d < rng_d
			sm_b[threadIdx.x][threadIdx.y + ll * 16] = dev_v2[(blk_idx_d * SIZE_SLICE_1_D + idx_d + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_d) * size_g + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 0];
			temp_bv[1] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 16];
			temp_bv[2] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 32];
			temp_bv[3] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 48];
			temp_bv[4] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 64];
			temp_bv[5] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_f + (idx_a) * SIZE_SLICE_1_F + (xx * 16)];

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
	if (idx_a < rng_a && idx_f < rng_f && idx_d < rng_d && idx_c < rng_c)
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			if(i < rng_b && j < rng_e)
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
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, int numBlk_e, int numBlk_f, 
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
	// # of indices mapped on TB_X: 2
	// # of indices mapped on TB_Y: 2
	int idx_a = threadIdx.x % SIZE_SLICE_1_A;
	int idx_f = threadIdx.x / SIZE_SLICE_1_A;
	int idx_d = threadIdx.y % SIZE_SLICE_1_D;
	int idx_c = threadIdx.y / SIZE_SLICE_1_D;

	int tmp_blkIdx;
	int blk_idx_f = blockIdx.x / (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_e * numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_e = tmp_blkIdx / (numBlk_d * numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_d * numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_d = tmp_blkIdx / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + idx_c + (blk_idx_d * SIZE_SLICE_1_D + idx_d + (blk_idx_e * SIZE_SLICE_1_E + (blk_idx_f * SIZE_SLICE_1_F + idx_f) * size_e) * size_d) * size_c) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c, rng_d, rng_e, rng_f;
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
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}
	if ((size_d - (blk_idx_d * SIZE_SLICE_1_D)) >= SIZE_SLICE_1_D)
	{
		rng_d = SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % SIZE_SLICE_1_D;
	}
	if ((size_e - (blk_idx_e * SIZE_SLICE_1_E)) >= SIZE_SLICE_1_E)
	{
		rng_e = SIZE_SLICE_1_E;
	}
	else
	{
		rng_e = size_e % SIZE_SLICE_1_E;
	}
	if ((size_f - (blk_idx_f * SIZE_SLICE_1_F)) >= SIZE_SLICE_1_F)
	{
		rng_f = SIZE_SLICE_1_F;
	}
	else
	{
		rng_f = size_f % SIZE_SLICE_1_F;
	}

	double temp_av;
	double temp_bv[6];
	double reg_tile[6][6];

	for (int i = 0; i < 6; i++)
	for (int j = 0; j < 6; j++)
	reg_tile[i][j] = 0.0;

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['e', 'f', 'g', 'a']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['g', 'd', 'b', 'c']], '+=']
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
		if (0 < rng_f && idx_a < rng_a && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_e; ll++)
		{
			// ['e', 'f', 'g', 'a']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: 0 < rng_f
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_e * SIZE_SLICE_1_E + ll + (blk_idx_f * SIZE_SLICE_1_F + 0 + ((blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_g) * size_f) * size_e + (threadIdx.y + l) * stride_int_t2];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_d < rng_d && 0 < rng_c && threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['g', 'd', 'b', 'c']
			// Exception: Temp. version!: threadIdx.x + l
			// Exception: Temp. version!: idx_d < rng_d
			sm_b[threadIdx.x][threadIdx.y + ll * 16] = dev_v2[(blk_idx_d * SIZE_SLICE_1_D + idx_d + (blk_idx_b * SIZE_SLICE_1_B + ll + (blk_idx_c * SIZE_SLICE_1_C + 0) * size_b) * size_d) * size_g + (threadIdx.x + l)];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 0];
			temp_bv[1] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 16];
			temp_bv[2] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 32];
			temp_bv[3] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 48];
			temp_bv[4] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 64];
			temp_bv[5] = sm_b[ll][idx_d + (idx_c) * SIZE_SLICE_1_D + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_f + (idx_a) * SIZE_SLICE_1_F + (xx * 16)];

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
	if (idx_a < rng_a && idx_f < rng_f && idx_d < rng_d && idx_c < rng_c)
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			if(i < rng_b && j < rng_e)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// written by tc_interface.tc_gen_code_interface_Header()
extern "C"
void sd_t_d2_fusion(int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g, double* t3, double* host_t2, double* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	double* dev_t3;
	double* dev_t2;
	double* dev_v2;


	num_thread_blocks_kernel_1 = CEIL(size_a, SIZE_SLICE_1_A) * CEIL(size_b, SIZE_SLICE_1_B) * CEIL(size_c, SIZE_SLICE_1_C) * CEIL(size_d, SIZE_SLICE_1_D) * CEIL(size_e, SIZE_SLICE_1_E) * CEIL(size_f, SIZE_SLICE_1_F);
	// cudaMalloc()
	cudaMalloc((void**) &dev_t3, sizeof(double) * size_a * size_b * size_c * size_d * size_e * size_f);
	cudaMalloc((void**) &dev_t2, sizeof(double) * size_a * size_g * size_f * size_e);
	cudaMalloc((void**) &dev_v2, sizeof(double) * size_c * size_b * size_d * size_g);

	// cudaMemcpy()
	cudaMemcpy(dev_t3, t3, sizeof(double) * size_a * size_b * size_c * size_d * size_e * size_f, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t2, host_t2, sizeof(double) * size_a * size_g * size_f * size_e, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v2, host_v2, sizeof(double) * size_c * size_b * size_d * size_g, cudaMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
	long long int tmp_operations = 2 * (long long int)(size_a * size_b * size_c * size_d * size_e * size_f) * size_g;
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
	int stride_output_c = stride_output_b * size_b;
	int stride_output_d = stride_output_c * size_c;
	int stride_output_e = stride_output_d * size_d;
	int stride_output_f = stride_output_e * size_e;

	int stride_reg_x_1 = stride_output_e;
	int stride_reg_y_1 = stride_output_b;

	int size_internal = size_g;

	int stride_int_t2 = size_e * size_f;
	int stride_int_v2 = 1;

	// Decision Tree for Kernel Types
	// No Chance to Utilize the Register Transpose
	if (size_a % SIZE_SLICE_1_A == 0 && size_b % SIZE_SLICE_1_B == 0 && size_c % SIZE_SLICE_1_C == 0 && size_d % SIZE_SLICE_1_D == 0 && size_e % SIZE_SLICE_1_E == 0 && size_f % SIZE_SLICE_1_F == 0)
	{
		// [2] Extenral Index: Full
		if (size_g % SIZE_SLICE_1_G == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Full && Internal: Full
			printf ("External: Full, Internal: Full\n");
			kernel__1_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, size_g, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), CEIL(size_e, SIZE_SLICE_1_E), CEIL(size_f, SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Full && Internal: Partial
			printf ("External: Full, Internal: Partial\n");
			kernel__2_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, size_g, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), CEIL(size_e, SIZE_SLICE_1_E), CEIL(size_f, SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
	}
	else
	{
		// [2] Extenral Index: Partial
		if (size_g % SIZE_SLICE_1_G == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Partial && Internal: Full
			printf ("External: Partial, Internal: Full\n");
			kernel__3_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, size_g, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), CEIL(size_e, SIZE_SLICE_1_E), CEIL(size_f, SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Partial && Internal: Partial
			printf ("External: Partial, Internal: Partial\n");
			kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, size_g, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), CEIL(size_e, SIZE_SLICE_1_E), CEIL(size_f, SIZE_SLICE_1_F), stride_int_t2, stride_int_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
		}
	}

	// Copy the Result from Device to Host
	cudaMemcpy(t3, dev_t3, sizeof(double) * (size_a * size_b * size_c * size_d * size_e * size_f), cudaMemcpyDeviceToHost);

	// cudaFree()
	cudaFree(dev_t3);	cudaFree(dev_t2);	cudaFree(dev_v2);

	// Shoule be Fixed
	// HostFree

}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void sd_t_d2_fusion_(int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, int size_g, double* t3, double* t2, double* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Pre-Processing for Split
	// Based on Tile-Sizes and Problem-Size
	// Currently, one index can be split into two indices

	// Call An Application
	sd_t_d2_fusion(size_a, size_b, size_c, size_d, size_e, size_f, size_g, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}
