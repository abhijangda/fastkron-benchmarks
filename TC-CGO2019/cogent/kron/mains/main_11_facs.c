
    //
//	Sample Code:
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Initialize t3 (t3_temp), 9 t2 and 9 v2.
void pre_Initializing_Input_Tensors(float* h_C, float* h_C_chk, int size_C, float* h_A, int size_A, float* h_B, int size_B)
{
	// t3
	int i, j;
	for (i = 0; i < size_C; i++)
	{
		h_C[i] 	= 0.0;
		h_C_chk[i] = 0.0;
	}

	for (j = 0; j < size_A; j++)
	{
		h_A[j] = ((float)rand() / RAND_MAX);
	}

	for (j = 0; j < size_B; j++)
	{
		h_B[j] = ((float)rand() / RAND_MAX);
	}
}

//#define DEBUG_CORRECTNESS
//
//# abcdef-gdab-efgc
//t3 [a,16,b,16,c,16,d,16,e,16,f,16] += sum(g,16) * t2 [g,d,a,b] * v2 [e,f,g,c];
//
int main(int argc, char** argv)
{
	// for sd2
	float *host_C, *host_C_chk;
	float *host_A;
	float *host_B;
    	 if (argc != 3) abort();
int size_a = atoi(argv[1]);
int kronRow = atoi(argv[2]);
int size_b = kronRow;
int size_c = kronRow;
int size_d = kronRow;
int size_e = kronRow;
int size_f = kronRow;
int size_g = kronRow;
int size_h = kronRow;
int size_i = kronRow;
int size_j = kronRow;
int size_k = kronRow;
int size_l = kronRow;
int size_m = kronRow;
int size_A = size_a * size_b * size_c * size_d * size_e * size_f * size_g * size_h * size_i * size_j * size_k * size_l ;
host_C 		= (float*)malloc(sizeof(float) * size_A);
host_A 		= (float*)malloc(sizeof(float) * size_A);
host_B 		= (float*)malloc(sizeof(float) * kronRow*kronRow);
pre_Initializing_Input_Tensors(host_C, host_C, size_A, host_A, size_A, host_B, kronRow*kronRow);
sd_t_d2_fusion_(size_a,size_b,size_c,size_d,size_e,size_f,size_g,size_h,size_i,size_j,size_k,size_l,size_m,host_C, host_A, host_B, 1, -1);
return 0;
} 