dims = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

def create_main(numFacs):
    init_func = """
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
}"""

    main_code = """
    //
//	Sample Code:
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>"""
    main_code += init_func + "\n"
    main_code += """
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
    """

    main_code += f"\t if (argc != {1 + 1 + 1}) abort();\n"
    main_code += f"int size_{dims[0]} = atoi(argv[1]);\n"
    main_code += f"int kronRow = atoi(argv[2]);\n"
    for fac in range(numFacs+1):
        main_code += f"int size_{dims[fac+1]} = kronRow;\n"
	
    size_code = f"size_{dims[0]} * "
    for fac in range(numFacs):
        size_code += f"size_{dims[fac + 1]} * "
    size_code = size_code[:-2]
    main_code += "int size_A = " + size_code + ";\n"
    main_code += "host_C 		= (float*)malloc(sizeof(float) * size_A);\n"
    main_code += "host_A 		= (float*)malloc(sizeof(float) * size_A);\n"
    main_code += f"host_B 		= (float*)malloc(sizeof(float) * kronRow*kronRow);\n"
    main_code += "pre_Initializing_Input_Tensors(host_C, host_C, size_A, host_A, size_A, host_B, kronRow*kronRow);\n"					
    kernel_call_code = "sd_t_d2_fusion_(" + "size_"+dims[0] + ","
    for fac in range(numFacs+1):
        kernel_call_code += f"size_{dims[fac + 1]},"
    kernel_call_code += "host_C, host_A, host_B, 1, -1);\n"
    main_code += kernel_call_code
    main_code += "return 0;\n} "
    return main_code

import sys

for facs in range(2, 20):
    main_code = create_main(facs)
    with open(f"mains/main_{facs}_facs.c", "w") as f:
        f.write(main_code)