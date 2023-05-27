run_320_3_16: mains/main_3_facs.c kernels/kernel_320_3_16.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@