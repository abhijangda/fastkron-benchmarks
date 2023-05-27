kernels/kernel_320_3_16.cu: gen_kernels.py
	python3 gen_kernels.py 320 3 16

run_320_3_16: mains/main_3_facs.c kernels/kernel_320_3_16.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@

kernels/kernel_320_3_32.cu: gen_kernels.py
	python3 gen_kernels.py 320 3 32

run_320_3_32: mains/main_3_facs.c kernels/kernel_320_3_32.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@