kernels/kernel_1024_5_8.cu: gen_kernels.py
	python3 gen_kernels.py 1024 5 8
run_320_5_8: mains/main_5_facs.c kernels/kernel_1024_5_8.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
kernels/kernel_1024_6_8.cu: gen_kernels.py
	python3 gen_kernels.py 1024 6 8
run_320_6_8: mains/main_6_facs.c kernels/kernel_1024_6_8.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
kernels/kernel_16_8_8.cu: gen_kernels.py
	python3 gen_kernels.py 16 8 8
run_16_8_8: mains/main_8_facs.c kernels/kernel_16_8_8.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@


kernels/kernel_1024_4_16.cu: gen_kernels.py
	python3 gen_kernels.py 1024 4 16
run_320_4_16: mains/main_4_facs.c kernels/kernel_1024_4_16.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
kernels/kernel_1024_5_16.cu: gen_kernels.py
	python3 gen_kernels.py 1024 5 16
run_320_5_16: mains/main_5_facs.c kernels/kernel_1024_5_16.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
kernels/kernel_16_6_16.cu: gen_kernels.py
	python3 gen_kernels.py 16 6 16
run_16_6_16: mains/main_6_facs.c kernels/kernel_16_6_16.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@

kernels/kernel_1024_3_32.cu: gen_kernels.py
	python3 gen_kernels.py 1024 3 32
run_320_3_32: mains/main_3_facs.c kernels/kernel_1024_3_32.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
kernels/kernel_1024_4_32.cu: gen_kernels.py
	python3 gen_kernels.py 1024 4 32
run_320_4_32: mains/main_4_facs.c kernels/kernel_1024_4_32.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
kernels/kernel_16_5_32.cu: gen_kernels.py
	python3 gen_kernels.py 16 5 32
run_16_5_32: mains/main_5_facs.c kernels/kernel_16_5_32.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@

kernels/kernel_1024_2_64.cu: gen_kernels.py
	python3 gen_kernels.py 1024 2 64
run_320_2_64: mains/main_2_facs.c kernels/kernel_1024_2_64.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
kernels/kernel_1024_3_64.cu: gen_kernels.py
	python3 gen_kernels.py 1024 3 64
run_320_3_64: mains/main_3_facs.c kernels/kernel_1024_3_64.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
kernels/kernel_16_4_64.cu: gen_kernels.py
	python3 gen_kernels.py 16 4 64
run_16_4_64: mains/main_4_facs.c kernels/kernel_16_4_64.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@


kernels/kernel_1024_2_128.cu: gen_kernels.py
	python3 gen_kernels.py 1024 2 128
run_320_2_128: mains/main_2_facs.c kernels/kernel_1024_2_128.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
kernels/kernel_1024_3_128.cu: gen_kernels.py
	python3 gen_kernels.py 1024 3 128
run_320_3_128: mains/main_3_facs.c kernels/kernel_1024_3_128.cu
	nvcc -O3 -std=c++11 -arch=sm_70 $? -Xptxas "-v " -o $@
