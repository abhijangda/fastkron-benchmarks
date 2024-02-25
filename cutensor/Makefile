CXX_FLAGS=-std=c++11 -I${CUTENSOR_ROOT}/include -L${CUTENSOR_ROOT}/lib -lcutensor -lcudart

all:
	nvcc contraction.cu -o  contraction ${CXX_FLAGS}
run:
	./contraction
clean:
	rm -f contraction
