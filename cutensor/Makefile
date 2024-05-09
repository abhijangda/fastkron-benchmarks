CXX_FLAGS=-std=c++11  -lcutensor -lcudart

all:
	nvcc contraction.cu -o  contraction ${CXX_FLAGS}
run:
	./contraction
clean:
	rm -f contraction
