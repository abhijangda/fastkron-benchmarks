CXX_FLAGS=-std=c++17 -I${HIPTENSOR_ROOT}/include -L${HIPTENSOR_ROOT}/lib -lhiptensor

all:
	hipcc contraction.cu -o  contraction ${CXX_FLAGS}
run:
	./contraction
clean:
	rm -f contraction
