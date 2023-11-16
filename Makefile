all: small-singlegpu large-singlegpu large-8GPUs batch_size-8GPUs kernel_selection-small kernel_selection-large

small-singlegpu: small_batch_size_float_and_double.py
	python3 small_batch_size_float_and_double.py results-M\=16.csv results-M\=16-double.csv 4 64 $@.pdf ms False

# large-singlegpu-double: max_d.py
# 	python3 max_d_only_speedup.py results-M\=16.csv 4 64 $@.pdf ms False

# large-singlegpu: large-kron-4-16-yaxis.pdf large-kron-8-16.pdf large-kron-16-16.pdf large-kron-32-16.pdf

# large-kron-%-16-yaxis.pdf: singlegpu.py
# 	python3 singlegpu.py results-M\=320.csv $* 4 1

# large-kron-%-16.pdf: singlegpu.py
# 	python3 singlegpu.py results-M\=320.csv $* 4 0

single-gpu-flops.pdf: plots/single-gpu-flops.py
	python3 plots/single-gpu-flops.py single-gpu-flops.csv $@

real-world-benchmarks.pdf: plots/real-world-benchmarks.py real-world-benchmarks.csv
	python3 plots/real-world-benchmarks.py real-world-benchmarks.csv $@

multi-gpu-64-4.pdf: plots/multi-gpu-flops.py 
	python3 plots/multi-gpu-flops.py multi-gpu-flops-64.csv $@

multi-gpu-128-4.pdf: plots/multi-gpu-flops.py 
	python3 plots/multi-gpu-flops.py multi-gpu-flops-128.csv $@ 

weak-scaling: weak-scaling-64-4.pdf weak-scaling-128-4.pdf

large-8GPUs: max_d_only_speedup.py
	python3 max_d_only_speedup.py 4 64 $@.pdf s results-N\=320-8GPU.csv results-N\=320-4GPU.csv results-N\=320-2GPU.csv results-N\=320-1GPU.csv

# batch_size-8GPUs: increasing_batch_size.py
# 	python3 increasing_batch_size.py $@.pdf s results-increasingN-8GPU.csv results-increasingN-4GPU.csv results-increasingN-2GPU.csv results-increasingN-1GPU.csv

gld_requests: gld_requests.py
	python3 gld_requests.py profile.csv $@.pdf

kernel_selection-small: non_power_two.py
	python3 non_power_two.py results-non-power-two-N=16.csv 4 64 $@.pdf s True

kernel_selection-large: non_power_two.py
	python3 non_power_two.py results-non-power-two-N=320.csv 4 64 $@.pdf s True

clean:
	rm -f *.pdf