Figure-9.pdf: plots/single-gpu-flops.py
	python3 plots/single-gpu-flops.py single-gpu-flops.csv $@

single-x86-flops.pdf: plots/single-gpu-flops.py
	python3 plots/single-gpu-flops.py single-x86-flops.csv $@

Figure-10.pdf: plots/real-world-benchmarks.py real-world-benchmarks.csv
	python3 plots/real-world-benchmarks.py real-world-benchmarks.csv $@

Figure-11-64.pdf: plots/multi-gpu-flops.py 
	python3 plots/multi-gpu-flops.py multi-gpu-flops-64.csv $@

Figure-11-128.pdf: plots/multi-gpu-flops.py 
	python3 plots/multi-gpu-flops.py multi-gpu-flops-128.csv $@ 

clean:
	rm -f *.pdf