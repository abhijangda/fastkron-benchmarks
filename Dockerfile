FROM nvcr.io/nvidia/pytorch:23.11-py3
RUN apt-get update
RUN pip3 install matplotlib gpytorch 
RUN apt-get install cmake -y
ADD fastkron /fastkron
ADD fastkron-benchmarks /fastkron-benchmarks
