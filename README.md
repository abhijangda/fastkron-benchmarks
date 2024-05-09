# fastkron-benchmarks

# Install cuTensor from https://developer.nvidia.com/cutensor-downloads

# HPTT
`
cd tccg/hptt
CXX=g++ make avx -j
`

# TCL
`
cd tccg/tcl
sudo apt install intel-mkl
make
`

# TCCG
`
sudo apt install python2
cd tccg
python2 setup.py install --user

`

Benchmark infrastructure for FastKron. Refer to instructions.pdf.
