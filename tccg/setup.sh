#HPTT
CXX=g++ cd hptt && make avx -j
cd ..

#TCL
#sudo apt install intel-mkl
cd tcl && make -j
cd ..

#TCCG
#sudo apt install python2
python2 setup.py install --user

