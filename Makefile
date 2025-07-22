HDF5_LD_FLAG := $(shell pkg-config --libs --cflags hdf5)
CXX = mpicc
NVCC = nvcc
CXXFLAGS = -g -O3 -Wall -fopenmp
NVCCFLAGS = -g -O3 -lineinfo
NVCCFLAGS_CC = -Xcompiler -Wall -Xcompiler -fopenmp
LDFLAGS = -lm -lhdf5 -lhdf5_cpp -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial

all: main.o utils.o ray_trace.o cbet.o
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS_CC) -o cbet main.o utils.o ray_trace.o cbet.o $(LDFLAGS) $(HDF5_LD_FLAG)

main.o: main.cpp consts.hpp structs.hpp utils.cuh ray_trace.cuh
	$(CXX) $(CXXFLAGS) -c main.cpp $(LDFLAGS) $(HDF5_LD_FLAG)

utils.o: utils.cu consts.hpp structs.hpp utils.cuh
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS_CC) -c utils.cu $(LDFLAGS)

ray_trace.o: ray_trace.cu ray_trace.hpp ray_trace.cuh consts.hpp structs.hpp utils.cuh
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS_CC) -c ray_trace.cu $(LDFLAGS)

cbet.o: cbet.cu cbet.hpp cbet.cuh structs.hpp consts.hpp
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS_CC) -c cbet.cu $(LDFLAGS)
