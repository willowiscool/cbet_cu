CXX = g++
NVCC = nvcc
CXXFLAGS = -g -O3 -Wall
NVCCFLAGS = -g -O3
NVCCFLAGS_CC = -Xcompiler -Wall
LDFLAGS = -lm -lhdf5 -lhdf5_cpp -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial

all: main.o utils.o ray_trace.o
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS_CC) -o cbet main.o utils.o ray_trace.o $(LDFLAGS)

main.o: main.cpp consts.hpp structs.hpp utils.cuh ray_trace.cuh
	$(CXX) $(CXXFLAGS) -c main.cpp $(LDFLAGS)

utils.o: utils.cu consts.hpp structs.hpp utils.cuh
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS_CC) -c utils.cu $(LDFLAGS)

ray_trace.o: ray_trace.cu ray_trace.cuh consts.hpp structs.hpp utils.cuh
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGS_CC) -c ray_trace.cu $(LDFLAGS)
