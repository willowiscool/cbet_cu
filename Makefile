CXX = g++
CXXFLAGS = -g -O3 -Wall
LDFLAGS = -lm -lhdf5 -lhdf5_cpp -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial

all: main.o utils.o ray_trace.o
	$(CXX) $(CXXFLAGS) -o cbet main.o utils.o ray_trace.o $(LDFLAGS)

main.o: main.cpp consts.hpp structs.hpp utils.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp $(LDFLAGS)

utils.o: utils.cpp consts.hpp structs.hpp utils.hpp
	$(CXX) $(CXXFLAGS) -c utils.cpp $(LDFLAGS)

ray_trace.o: ray_trace.cpp ray_trace.hpp consts.hpp structs.hpp utils.hpp
	$(CXX) $(CXXFLAGS) -c ray_trace.cpp $(LDFLAGS)
