CXX = g++
CXXFLAGS = -g -O3 -Wall
LDFLAGS = -lm -lhdf5 -lhdf5_cpp

all: main.o mesh.o beam.o ray_trace.o
	$(CXX) $(CXXFLAGS) -o cbet main.o mesh.o beam.o ray_trace.o $(LDFLAGS)

main.o: main.cpp mesh.hpp consts.hpp main.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp $(LDFLAGS)

mesh.o: mesh.cpp mesh.hpp consts.hpp
	$(CXX) $(CXXFLAGS) -c mesh.cpp $(LDFLAGS)

beam.o: beam.cpp beam.hpp consts.hpp
	$(CXX) $(CXXFLAGS) -c beam.cpp $(LDFLAGS)

ray_trace.o: ray_trace.cpp ray_trace.hpp consts.hpp
	$(CXX) $(CXXFLAGS) -c ray_trace.cpp $(LDFLAGS)
