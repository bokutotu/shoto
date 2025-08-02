CXX = g++
CUDA_PATH ?= /usr/local/cuda
CXXFLAGS = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lnvrtc

all: nvrtc_minimal nvrtc_sample

nvrtc_minimal: nvrtc_minimal.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

nvrtc_sample: nvrtc_sample.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f nvrtc_minimal nvrtc_sample

.PHONY: all clean