exe = graph_loader.bin 

# cc = "$(shell which g++)" 
cc = "$(shell which nvcc)"
cupath=$(shell dirname "$(shell which nvcc)")
flags=  -I. -I../.. -I../ -I$(cupath)/../include -O3 -arch=sm_50 -Xptxas -v 
# flags = -I../../lib/ -I. -laio -fopenmp -O3

#flags = -I. -fopenmp -march=athlon64 -O3
#flags = -I. -O3
#flags += -std=c++11

ifeq ($(debug), 1)
	flags+= -DDEBUG 
endif

objs = $(patsubst %.cpp,%.o,$(wildcard ../../lib/*.cpp)) \
			$(patsubst %.cu,%.o,$(wildcard *.cu))

deps = $(wildcard ./*.hpp) \
				$(wildcard *.h) \
				Makefile

%.o:%.cu $(deps)
	$(cc) -c $< -o $@ $(flags)

$(exe):$(objs)
	$(cc) $(objs) -o $(exe) $(flags)

test:$(exe)
	./cpu_ibfs /home/hang/scale_16/beg_16_16.bin /home/hang/scale_16/csr_16_16.bin 64 128 2 

clean:
	rm -rf $(exe) $(objs) 
