
INCLUDES = -I. -I/usr/local/cuda/include
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
CYTHON = cython
#CXXFLAGS = -Wall -g -shared -pthread -fPIC -I/usr/include/python2.7
#NVCCFLAGS = -g -rdc=true -arch=sm_20
NVCCFLAGS = -O3 -Xcompiler -fPIC -Xptxas -dlcm=ca -arch=sm_30
CXXFLAGS = -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7
CXXLINKFLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart
PYLIB = -lpython2.7
ODIR=obj
BINDIR=bin
_BIN=md
_SO=md3dsys.so

CPPSRCS = $(wildcard *.cpp)
CPPOBJS = $(CPPSRCS:%.cpp=%.o)

CUSRCS = $(wildcard *.cu)
CUOBJS = $(CUSRCS:%.cu=%.o)

#PYXSRCS = $(wildcard *.pyx)
#PYXBJS = $(PYXSRCS:%.pyx=%.o)

$(info "-------cpp objects--------")
$(info $(CPPOBJS))
$(info "-------cu objects---------")
$(info $(CUOBJS))
$(info "-------pyx objects---------")
$(info $(PYXBJS))

#_OBJS = IntegratorKernels.o md1.o PairInteractions.o Properties.o System.o Integrator.o PairInteractionsKernels.o PropertiesKernels.o SystemKernels.o
_OBJS = $(CUOBJS) $(CPPOBJS) $(PYXBJS)
OBJS = $(patsubst %, $(ODIR)/%,$(_OBJS))
$(info "----------objects--------")
$(info $(OBJS))

BIN= $(patsubst %, $(BINDIR)/%,$(_BIN))
$(info "----------binary---------")
$(info $(BIN))

SO= $(patsubst %, $(BINDIR)/%,$(_SO))
$(info "----------library---------")
$(info $(SO))

#$(ODIR)/%.o: %.pyx
#	$(CYTHON) --cplus md3dsys.pyx
#	$(CXX) -c $(CXXFLAGS)  -o obj/md3dsys.o md3dsys.cpp $(INCLUDES)
#	#rm md3dsys.cpp 

$(ODIR)/%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)  $(INCLUDES) 

$(ODIR)/%.o: %.cu
	$(NVCC) -dc -o $@ $< $(NVCCFLAGS)  $(INCLUDES)  

$(BIN): $(OBJS)
	$(NVCC) -o $(BIN) $(OBJS) $(GPUOBJS) $(PYLIB)

#$(SO): $(OBJS)
#	$(NVCC) --shared -o $(SO) $(OBJS) $(GPUOBJS)

clean: 
	rm -rf $(ODIR)/*.o $(BINDIR)/*.xyz

