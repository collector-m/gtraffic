LIB = libtraffic_sync.so
SOLV = Solve
OV2D = OV2DSFrame

CUDA_ARCH=sm_20
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

NVCC = nvcc -arch=$(CUDA_ARCH) -cubin
GCC = gcc 
CFLAGS = -L$(CUDA_LIB) -I$(CUDA_INC)
LIBR = -lcuda -lm
CLASSPATH = .:/usr/local/JNA/jna.jar

all: LIB SOLV OV2D OVModel

LIB: solve_by_cuda_sync.cu traffic_lib_source_sync.c 2Dimtraffic_sync.h 
	$(NVCC) $(CFLAGS) -o solve_by_cuda_sync.cubin solve_by_cuda_sync.cu $(LIBR)
	$(GCC) $(CFLAGS) -Wall -shared traffic_lib_source_sync.c -o libtraffic_sync.so $(LIBR) -fPIC

SOLV: Solve.java
	javac -classpath $(CLASSPATH) Solve.java

OV2D: OV2DFrame.java
	javac -classpath $(CLASSPATH) OV2DFrame.java  

OVModel: OVModel_2D.java
	javac -classpath $(CLASSPATH) OVModel_2D.java

exe:;
	LD_LIBRARY_PATH=.:$(CUDA_LIB) java -classpath $(CLASSPATH) OVModel_2D

clean:
	rm -f solve_by_cuda_sync.cubin
	rm -f *~
	rm -f *.class
