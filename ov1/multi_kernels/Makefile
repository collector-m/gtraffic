#------------------------------------------------------
# Makefile for C Program
#------------------------------------------------------


TARGET=gtraf
DBGTARGET=gtrafd

CUDA_ARCH=sm_20
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64 

NVCC=nvcc -arch=$(CUDA_ARCH) -cubin
CFLAGS=-L$(CUDA_LIB) -I$(CUDA_INC)
LIB=-lcuda -lm

#------------------------------------------------------
# Default Parameters
#------------------------------------------------------

CC=gcc
OPT= 

#------------------------------------------------------
# Compile Option
#------------------------------------------------------

-include makefile.opt

#------------------------------------------------------
# Definition
#------------------------------------------------------

.SUFFIXES: .c .cc .h. .o

#---
# Source Files
#---

SRC=$(shell ls *.c)
HED=$(shell ls *.h)
OBJ=$(SRC:.cc=.o)

#------------------------------------------------------
# rules
#------------------------------------------------------

all: $(TARGET)
$(TARGET): $(OBJ)
	$(NVCC) $(OPT) -o rungekutta.cubin rungekutta.cu -lm
	$(CC) $(OPT) -o $(TARGET) $(CFLAGS) $(LIB) $(OBJ)

debug: $(DBGTARGET)
$(DBGTARGET): $(OBJ)
	$(NVCC) -g -G -o rungekutta.cubin rungekutta.cu -lm
	$(CC) $(OPT) -g -o $(DBGTARGET) $(CFLAGS) $(LIB) $(OBJ)


.c.o:
	$(CC) $(OPT) -c $< $(MPI)


dep:
	g++ -MM -MG $(SRC) >makefile.depend

clean:
	rm -f $(TARGET)
	rm -f $(DBGTARGET)
	rm -f *.o
	rm -f *~
	rm -f *.~*
	rm -f rungekutta.cubin

#--------------------------------------------------
-include makefile.depend
