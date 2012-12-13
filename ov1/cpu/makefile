#------------------------------------------------------
# Makefile for C/C++ Program
# Authour: H. Watanabe
# $Id: makefile 246 2005-12-10 08:24:50Z kaityo $
#------------------------------------------------------


TARGET=ov
GDBTARGET=ovd
LIB=
INC=

#------------------------------------------------------
# Default Parameters
#------------------------------------------------------

CC=g++
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

SRC=$(shell ls *.cc)
HED=$(shell ls *.h)
OBJ=$(SRC:.cc=.o)

#------------------------------------------------------
# rules
#------------------------------------------------------

all: $(TARGET)
$(TARGET): $(OBJ)
	$(CC) $(OPT) -o $(TARGET) $(OBJ) $(LIB)

gdb: $(GDBTARGET)
$(GDBTARGET): $(OBJ)
	$(CC) $(OPT) -g -o $(GDBTARGET) $(OBJ) $(LIB)

.c.o:
	$(CC) $(OPT) -c $< $(MPI)
.cc.o:
	$(CC) $(OPT) -c $< $(MPI)

dep:
	g++ -MM -MG $(SRC) >makefile.depend

clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f *~
	rm -f *.~*

html: $(SRC)
	gtags
	htags

tar:
	tar cvzf $(TARGET).tar.gz *.cc makefile

#--------------------------------------------------
-include makefile.depend
