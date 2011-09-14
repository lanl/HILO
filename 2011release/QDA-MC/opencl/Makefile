#
# QDAMC Makefile
#

PROGRAM=qdmc
CC = gcc
CPP = g++
OPENCL_LIBDIR=/usr/lib64
OPENCL_INCDIR=/usr/local/cuda/include
override CFLAGS += -Wall -g -msse2 -m64 -O3
CFLAGS += -finline-functions -funswitch-loops -fpredictive-commoning -fgcse-after-reload -fipa-cp-clone
DFLAGS = #-pg -ggdb #-ftree-vectorizer-verbose=6 
CLIBS = -lm -L$(OPENCL_LIBDIR) -lOpenCL
CINCLUDE = -I$(OPENCL_INCDIR)

OBJS = ./src/montecarlo.o ./src/qdamc_cli.o
DEPS = 

cmc: $(OBJS)  
	$(CC) $(CINCLUDE) -o $(PROGRAM) $(CFLAGS) $(CLIBS) $(OBJS) $(DFLAGS) 

%.o: %.c $(DEPS)
	$(CC) $(CINCLUDE) -c -o $@ $< $(CFLAGS) $(CLIBS) $(DFLAGS) 

clean:
	rm -f $(PROGRAM) $(OBJS) 

