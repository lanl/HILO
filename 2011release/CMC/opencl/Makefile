#
# QDAMC Makefile
#

PROGRAM=cmc
CC = gcc
CPP = g++
override CFLAGS += -Wall -msse2 -m64 -O3
OPENCL_LIBDIR = /usr/lib64/
OPENCL_INCDIR = /usr/local/cuda/include/
CFLAGS += -finline-functions -funswitch-loops -fpredictive-commoning -fgcse-after-reload -fipa-cp-clone
DFLAGS = #-pg -ggdb #-ftree-vectorizer-verbose=6 
CLIBS = -lm -L/$(OPENCL_LIBDIR) -lOpenCL
CINCLUDE = -I/$(OPENCL_INCDIR)

DS_FLAGS = -DDSFMT_MEXP=521 -msse2 -DHAVE_SSE2 -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800 \
-Wmissing-prototypes -Wall -std=c99 --param inline-unit-growth=500 \
--param large-function-growth=900

OBJS = ./src/cmc/montecarlo.o ./src/cmc_cli.o ./src/getopt/getopt.o
DEPS = 

cmc: $(OBJS)  
	$(CC) $(CINCLUDE) -o $(PROGRAM) $(CFLAGS) $(CLIBS) $(OBJS) $(DFLAGS) 

%.o: %.c $(DEPS)
	$(CC) $(CINCLUDE) -c -o $@ $< $(CFLAGS) $(CLIBS) $(DFLAGS) 

clean:
	rm -f $(PROGRAM) $(OBJS) 

