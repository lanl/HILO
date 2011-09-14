#
# QDAMC Makefile
#

PROGRAM=qdmc #qdmc_omp qdmc_mpi qdmc_omp_mpi
CC = gcc
CPP = g++
override CFLAGS += -Wall -msse2 -DAFLOAT64 -DPFLOAT64 -O2 -DNDEBUG #-fopenmp -DDO_OMP   #-DDO_MPI #-ffast-math
CFLAGS += -finline-functions -funswitch-loops -fpredictive-commoning -fgcse-after-reload -fipa-cp-clone 
DFLAGS = #-pg -ggdb #-ftree-vectorizer-verbose=6 
PAPI_LIBS=
CLIBS = -lm $(PAPI_LIBS)

DS_FLAGS = -DDSFMT_MEXP=521 -msse2 -DHAVE_SSE2 -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800 \
-Wmissing-prototypes -Wall -std=c99 --param inline-unit-growth=500 \
--param large-function-growth=900

OBJS = ./src/qda-mc/qdamc.o ./src/qdamc_cli.o ./src/getopt/getoptl.o
DEPS = 

all: $(PROGRAM)

qdmc: $(OBJS)  ./src/dSFMT-2.1/dSFMT.o
	$(CC) -o $@ $(CFLAGS) $(CLIBS) $(OBJS) $(DFLAGS) ./src/dSFMT-2.1/dSFMT.o

# qdmc_omp:$(OBJS)  ./src/dSFMT-2.1/dSFMT.o
# 	$(CC) -o $@ $(CFLAGS) -DDO_OMP -fopenmp $(CLIBS) $(OBJS) $(DFLAGS) ./src/dSFMT-2.1/dSFMT.o

# qdmc_mpi:$(OBJS)  ./src/dSFMT-2.1/dSFMT.o
# 	$(CC) -o $@ $(CFLAGS) -DDO_MPI $(CLIBS) $(OBJS) $(DFLAGS) ./src/dSFMT-2.1/dSFMT.o

# qdmc_omp_mpi:$(OBJS)  ./src/dSFMT-2.1/dSFMT.o
# 	$(CC) -o $@ $(CFLAGS) -DDO_MPI -DDO_OMP -fopenmp $(CLIBS) $(OBJS) $(DFLAGS) ./src/dSFMT-2.1/dSFMT.o


./src/dSFMT-2.1/dSFMT.o: ./src/dSFMT-2.1/dSFMT.c
	$(CC) -c -o $@ $< $(DS_FLAGS)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(CLIBS) $(DFLAGS) -DDSFMT_MEXP=521

clean:
	rm -f $(PROGRAM) $(OBJS) ./src/dSFMT-2.1/dSFMT.o

