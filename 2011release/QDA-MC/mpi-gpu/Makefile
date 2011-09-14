#
# QDAMC Makefile
#

PROGRAM=qdmc
CC = mpicc
CPP = mpiCC
override CFLAGS += -Wall -msse2 -DAFLOAT64 -DPFLOAT64 -O2 -DNDEBUG
CFLAGS += -finline-functions -funswitch-loops -fpredictive-commoning -fgcse-after-reload -fipa-cp-clone 
DFLAGS =
CLIBS = -lm -I/usr/local/cuda/include/ -lOpenCL

DS_FLAGS = -msse2 -DHAVE_SSE2 -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800 \
-Wmissing-prototypes -Wall -std=c99 --param inline-unit-growth=500 \
--param large-function-growth=900

OBJS = ./src/qda-mc/qdamc.o ./src/qdamc_cli.o ./src/getopt/getoptl.o
DEPS = 

all: $(PROGRAM)

qdmc: $(OBJS)
	$(CC) -o $@ $(CFLAGS) $(CLIBS) $(OBJS) $(DFLAGS)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(CLIBS) $(DFLAGS)

clean:
	rm -f $(PROGRAM) $(OBJS)

