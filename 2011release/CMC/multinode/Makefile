#
# CMC Makefile
#

PROGRAM=cmc
CC = gcc
override CFLAGS += -Wall -msse2 -DAFLOAT64 -DPFLOAT64 -O2 -DNDEBUG -DOLDCOMPARE -finline-functions -funswitch-loops -fpredictive-commoning -fgcse-after-reload -fipa-cp-clone
PAPI_LIBS=

DFLAGS = #-pg -ggdb #-ftree-vectorizer-verbose=6 
CLIBS = -lm $(PAPI_LIBS)

DS_FLAGS = -DDSFMT_MEXP=521 -msse2 -DHAVE_SSE2 -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800 \
-Wmissing-prototypes -Wall -std=c99 --param inline-unit-growth=500 \
--param large-function-growth=900

OBJS = ./src/cmc/cmc.o ./src/cmc_cli.o ./src/getopt/getoptl.o
DEPS = 

cmc: $(OBJS)  ./src/dSFMT-2.1/dSFMT.o
	$(CC) -o $(PROGRAM) $(CFLAGS) $(CLIBS) $(OBJS) $(DFLAGS) ./src/dSFMT-2.1/dSFMT.o

./src/dSFMT-2.1/dSFMT.o: ./src/dSFMT-2.1/dSFMT.c
	$(CC) -c -o $@ $< $(DS_FLAGS)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(CLIBS) $(DFLAGS) -DDSFMT_MEXP=521

clean:
	rm -f $(PROGRAM) $(OBJS) ./src/dSFMT-2.1/dSFMT.o

