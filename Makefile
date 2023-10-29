CC=gcc
INC_PATH=-I.
LIB_PATH=#-L/home/dev_tools/apr/lib
LIBS=-lm -lavcodec -lavformat -lavutil
CFLAGS=-g -Wall -Wextra -O3

.PHONY: all
all: FreqSync

FreqSync: FreqSync.c
	$(CC) $(CFLAGS) $(INC_PATH) -o $@ $< $(LIB_PATH) $(LIBS)

.PHONY: clean
clean:
	rm -f FreqSync
