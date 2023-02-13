.PHONY: all clean

CFLAGS = -O3 -g -march=native -Wall -fopenmp 

all: example

clean:
	rm -rf example

example: example.c
	gcc -o example $(CFLAGS) example.c -I ../../include/ -L ../../lib/ -lsphericart -lm
