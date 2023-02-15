.PHONY: all clean parallel serial

serial: CFLAGS = -O3 -g -march=native -Wall 
parallel: CFLAGS = -O3 -g -march=native -Wall -fopenmp

all: parallel

serial: example
parallel: example

clean:
	rm -rf example

example: example.c
	gcc -o example $(CFLAGS) example.c -I ../../include/ -L ../../lib/ -lsphericart -lm
