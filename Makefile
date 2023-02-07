.PHONY: all clean

CFLAGS = -O3 -g -march=native -fopenmp -Wall

all: libsphericart.so example

clean:
	rm -rf libsphericart.so example

libsphericart.so: sphericart.c sphericart.h
	gcc --shared $(CFLAGS) sphericart.c -o libsphericart.so -lm -fpic -lopenblas 

example: libsphericart.so example.c
	gcc -o example $(CFLAGS) -O3 example.c -L. -lsphericart -lm
