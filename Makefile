.PHONY: all

all: libsphericart.so

libsphericart.so:
	gcc --shared -O3 -lm sphericart.c -o libsphericart.so
