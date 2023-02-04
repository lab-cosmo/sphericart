.PHONY: all clean

CFLAGS = -O3 -g

all: libsphericart.so example

clean:
	rm -rf libsphericart.so example

libsphericart.so: sphericart.c sphericart.h
	gcc --shared $(CFLAGS) -lm sphericart.c -o libsphericart.so

example: libsphericart.so example.c
	gcc -o example $(CFLAGS) -O3 example.c -lsphericart -lm
