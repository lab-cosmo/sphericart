CFLAGS = -O3 -g -march=native -fopenmp -Wall

example: example.c
	g++ -o example $(CFLAGS) -O3 example.c -lsphericart -L .


.PHONY: clean

clean:
	rm -rf example
