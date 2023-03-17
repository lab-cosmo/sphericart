CFLAGS = -O3 -g -march=native -fopenmp -Wall

example: example.c
	gcc -o example $(CFLAGS) -O3 example.c -lsphericart -lstdc++ -lm


.PHONY: clean

clean:
	rm -rf example
