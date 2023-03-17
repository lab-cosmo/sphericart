CXXFLAGS = -O3 -g -march=native -fopenmp -Wall

example: example.cpp
	g++ -o example $(CXXFLAGS) -O3 example.cpp -lsphericart


.PHONY: clean

clean:
	rm -rf example
