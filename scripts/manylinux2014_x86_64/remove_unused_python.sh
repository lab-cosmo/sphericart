#!/bin/bash

#FindPython is weird and doesn't respect $PATH or even CMAKE $Python_Exec variables.
for python in /usr/local/bin/python*; do 
if [[ "$python" != *"$1"* ]]; then 
    echo "Removing $python"; 
    rm -f "$python"; 
fi; 
done