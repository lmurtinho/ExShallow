#!/bin/sh

gcc -shared -fPIC -o lib_best_cut.so best_cut_lib.c
gcc -shared -fPIC -o lib_best_cut_shallow.so best_cut_lib_shallow.c
