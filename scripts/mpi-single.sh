#!/bin/bash

cd ../bld/ && \
	make && \
	time ./test_mpi
