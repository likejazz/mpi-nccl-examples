#!/bin/bash

cd ../bld/ && \
	make && \
	time ./test_nccl
