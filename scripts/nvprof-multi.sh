#!/bin/bash

cd ../bld/ && \
	make && \
	time mpirun -x NCCL_SOCKET_IFNAME=ens \
	-x LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH \
	--host gpu01,gpu02,gpu03 \
	--mca btl_base_warn_component_unused 0 \
	--mca btl_tcp_if_include ens5 \
	/usr/local/cuda/bin/nvprof ./test_nccl
