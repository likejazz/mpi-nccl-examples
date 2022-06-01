# MPI + NCCL examples

[NCCL](https://developer.nvidia.com/nccl) examples derived from [Official NVIDIA NCCL Developer Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html) for faster benchmark and deployment.

### Requirements

- [CUDA](https://developer.nvidia.com/cuda-zone)
- MPI, e.g. [OpenMPI](https://www.open-mpi.org/)
- [NCCL](https://developer.nvidia.com/nccl)

### Build

```
$ mkdir bld \
    && cd bld \
    && cmake .. \
    make
```

* Heavily derived from <https://github.com/1duo/nccl-examples>
