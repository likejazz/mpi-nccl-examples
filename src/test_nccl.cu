#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static uint64_t getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

// --
// CUDA Functions

__global__ void initialize_a(double *a, int my_start, int my_end) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + my_start;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < my_end; i += stride) {
        a[i] = 1.0;
    }
}

__global__ void initialize_b(double *b, int my_start, int my_end) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + my_start;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < my_end; i += stride) {
        b[i] = 1.0 + double(i);
    }
}

__global__ void add_two_arrays(double *a, double *b, int my_start, int my_end) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + my_start;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < my_end; i += stride) {
        a[i] = a[i] + b[i];
    }
}

// --

int main(int argc, char *argv[]) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    int rank, size, localRank = 0;

    // Initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    double start_time = MPI_Wtime();
    int N = 200000000;  // 0.2B

    // Determine the workload of each ran
    int workloads[size];
    for (int i = 0; i < size; i++) {
        workloads[i] = N / size;
        if (i < N % size) {
            workloads[i]++;
        }
    }
    int my_start = 0;
    for (int i = 0; i < rank; i++) {
        my_start += workloads[i];
    }
    int my_end = my_start + workloads[rank];

    // Calculating localRank which is used in selecting a GPU
    uint64_t hostHashs[size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < size; p++) {
        if (p == rank) {
            break;
        }
        if (hostHashs[p] == hostHashs[rank]) {
            localRank++;
        }
    }

    std::cout << "Host: " << hostname
              << " rank(" << rank << "/" << size << "),"
              << " my_start(" << my_start << ") ~"
              << " my_end(" << my_end << ")" << std::endl;

    // Extract number of GPU devices from CUDA API.
    int deviceCount = 0;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    // IMPORTANT: Hard-coded value for easy testing.
    // This code wouldn't work on Multiple GPUs.
    deviceCount = 1;

    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t));

    // Generating NCCL unique ID at one process and broadcasting it to all
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPICHECK(MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0,
                       MPI_COMM_WORLD));

    // Setting up GPU variables and a CPU variable.
    double *d_a, *d_b;
    CUDACHECK(cudaMalloc(&d_a, sizeof(double) * N));
    CUDACHECK(cudaMalloc(&d_b, sizeof(double) * N));
    double *a = (double *) malloc(sizeof(double) * N);

    // Initializing NCCL, group API is required around ncclCommInitRank
    // as it is called across multiple GPUs in each thread/process
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < deviceCount; i++) {
        CUDACHECK(cudaSetDevice(0));
        CUDACHECK(cudaStreamCreate(&s[0]));
        NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

        initialize_a<<<10, 256>>>(d_a, my_start, my_end);
        initialize_b<<<10, 256>>>(d_b, my_start, my_end);
        add_two_arrays<<<10, 256>>>(d_a, d_b, my_start, my_end);
    }
    NCCLCHECK(ncclGroupEnd());
    NCCLCHECK(ncclReduce((const void *) d_a,
                         (void *) d_a,
                         N, ncclDouble, ncclSum, 0, comm, s[0]));
    CUDACHECK(cudaStreamSynchronize(s[0]));
    CUDACHECK(cudaMemcpy(a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost));

    // Calculate average
    double average = 0.0;
    for (int i = 0; i < N; i++) {
        average += a[i] / double(N);
    }

    // Elapsed Times
    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "# Elapsed Time" << std::endl;
        std::cout << "- Total : " << end_time - start_time << std::endl;
        if (rank == 0) {
            std::cout << "(Average Value : " << average << ")" << std::endl;
        }
    }

    // Finalize
    CUDACHECK(cudaFree(d_a));
    CUDACHECK(cudaFree(d_b));
    free(a);
    NCCLCHECK(ncclCommDestroy(comm));
    MPICHECK(MPI_Finalize());

    std::cout << "Host: " << hostname
              << " rank(" << rank << "/" << size << ") Succeed." << std::endl;
    return 0;
}