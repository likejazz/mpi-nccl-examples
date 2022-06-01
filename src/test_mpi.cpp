#include <mpi.h>
#include <iostream>

int main(int argc, char **argv) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    // Initialize MPI
    int size, rank, name_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &name_len);

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

    // Print ID Information.
    std::cout << "Host: " << hostname
              << " rank(" << rank << "/" << size << "),"
              << " my_start(" << my_start << ") ~"
              << " my_end(" << my_end << ")" << std::endl;

    // Initialize a
    double start_time = MPI_Wtime();
    auto *a = new double[N];
    for (int i = my_start; i < my_end; i++) {
        a[i] = 1.0;
    }
    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "# Elapsed Times" << std::endl;
        std::cout << "- Initialize a : " << end_time - start_time << std::endl;
    }

    // Initialize b
    start_time = MPI_Wtime();
    auto *b = new double[N];
    for (int i = my_start; i < my_end; i++) {
        b[i] = 1.0 + double(i);
    }
    end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "- Initialize b : " << end_time - start_time << std::endl;
    }

    // Add the two arrays
    start_time = MPI_Wtime();
    for (int i = my_start; i < my_end; i++) {
        a[i] = a[i] + b[i];
    }
    end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "- Add two arrays : " << end_time - start_time << std::endl;
    }

    // Calculate average
    start_time = MPI_Wtime();
    double average = 0.0;
    for (int i = my_start; i < my_end; i++) {
        average += a[i] / double(N);
    }
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            double partial_average;
            MPI_Status status;
            MPI_Recv(&partial_average, 1, MPI_DOUBLE, i, 77, MPI_COMM_WORLD, &status);
            average += partial_average;
        }
    } else {
        MPI_Send(&average, 1, MPI_DOUBLE, 0, 77, MPI_COMM_WORLD);
    }
//    double partial_average = average;
//    MPI_Reduce(&partial_average, &average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "- Calculate average : " << end_time - start_time << std::endl;
    }

    // Print average of two lists
    if (rank == 0) {
        std::cout << "(Average Value : " << average << ")" << std::endl;
    }
    delete[] a;
    delete[] b;

    MPI_Finalize();

    // Print succeed footer.
    std::cout << "Host: " << hostname
              << " rank(" << rank << "/" << size << ") Succeed." << std::endl;
    return 0;
}
