file(GLOB_RECURSE SOURCES "*.cu" "*.cpp" "*.c" "*.h" "*.cxx")
foreach (source ${SOURCES})
    get_filename_component(name ${source} NAME_WE)
    cuda_add_executable(${name} ${source})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_link_libraries(${name} /usr/local/cuda/lib64/libcudart.so
            ${MPI_LIBRARIES})
    target_link_libraries(${name} ${NCCL_LIBRARY})
endforeach (source)