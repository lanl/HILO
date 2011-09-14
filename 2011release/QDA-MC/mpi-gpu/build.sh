if [ $# -eq 0 ]
then
    make clean
    make CFLAGS="-DDO_GPU"
    mv qdmc qdmc_mpi_gpu
    
    make clean
    make CFLAGS="-DDO_ASYNC"
    mv qdmc qdmc_mpi_gpu_async
else
    make clean
    rm -f qdmc qdmc_mpi qdmc_omp qdmc_omp_mpi qdmc_mpi_l2_1 qdmc_mpi_l2_2 qdmc_mpi_gpu qdmc_mpi_gpu_async
fi
