if [ $# -eq 0 ]
then
    
    make clean
    make CFLAGS="-DDO_GPU"
    mv qdmc qdmc_omp_gpu
    
    make clean
    make CFLAGS="-DDO_ASYNC"
    mv qdmc qdmc_omp_gpu_async
    
    rm -f qdmc
    
else
    make clean
    rm -f qdmc qdmc_omp qdmc_omp_l2_2 qdmc_omp_gpu_l2_2 qdmc_omp_gpu_async_l2_2 qdmc_omp_gpu_async qdmc_omp_gpu
fi
