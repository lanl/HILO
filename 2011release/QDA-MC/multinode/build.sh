if [ $# -eq 0 ]
then
    make clean
    make CFLAGS=""

    make clean
    make CFLAGS="-DDO_MPI"
    mv qdmc qdmc_mpi
    
    make clean
    make CFLAGS="-DDO_MPI -DL2_1"
    mv qdmc qdmc_mpi_l2_1

    make clean
    make CFLAGS="-DDO_MPI -DL2_2"
    mv qdmc qdmc_mpi_l2_2

    make clean
    make CFLAGS="-DDO_MPI -DL2_3"
    mv qdmc qdmc_mpi_l2_3

    make clean
    make CFLAGS="-DDO_OMP -fopenmp"
    mv qdmc qdmc_omp
    
    make clean
    make CFLAGS="-DDO_MPI -DDO_OMP -fopenmp"
    mv qdmc qdmc_omp_mpi
    
    make clean 
    make CFLAGS="-DL2_2"
else
    make clean
    rm -f qdmc qdmc_mpi qdmc_omp qdmc_omp_mpi qdmc_mpi_l2_1 qdmc_mpi_l2_2
fi
