#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  N   (2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double eps;
double *A, *B, *local_B;
int *recvcounts, *dislps;

void relax();
void resid();
void init();
void verify();

int main(int argc, char **argv)
{
    int it, rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    A = (double*)malloc(N*N*sizeof(double));
    B = (double*)malloc(N*N*sizeof(double));
    struct timeval starttime, stoptime;
    gettimeofday(&starttime, NULL);
    int rows = N%size > rank ? N / size + 1: N / size;
    local_B = (double*)malloc(N*rows*sizeof(double));
    recvcounts = (int*)malloc(size*sizeof(int));
    dislps = (int*)malloc(size*sizeof(int));
    dislps[0] = 0;
    recvcounts[0] = N%size > 0 ? (N / size + 1)*N: (N / size)*N;
    for (int i = 1; i < size; i++){
        recvcounts[i] = N%size > i ? (N / size + 1)*N: (N / size)*N;
        dislps[i] = dislps[i-1] +recvcounts[i-1];
    }

    int start = rank == 0? 2 : rank*(N/size);
    if (N%size > rank){
        start += rank;
    } else {
        start += N % size;
    }

    int end = (rank+1)*(N/size);
    if (N%size > rank){
        end += rank;
    } else {
        end += N % size-1;
    }
    if (rank == size-1)
        end = N-3;
    init();

    for (it = 1; it <= itmax; it++)
    {
        eps = 0.;
        relax(start, end, rank);
        if (rank == 0)
        resid(start-1,end,rank);
        else{
            if (rank == size-1)
                resid(start,end+1, rank);
            else resid(start, end, rank);
        }
        MPI_Gatherv(local_B, N*rows, MPI_DOUBLE, B, recvcounts, dislps, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0)
            for(j=1; j<=N-2; j++)
                for(i=1; i<=N-2; i++)
                    A[i*N + j] = B[i*N + j];
        MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        
        if (rank == 0)
        printf("it=%4i   eps=%f\n", it, eps);

        if (eps < maxeps)
            break;
    }
    if (rank == 0){
        verify();
        gettimeofday(&stoptime, NULL);
        long seconds = stoptime.tv_sec - starttime.tv_sec;
        long micsec  = stoptime.tv_usec - starttime.tv_usec;
        FILE * F = fopen("mpi_times.txt", "a");
        fprintf(F, "%f\n", seconds + micsec*1e-6);
    }
    fprintf(F, "%f\n", seconds + micsec*1e-6);
    free(A);
    free(B);
    free(local_B);
    MPI_Finalize();
    return 0;
}

void init()
{ 
    for (i = 0; i <= N-1; i++)
    {
        for (j = 0; j <= N-1; j++)
        {
            if (i == 0 || i == N-1 || j == 0 || j == N-1)
                A[i*N + j] = 0.;
            else
                A[i*N + j] = (1. + i + j);
        }
    }
}

void relax(int start, int end, int rank)
{
    int k = rank == 0 ? 2: 0;
    for (i = start; i <= end; i++)
    {
        for (j = 2; j <= N-3; j++)
        {
            local_B[(i-start+k)*N + j]=(A[(i-2) * N + j]+A[(i-1)*N + j]+A[(i+2)*N +j]+A[(i+1)*N + j]+A[i*N + j-2]+A[i*N + j-1]+A[i*N + j+2]+A[i*N + j+1])/8.;
        }
    }

   
}

void resid(int start,int end, int rank)
{ 
    double tmp = 0;
    int k = rank == 0 ? 1:0; 
    for (i = start; i <= end; i++)
    {
        for (j = 1; j <= N-2; j++)
        {
            double e;
            e = fabs(A[i*N + j] - local_B[(i-start+k)*N + j]);
            tmp = Max(tmp,e);
        }
    }

    MPI_Allreduce(&tmp, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

}

void verify()
{
    double s = 0.;
    for (i = 0; i <= N-1; i++)
    {
        for (j = 0; j <= N-1; j++)
        {
            s += A[i*N+j]*(i+1)*(j+1)/(N*N);;
        }
    }

    printf("  S = %f\n",s);
    
}
