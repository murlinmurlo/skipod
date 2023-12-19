
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h> 

#define Max(a,b) ((a)>(b)?(a):(b))
#define N (2*2*2*2*2*2+2)

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double eps;
double A[N][N], B[N][N];

void relax();
void resid();
void init();
void verify();

int tred;

int main(int an, char **as)
{
    tred = strtoll(as[1], 0, 10);
    double start = omp_get_wtime();

    int it;
    init();
    for (it = 1; it <= itmax; it++)
    {
        eps = 0.;
        relax();
        resid();
        printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps)
            break;
    }
    verify();
    clock_t end = clock();

    printf("TIME %f", omp_get_wtime()-start);

    
    return 0;
}


void init()
{ 
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	
	{
		if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
		else A[i][j]= ( 1. + i + j ) ;
	}
}
void relax()
{
#pragma omp parallel for shared(B, A) private(i, j) num_threads(tred)
    for (i = 2; i <= N - 3; i++)
        for (j = 2; j <= N - 3; j++)
        {
            B[i][j] = (A[i - 2][j] + A[i - 1][j] + A[i + 2][j] + A[i + 1][j] + A[i][j - 2] + A[i][j - 1] + A[i][j + 2] + A[i][j + 1]) / 8.;
        }
}

void resid()
{
#pragma omp parallel for private(i, j) reduction(max : eps) num_threads(tred)
    for (i = 1; i <= N - 2; i++)
        for (j = 1; j <= N - 2; j++)
        {
            double e;
            e = fabs(A[i][j] - B[i][j]);
            A[i][j] = B[i][j];
            eps = Max(eps, e);
        }
}

void verify()
{
	double s;
	s=0.;
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	
	{
		s=s+A[i][j]*(i+1)*(j+1)/(N*N);
	}
	printf("  S = %f\n",s);
}

