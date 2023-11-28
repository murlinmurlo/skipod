#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N (4*2*2*2*2*2*2+2)

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double eps;
double A[N][N], B[N][N];

void relax();
void resid();
void init();
void verify();

int main(int an, char **as)
{
    double time_spent = 0.0;
    clock_t begin = clock();

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

    /*
	for (int i = 0; i < N; i++) 
	{ 
		for (int j = 0; j < N; j++) 
		{ 
			printf("%f\t", A[i][j]); 
		} 
		printf("\n"); 
    }*/

    clock_t end = clock();

    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
 
    printf("TIME %f seconds", time_spent);

    return 0;
}

void init()
{ 
    #pragma omp parallel
    {
        #pragma omp single
        {
            for(int i=0; i<=N-1; i++)
            {
                for(int j=0; j<=N-1; j++)
                {
                    #pragma omp task
                    {
                        if(i==0 || i==N-1 || j==0 || j==N-1) 
                            A[i][j]= 0.;
                        else 
                            A[i][j]= ( 1. + i + j ) ;
                    }
                }
            }
        }
    }
}

void relax()
{
    #pragma omp for schedule(static) nowait
    for(i=2; i<=N-3; i++)
        for(j=2; j<=N-3; j++)
        {
            B[i][j]=(A[i-2][j]+A[i-1][j]+A[i+2][j]+A[i+1][j]+A[i][j-2]+A[i][j-1]+A[i][j+2]+A[i][j+1])/8.;
        }
}

void resid()
{ 
	#pragma omp parallel
	#pragma omp single
	{
		for(i=1; i<=N-2; i++)
		{
			for(j=1; j<=N-2; j++)
			{
				#pragma omp task
				{
					double e;
					e = fabs(A[i][j] - B[i][j]);         
					A[i][j] = B[i][j]; 
					#pragma omp critical
					eps = Max(eps,e);
				}
			}
		}
	}
}


void verify()
{
    double s = 0.0;
    int i, j;

    #pragma omp parallel shared(s) private(i, j)
    {
        #pragma omp single
        {
            for (i = 0; i <= N - 1; i++)
            {
                for (j = 0; j <= N - 1; j++)
                {
                    #pragma omp task
                    {
                        double temp = A[i][j] * (i + 1) * (j + 1) / (N * N);
                        #pragma omp atomic
                        s += temp;
                    }
                }
            }
        }
    }

    printf("S = %f\n", s);
}
