#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h> 

#define Max(a,b) ((a)>(b)?(a):(b))
#define N (8*2*2*2*2*2*2+2)

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
    int tred = strtoll(as[1], 0, 10);
    double start = omp_get_wtime();

    int it;
    #pragma omp parallel num_threads(tred)
    {
        #pragma omp master
        {
            init();
            
            for (it = 1; it <= itmax; it++)
            {
                eps = 0.;
                #pragma omp taskwait
                relax();
                #pragma omp taskwait
                resid();
                #pragma omp taskwait
                printf("it=%4i   eps=%f\n", it, eps);
                if (eps < maxeps)
                    break;
            }
            
            verify();
        }
    }

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
    
    for(i=2; i<=N-3; i++)
        #pragma omp task shared(B, A) firstprivate(i, j)
        {
            for(j=2; j<=N-3; j++)
            {
                B[i][j]=(A[i-2][j]+A[i-1][j]+A[i+2][j]+A[i+1][j]+A[i][j-2]+A[i][j-1]+A[i][j+2]+A[i][j+1])/8.;
            }
        }
}

void resid()
{ 
	
    for(i=1; i<=N-2; i++)
    {
        #pragma omp task shared(eps, A) firstprivate(j,i)
        
        {   
            double tmp = 0;
            for(j=1; j<=N-2; j++)
            {
                double e;
                e = fabs(A[i][j] - B[i][j]);         
                A[i][j] = B[i][j]; 
                tmp = Max(tmp,e);
            }
            #pragma omp critical
            {
                eps = Max(eps, tmp);
            }
        }
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
 

