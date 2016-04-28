#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <errno.h>
#include <limits.h>
#include <time.h>
#include <math.h>

// for Debug purpose
#ifdef DEBUG_ENABLED
#	define dprintf(...) {			\
		printf("DEBUG] ");			\
		printf(__VA_ARGS__);		\
	}
#else
#	define dprintf(...) (0)
#endif

#define NANOSECOND 1000000000.0
#define ELAPSED(a, b) ((b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / NANOSECOND)

#define true 1
#define false 0

void do_solve();

double *_A, *_B, *_origA, *_origB;
int n;
int current = -1;

#define A(a, b) (_A[(a) * n + (b)])
#define B(i) (_B[i])
#define origA(a, b) (_origA[a * n + b])
#define origB(i) (_origB[i])

// for debug-purpose
#ifdef DEBUG_ENABLED
void print_result()
{
	for (int i = 0; i < n; i++)
	{
		dprintf("  %03d: ", i);
		for (int j = 0; j < n; j++)
		{
			printf("%f ", A(i, j));
		}
		printf("%f\n", B(i));
	}
}
#endif

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("usage: ./assignment2 matrix_size");
		return -1;
	}

	int i, j;
	char *ptr;

	errno = 0;
	n = strtol(argv[1], &ptr, 10);
	if (errno > 0 || *ptr != '\0' || n < 1 || n > INT_MAX) { return -1; }

	errno = posix_memalign((void **)&_A, 0x40, n * n * sizeof(double));
	if (errno > 0) { return -1; }
	errno = posix_memalign((void **)&_B, 0x40, n * sizeof(double));
	if (errno > 0) { return -1; }

	_origA = (double*)malloc(n * n * sizeof(double));
	_origB = (double*)malloc(n * sizeof(double));

	// time elapse
	struct timespec start, finish;
	double elapsed_time, total_time;

	clock_gettime(CLOCK_MONOTONIC, &start);

	// set array
	struct drand48_data rand_buffer;
	#ifndef NORAND
		srand48_r(time(NULL), &rand_buffer);
	#endif
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			drand48_r(&rand_buffer, &A(i, j));
			#ifdef NORAND
				A(i, j) = ((double)rand() / (RAND_MAX));
			#endif
		}
		drand48_r(&rand_buffer, &B(i));
		#ifdef NORAND
			B(i) = (double)rand() / (RAND_MAX);
		#endif
	}
	clock_gettime(CLOCK_MONOTONIC, &finish);
	total_time = ELAPSED(start, finish);

	memcpy(_origA, _A, sizeof(double) * n * n);
	memcpy(_origB, _B, sizeof(double) * n);

	#ifdef DEBUG_ENABLED
		// print original array
		dprintf("original:\n");
		print_result();
	#endif

	clock_gettime(CLOCK_MONOTONIC, &start);
	do_solve();
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed_time = ELAPSED(start, finish);

	#ifdef DEBUG_ENABLED
		// print result array
		dprintf("result:\n");
		print_result();
	#endif

	double diff = 0.0;
	for (i = 0; i < n; i++)
	{
		double check_value = 0.0;
		for (j = 0; j < n; j++)
		{
			check_value += origA(i, j) * B(j);
		}
		diff += (check_value - origB(i)) * (check_value - origB(i));
	}

	printf( "\n"
			"error (L2 norm): %.32lf\n"
			"matrix gen. time: %.32lf\n"
			"gaussian elim. time: %.32lf\n"
			, sqrt(diff), total_time, elapsed_time);

	total_time += elapsed_time;
	printf("\nTotal time: %.32lf\n", total_time);

	return 0;
}

#ifdef NAIVE
void do_solve()
{
	int i, j;

	// - 1st phase -
	while (++current < n)
	{
		int pivot_line = -1;
		double pivot = 0;

		// -- search pivot --
		for (i = current; i < n; i++)
		{
			if (pivot < fabs(A(i, current)))
			{
				pivot = fabs(A(i, current));
				pivot_line = i;
			}
		}
		pivot = A(pivot_line, current);
		printf("find pivot %.32f in %d\n\n", pivot, pivot_line);

		// -- switch pivot --
		if (current != pivot_line)
		{
			double temp;
			for (j = current; j < n; j++)
			{
				temp = A(current, j);
				A(current, j) = A(pivot_line, j);
				A(pivot_line, j) = temp;
			}

			temp = B(current);
			B(current) = B(pivot_line);
			B(pivot_line) = temp;

			#ifdef DEBUG_ENABLED
				dprintf("switched\n");
				print_result();
			#endif
		}

		// -- set pivot to 1 --
		for (j = current + 1; j < n; j++)
			A(current, j) /= pivot;

		B(current) /= pivot;
		A(current, current) = 1;

		#ifdef DEBUG_ENABLED
			dprintf("divided\n");
			print_result();
		#endif

		// -- set other to 0 --
		for (i = current + 1; i < n; i++)
		{
			double target = -A(i, current);
			for (j = current; j < n; j++)
			{
				A(i, j) += target * A(current, j);
			}
			B(i) += target * B(current);
		}

		#ifdef DEBUG_ENABLED
			dprintf("subtracted\n");
			print_result();
		#endif
	}
	// - 1st phase end -

	// - 2nd phase -
	while (--current > 0)
	{
		// -- set other to 0 --
		for (i = 0; i < current; i++)
		{
			double target = -A(i, current);
			// doesn't need
			// A(i, current) = 0;
			B(i) += target * B(current);
		}
	}
	// - 2nd phase end -
}
#endif

#ifdef INFO

/* copied from helper_cuda.h */
// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

void do_solve()
{
	int count;
	cudaGetDeviceCount(&count);

	for (int i = 0; i < count; i++)
	{
		struct cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("\n\n"
			"Device info #%d =====================================\n\n"
			"Device name: %s\n"
			"Total global memory: %zd bytes\n"
			"warp size: %d threads\n"
			"Max threads per block: %d\n"
			"Max threads per dimension: [%d, %d, %d]\n"
			"Max grid size: [%d, %d, %d]\n"
			"Streaming Multiprocessor count: %d\n"
			"Cuda cores per SM: %d\n"
			, i
			, prop.name, prop.totalGlobalMem, prop.warpSize
			, prop.maxThreadsPerBlock, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]
			, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]
			, prop.multiProcessorCount
			, _ConvertSMVer2Cores(prop.major, prop.minor));
	}
}
#endif

#ifdef CUDA

#define THREADCOUNT 256
#define PIVOTLSEARCH false
#define d_Array(_d, _pitch, x, y) ( *( (double*)( (int8_t*)(_d) + (x) * (_pitch) ) + (y) ) )

#define getMax(local_pivot, local_pivot_line, tid, s)			\
{																\
	if (local_pivot[tid] < local_pivot[tid + s])				\
	{															\
		local_pivot[tid] = local_pivot[tid + s];				\
		local_pivot_line[tid] = local_pivot_line[tid + s];		\
	}															\
}

__global__
void getLocalMax(double *d_A, double *d_pivot, int *d_pivot_line, int size, size_t d_pitch, int current)
{
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;

	__shared__ double	local_pivot		[THREADCOUNT];
	__shared__ int		local_pivot_line[THREADCOUNT];

	if (i < size)
	{
		local_pivot[tid] = fabs(d_Array(d_A, d_pitch, i + current, current));
		local_pivot_line[tid] = i + current;
	}
	else
	{
		local_pivot[tid] = 0;
	}
	__syncthreads();

	if (THREADCOUNT >= 512)
	{
		if (tid < 256) { getMax(local_pivot, local_pivot_line, tid, 256); } __syncthreads();
	}

	if (THREADCOUNT >= 256)
	{
		if (tid < 128) { getMax(local_pivot, local_pivot_line, tid, 128); } __syncthreads();
	}

	if (THREADCOUNT >= 128)
	{
		if (tid < 64)  { getMax(local_pivot, local_pivot_line, tid, 64);  } __syncthreads();
	}

	if (tid < 32)
	{
		getMax(local_pivot, local_pivot_line, tid, 32);
		getMax(local_pivot, local_pivot_line, tid, 16);
		getMax(local_pivot, local_pivot_line, tid, 8);
		getMax(local_pivot, local_pivot_line, tid, 4);
		getMax(local_pivot, local_pivot_line, tid, 2);
		getMax(local_pivot, local_pivot_line, tid, 1);
	}

	if (tid == 0)
	{
		d_pivot[blockIdx.x]			= local_pivot[0];
		d_pivot_line[blockIdx.x]	= local_pivot_line[0];
	}
}

__global__
void getRealMax(double *d_pivot, int *d_pivot_line, int size)
{
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + tid;

	__shared__ double	local_pivot		[THREADCOUNT];
	__shared__ int		local_pivot_line[THREADCOUNT];

	if (i < size)
	{
		local_pivot[tid]		= d_pivot[i];
		local_pivot_line[tid]	= d_pivot_line[i];
	}
	else
	{
		local_pivot[tid] = 0;
	}
	__syncthreads();

	if (THREADCOUNT >= 512)
	{
		if (tid < 256) { getMax(local_pivot, local_pivot_line, tid, 256); }
	}
	__syncthreads();

	if (THREADCOUNT >= 256)
	{
		if (tid < 128) { getMax(local_pivot, local_pivot_line, tid, 128); }
	}
	__syncthreads();

	if (THREADCOUNT >= 128)
	{
		if (tid < 64)  { getMax(local_pivot, local_pivot_line, tid, 64);  }
	}
	__syncthreads();

	if (tid < 32)
	{
		getMax(local_pivot, local_pivot_line, tid, 32);
		getMax(local_pivot, local_pivot_line, tid, 16);
		getMax(local_pivot, local_pivot_line, tid, 8);
		getMax(local_pivot, local_pivot_line, tid, 4);
		getMax(local_pivot, local_pivot_line, tid, 2);
		getMax(local_pivot, local_pivot_line, tid, 1);
	}

	if (tid == 0)
	{
		d_pivot[0] = local_pivot[0];
		d_pivot_line[0] = local_pivot_line[0];
	}
}

__global__
void swap_divide(double *d_A, size_t d_pitch, double pivot, int pivot_line, int current, int size)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size)
	{
		i += current;

		double current_value;

		if (pivot_line != current)
		{
			// swap
			double current_temp = d_Array(d_A, d_pitch, current, i);

			current_value = d_Array(d_A, d_pitch, pivot_line, i);
			d_Array(d_A, d_pitch, pivot_line, i) = current_temp;
		}
		else
		{
			current_value = d_Array(d_A, d_pitch, current, i);
		}

		// divide
		d_Array(d_A, d_pitch, current, i) = current_value / pivot;
	}
}

__global__
void subtractPivot(double *d_A, size_t d_pitch, int size, int current)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < size)
	{
		int y = current + 1 + j;
		double pivot_line = d_Array(d_A, d_pitch, current, y);
		for (int i = 0; i < size; i++)
		{
			int x = current + 1 + i;
			double target = d_Array(d_A, d_pitch, x, current);

			d_Array(d_A, d_pitch, x, y) -= target * pivot_line;
		}
	}
}

__global__
void subtractB(double *d_A, size_t d_pitch, double *d_B, int size, int current)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid < size)
	{
		tid += current + 1;
		d_B[tid] -= d_Array(d_A, d_pitch, tid, current) * d_B[current];
	}
}

__global__
void backSubtract(double *d_A, size_t d_pitch, double *d_B, int current)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < current)
	{
		double target = d_Array(d_A, d_pitch, tid, current);
		d_B[tid] -= target * d_B[current];
	}
}

#ifdef DEBUG_ENABLED
__global__
void debug_print(double *d_A, size_t d_pitch, int size)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = 0; i < size && j < size && i == 4; j++)
	{
		printf("[%d, %d]: %f\n", i, j, d_Array(d_A, d_pitch, i, j));
	}
}
#endif

#define cudaCheckErrors(msg) do \
{ \
	cudaError_t __err = cudaGetLastError(); \
	if (__err != cudaSuccess) \
	{ \
		fprintf(stderr, "Error in %s : %d, (%s at %s:%d)\n", \
			msg, __err, cudaGetErrorString(__err), \
			__FILE__, __LINE__); \
		exit(1); \
	} \
} while (0)

#define BLOCK_COUNT(c, tc) ((n - 1 - (c)) / (tc) + 1)

void do_solve()
{
	double *d_A, *d_B;
	size_t d_pitch;

	cudaStream_t b_stream, swap_stream;
	cudaStreamCreate(&b_stream);
	cudaStreamCreate(&swap_stream);

	#ifdef EBM
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	#endif

	cudaMallocPitch((void**)&d_A, &d_pitch, sizeof(double) * n, n);
	cudaCheckErrors("cudaMallocPitch");

	cudaMalloc((void**)&d_B, sizeof(double) * n);
	cudaCheckErrors("cudaMalloc");

	cudaMemcpy2D(d_A, d_pitch, _A, sizeof(double) * n, sizeof(double) * n, n, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy2D");

	cudaMemcpy(d_B, _B, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy");

	// array for searching pivot
	double *d_pivot;
	int *d_pivot_line;

	cudaMalloc((void**)&d_pivot, BLOCK_COUNT(0, THREADCOUNT) * sizeof(double));
	cudaMalloc((void**)&d_pivot_line, BLOCK_COUNT(0, THREADCOUNT) * sizeof(int));

	// - 1st phase -
	while (++current < n)
	{
		// get Block count
		int block_count = BLOCK_COUNT(current, THREADCOUNT);

		double pivot;
		int pivot_line;

		// -- find pivot --
		getLocalMax<<<block_count, THREADCOUNT>>>(d_A, d_pivot, d_pivot_line, n - current, d_pitch, current);

		if (PIVOTLSEARCH)
		{
			int size = block_count;
			block_count = (block_count - 1) / THREADCOUNT + 1;

			getRealMax<<<block_count, THREADCOUNT>>>(d_pivot, d_pivot_line, size);
		}

		double local_pivot[THREADCOUNT];
		int local_pivot_line[THREADCOUNT];

		cudaMemcpy(local_pivot, d_pivot, (block_count * sizeof(double)), cudaMemcpyDeviceToHost);
		cudaCheckErrors("cudaMemcpy");
		cudaMemcpy(local_pivot_line, d_pivot_line, (block_count * sizeof(int)), cudaMemcpyDeviceToHost);
		cudaCheckErrors("cudaMemcpy");

		pivot = local_pivot[0]; pivot_line = local_pivot_line[0];
		for (int i = 1; i < block_count; i++)
		{
			if (pivot < local_pivot[i])
			{
				pivot = local_pivot[i];
				pivot_line = local_pivot_line[i];
			}
		}
		// get real pivot (not absolute value)
		cudaMemcpy(&pivot, &d_Array(d_A, d_pitch, pivot_line, current), sizeof(double), cudaMemcpyDeviceToHost);
		cudaCheckErrors("cudaMemcpy");

		// get block count again
		block_count = BLOCK_COUNT(current, THREADCOUNT);

		#ifdef DEBUG_ENABLED
			dprintf("find pivot %.32f in %d\n", pivot, pivot_line);
		#endif

		// -- switch pivot --
		cudaStreamSynchronize(b_stream);
		swap_divide<<<block_count, THREADCOUNT, 1, swap_stream>>>(d_A, d_pitch, pivot, pivot_line, current, n - current);
		if (pivot_line != current)
		{
			double temp = B(pivot_line);
			B(pivot_line) = B(current);
			B(current) = temp;
		}

		// -- divide pivot line --
		B(current) /= pivot;
		cudaMemcpyAsync(d_B, _B, sizeof(double) * n, cudaMemcpyHostToDevice, b_stream);
		cudaCheckErrors("cudaMemcpyAsync");

		cudaStreamSynchronize(swap_stream);

		#ifdef DEBUG_ENABLED
			printf("divided\n");
			cudaMemcpy2D(_A, sizeof(double) * n, d_A, d_pitch, sizeof(double) * n, n, cudaMemcpyDeviceToHost);
			print_result();
		#endif

		// -- subtract other lines --
		subtractPivot<<<block_count, THREADCOUNT>>>(d_A, d_pitch, n - current - 1, current);

		cudaStreamSynchronize(b_stream);
		subtractB<<<block_count, THREADCOUNT>>>(d_A, d_pitch, d_B, n - current - 1, current);
		cudaMemcpyAsync(_B, d_B, sizeof(double) * n, cudaMemcpyDeviceToHost, b_stream);
		cudaCheckErrors("cudaMemcpyAsync");

		#ifdef DEBUG_ENABLED
			printf("subtracted\n");
			cudaMemcpy2D(_A, sizeof(double) * n, d_A, d_pitch, sizeof(double) * n, n, cudaMemcpyDeviceToHost);
			print_result();
		#endif
	}

	// - 2nd phase -
	while (--current > 0)
	{
		int block_count = BLOCK_COUNT(n - current, THREADCOUNT);
		backSubtract<<<block_count, THREADCOUNT>>>(d_A, d_pitch, d_B, current);

		#ifdef DEBUG_ENABLED
			printf("2nd phase: current %d\n", current);
			cudaMemcpy(_B, d_B, sizeof(double) * n, cudaMemcpyDeviceToHost);
			print_result();
		#endif
	}

	cudaMemcpy(_B, d_B, sizeof(double) * n, cudaMemcpyDeviceToHost);
}
#endif