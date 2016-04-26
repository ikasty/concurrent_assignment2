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
int n, p;
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
	if (argc < 3)
	{
		printf("usage: ./assignment1 matrix_size thread_num");
		return -1;
	}

	int i, j;
	char *ptr;

	errno = 0;
	n = strtol(argv[1], &ptr, 10);
	if (errno > 0 || *ptr != '\0' || n < 1 || n > INT_MAX) { return -1; }
	p = strtol(argv[2], &ptr, 10);
	if (errno > 0 || *ptr != '\0' || p < 1 || p > INT_MAX) { return -1; }

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

#define THREADCOUNT 512
#define THREADCOUNT2D 32
#define PIVOTLBOUND 4
#define d_Array(_d, _pitch, x, y) ( *( (double*)( (int8_t*)(_d) + (x) * (_pitch) ) + (y) ) )
#define BLOCK_COUNT(c, tc) ((n - 1 - (c)) / (tc) + 1)

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
		//printf("tid %d load %f, at [%d, %d]\n", tid, local_pivot[tid], i + current, current);
	}
	else
	{
		local_pivot[tid] = 0;
	}
	__syncthreads();
/*
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
*/

	for (int i = THREADCOUNT / 2; i > 0; i /= 2)
	{
		if (tid < i)
		{
			getMax(local_pivot, local_pivot_line, tid, i);
		} __syncthreads();
	}

	if (tid == 0)
	{
		d_pivot[blockIdx.x]			= local_pivot[0];
		d_pivot_line[blockIdx.x]	= local_pivot_line[0];
	}
}

// TODO: thread_count 를 constant로 변경할 것
__global__
void getRealMax(double *d_pivot, int *d_pivot_line, int size, int thread_count)
{
	extern __shared__ float array[];
	unsigned int tid = threadIdx.x;

	double	*local_pivot 		= (double*)&array[0];
	int		*local_pivot_line	= (int*)&local_pivot[thread_count];

	if (tid < size)
	{
		local_pivot[tid]		= d_pivot[tid];
		local_pivot_line[tid]	= d_pivot_line[tid];
	}
	else
	{
		local_pivot[tid] = 0;
	}
	__syncthreads();
/*
	if (thread_count >= 512)
	{
		if (tid < 256) { getMax(local_pivot, local_pivot_line, tid, 256); }
	}
	__syncthreads();

	if (thread_count >= 256)
	{
		if (tid < 128) { getMax(local_pivot, local_pivot_line, tid, 128); }
	}
	__syncthreads();

	if (thread_count >= 128)
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
*/
	for (int i = thread_count / 2; i > 0; i /= 2)
	{
		if (tid < i)
		{
			getMax(local_pivot, local_pivot_line, tid, i);
		} __syncthreads();
	}

	if (tid == 0)
	{
		d_pivot[0] = local_pivot[0];
		d_pivot_line[0] = local_pivot_line[0];
	}
}
/*
__global__
void switchPivot(double *d_A, double *d_current, size_t d_pitch, int size, int pivot_line, int current)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size)
	{
		i += current;
		printf("switch [%d, %d]:%f, [%d, %d]:%f\n", pivot_line, i, d_Array(d_A, d_pitch, pivot_line, i), current, i, d_Array(d_current, 1, 0, i));
		d_Array(d_A, d_pitch, current, i) = d_Array(d_A, d_pitch, pivot_line, i);
		d_Array(d_A, d_pitch, pivot_line, i) = d_Array(d_current, d_pitch, 0, i);
	}
}*/

__global__
void switchPivot(double *d_A, size_t d_pitch, int size, int pivot_line, int current)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size)
	{
		i += current;
		//printf("switch [%d, %d]:%f, [%d, %d]:%f\n", pivot_line, i, d_Array(d_A, d_pitch, pivot_line, i), current, i, d_Array(d_current, 1, 0, i));
		double temp = d_Array(d_A, d_pitch, current, i);
		d_Array(d_A, d_pitch, current, i) = d_Array(d_A, d_pitch, pivot_line, i);
		d_Array(d_A, d_pitch, pivot_line, i) = temp;
	}
}

__global__
void dividePivot(double *d_A, size_t d_pitch, double pivot, int size, int current)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size)
	{
		i += current;// printf("switch %dth value\n", i);
		d_Array(d_A, d_pitch, current, i) /= pivot; //printf("[%d, %d]:%f\n", current, i, d_Array(d_A, d_pitch, current, i));
	}
}

/*
__global__
void subtractPivot(double *d_A, size_t d_pitch, int size, int current)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = threadIdx.y;
	
	unsigned int grp = blockDim.y;

	for (; j < size && i < size - 1; j += grp)
	{//printf("[%d, %d] -= [%d, %d] * [%d, %d]\n", i+current+1, j+current,  i+current+1,current, current, j+current);
		d_Array(d_A, d_pitch, i + current + 1, j + current) -=
				d_Array(d_A, d_pitch, i + current + 1, current) *
				d_Array(d_A, d_pitch, current, j + current);
	}
}
*/
__global__
void subtractPivot(double *d_A, size_t d_pitch, int size, int current)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int grp = blockDim.x;
	unsigned int boundary = (blockIdx.x + 1) * blockDim.x;
	if (boundary > size - 1) boundary = size - 1;

	__shared__ double target[THREADCOUNT];
	if (tid < size - 1)
	{
		target[threadIdx.x] = -d_Array(d_A, d_pitch, tid + current + 1, current);
	}
	__syncthreads();

	for (int i = blockIdx.x * blockDim.x; i < boundary; i++)
	{
		for (int j = tid; j < size; j += grp)
		{
			d_Array(d_A, d_pitch, i + current + 1, j + current) += target[i] * d_Array(d_A, d_pitch, current, j + current);
			//printf("[%d, %d]:%f += [%d, %d]:%f * [%d, %d]:%f\n", i + current + 1, j + current, d_Array(d_A, d_pitch, i + current + 1, j + current),
			//	i + current + 1, current, target[i],
			//	current, j + current, d_Array(d_A, d_pitch, current, j + current));
		}
	}
}

__global__
void subtractB(double *d_A, size_t d_pitch, double *d_B, int size, int current)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < size)
	{
		i += current + 1;//printf("tid %d d_B: %f, %f * %f\n", i, d_B[i], d_Array(d_A, d_pitch, i, current), d_B[current]);
		d_B[i] -= d_Array(d_A, d_pitch, i, current) * d_B[current];
	}
}

__global__
void backSubtract(double *d_A, size_t d_pitch, double *d_B, int current)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	double target = -d_Array(d_A, d_pitch, i, current);

	if (i < current)
	{//printf("tid %d: %f * %f\n", i, target, d_Array(d_A, d_pitch, i, current));
		d_B[i] += target * d_B[current];
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

void do_solve()
{
	double *d_A, *d_B;
	size_t d_pitch;

	double *pivot_candidate;
	errno = posix_memalign((void **)&pivot_candidate, 0x40, n * sizeof(double));
	if (errno) return ;
	cudaStream_t pivot_stream, b_stream;
	cudaStreamCreate(&pivot_stream);
	cudaStreamCreate(&b_stream);

	#ifdef EBM
	//TODO: compare time w/o following code:
//		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
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

		// get pivot line from device
		cudaMemcpy2DAsync(pivot_candidate, sizeof(double), &d_Array(d_A, 0, current), d_pitch, sizeof(double), n, cudaMemcpyDeviceToHost, pivot_stream);
		cudaCheckErrors("cudaMemcpy2DAsync");

		// -- find pivot --
		getLocalMax<<<block_count, THREADCOUNT>>>(d_A, d_pivot, d_pivot_line, n - current, d_pitch, current);
		if (block_count > PIVOTLBOUND)
		{
			int thread_count = THREADCOUNT;
			while (thread_count < block_count) thread_count *= 2;

			int shared_size = (sizeof(double) + sizeof(int)) * thread_count;

			cudaDeviceSynchronize();
			getRealMax<<<1, thread_count, shared_size>>>(d_pivot, d_pivot_line, block_count, thread_count);
			cudaDeviceSynchronize();
			cudaMemcpy(&pivot_line, d_pivot_line, sizeof(int), cudaMemcpyDeviceToHost);
		}
		else
		{
			double local_pivot[PIVOTLBOUND];
			int local_pivot_line[PIVOTLBOUND];

			cudaDeviceSynchronize();
			cudaMemcpy(local_pivot, d_pivot, block_count * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(local_pivot_line, d_pivot_line, block_count * sizeof(int), cudaMemcpyDeviceToHost);

			pivot = local_pivot[0]; pivot_line = local_pivot_line[0];
			for (int i = 1; i < block_count; i++)
			{
				if (pivot < local_pivot[i])
				{
					pivot = local_pivot[i];
					pivot_line = local_pivot_line[i];
				}
			}
		}

		cudaStreamSynchronize(pivot_stream);
		cudaDeviceSynchronize();
		pivot = pivot_candidate[pivot_line];

		printf("find pivot %.32f in %d\n", pivot, pivot_line);

		// -- switch pivot --
		if (pivot_line != current)
		{
			cudaDeviceSynchronize();
			switchPivot<<<block_count, THREADCOUNT>>>(d_A, d_pitch, n - current, pivot_line, current);
			cudaDeviceSynchronize();

			cudaStreamSynchronize(b_stream);
			double temp = B(pivot_line);
			B(pivot_line) = B(current);
			B(current) = temp;

			temp = pivot_candidate[pivot_line];
			pivot_candidate[pivot_line] = pivot_candidate[current];
			pivot_candidate[current] = temp;

			#ifdef DEBUG_ENABLED
//				printf("switched\n");
//				cudaMemcpy2D(_A, sizeof(double) * n, d_A, d_pitch, sizeof(double) * n, n, cudaMemcpyDeviceToHost);
//				print_result();
			#endif
		}

		// -- divide pivot line --
		cudaStreamSynchronize(b_stream);
		cudaDeviceSynchronize();
		B(current) /= pivot;
		cudaMemcpyAsync(d_B, _B, sizeof(double) * n, cudaMemcpyHostToDevice, b_stream);
		dividePivot<<<block_count, THREADCOUNT>>>(d_A, d_pitch, pivot, n - current, current);

		#ifdef DEBUG_ENABLED
//			printf("divided\n");
//			cudaMemcpy2D(_A, sizeof(double) * n, d_A, d_pitch, sizeof(double) * n, n, cudaMemcpyDeviceToHost);
//			print_result();
		#endif

		// -- subtract other lines --
		cudaStreamSynchronize(b_stream);
		cudaDeviceSynchronize();
		subtractB<<<block_count, THREADCOUNT>>>(d_A, d_pitch, d_B, n - current - 1, current);
		cudaMemcpyAsync(_B, d_B, sizeof(double) * n, cudaMemcpyDeviceToHost, b_stream);
		cudaCheckErrors("cudaMemcpyAsync");

		cudaDeviceSynchronize();
		subtractPivot<<<block_count, THREADCOUNT>>>(d_A, d_pitch, n - current, current);
		cudaDeviceSynchronize();

		#ifdef DEBUG_ENABLED
//			printf("subtracted\n");
//			cudaMemcpy2D(_A, sizeof(double) * n, d_A, d_pitch, sizeof(double) * n, n, cudaMemcpyDeviceToHost);
//			print_result();
		#endif
	}

	// - 2nd phase -
	while (--current > 0)
	{
		int block_count = BLOCK_COUNT(current, THREADCOUNT);
		cudaDeviceSynchronize();
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