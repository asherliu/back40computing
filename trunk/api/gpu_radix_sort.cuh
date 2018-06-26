#include <assert.h>
#include "../b40c/util/multiple_buffering.cuh"
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <math.h> 
#include <float.h>
#include <algorithm>
#include "../b40c/util/error_utils.cuh"
#include "../b40c/radix_sort/enactor.cuh"
#include "../test/b40c_test_util.h"
#include <stdio.h>

using namespace b40c;
typedef int comp_t;
typedef int sa_t;
#define MAX_NUM_ELEMENTS (1<<28)


//global variable
static void HandleError( cudaError_t err,
		const char *file,
		int line ) { 
	if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", \
				cudaGetErrorString( err ),
				file, line );
		exit( EXIT_FAILURE );
	}   
}
#define H_ERR( err ) \
	(HandleError( err, __FILE__, __LINE__ ))

void 
gpu_allocator(comp_t** &keys_d, int argc, char **argv){
	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	//pre-alloc data to avoid alloc @ runtime
	keys_d=new comp_t*[2];
	for(long i=0;i<2;i++)
	{
		H_ERR(cudaMalloc((void**) &keys_d[i], 
			sizeof(comp_t)*MAX_NUM_ELEMENTS));
	}
}

void 
gpu_allocator_kv(comp_t** &keys_d, sa_t ** &vals_d, 
							int argc, char **argv){
	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	//pre-alloc data to avoid alloc @ runtime
	keys_d=new comp_t*[2];
	for(long i=0;i<2;i++)
	{
		H_ERR(cudaMalloc((void**) &keys_d[i], 
			sizeof(comp_t)*MAX_NUM_ELEMENTS));
	}
	
	vals_d=new sa_t*[2];
	for(long i=0;i<2;i++)
	{
		H_ERR(cudaMalloc((void**) &vals_d[i], 
			sizeof(sa_t)*MAX_NUM_ELEMENTS));
	}
}

void host_allocator_kv(
			comp_t* &key_h,
			sa_t* &val_h)
{
	H_ERR(cudaMallocHost((void**) &key_h, 
				sizeof(comp_t)*MAX_NUM_ELEMENTS));
	H_ERR(cudaMallocHost((void**) &val_h, 
				sizeof(sa_t)*MAX_NUM_ELEMENTS));
}

void host_allocator(comp_t* &key_h)
{
	H_ERR(cudaMallocHost((void**) &key_h, 
				sizeof(comp_t)*MAX_NUM_ELEMENTS));
}


void
gpu_kv_freer(
		comp_t **keys_d,
		sa_t **vals_d)
{
	cudaFree(keys_d[0]);
	cudaFree(keys_d[1]);
	cudaFree(vals_d[0]);
	cudaFree(vals_d[1]);
}

void
gpu_freer(comp_t **keys_d)
{
	cudaFree(keys_d[0]);
	cudaFree(keys_d[1]);
}

void
host_kv_freer(
		comp_t *key_h,
		sa_t *val_h
){
	cudaFreeHost(key_h);
	cudaFreeHost(val_h);
}

void
host_freer(comp_t *key_h)
{
	cudaFreeHost(key_h);
}


void 
radix_sort_kv_sync(
	comp_t *keys_h,
	comp_t **keys_d,
	sa_t *vals_h,
	sa_t **vals_d,
	long num_elements
){

	GpuTimer timer;
	double elapsed = 0;
	radix_sort::Enactor sorting_enactor;
	
	util::DoubleBuffer<comp_t, sa_t> device_storage;
	device_storage.d_keys[0]=keys_d[0];
	device_storage.d_keys[1]=keys_d[1];
	device_storage.d_values[0]=vals_d[0];
	device_storage.d_values[1]=vals_d[1];
	
	//using the spare neighbors array for sorting
	H_ERR(cudaMemcpy(device_storage.d_keys[0],keys_h, 
					sizeof(comp_t)*num_elements, cudaMemcpyHostToDevice));

	H_ERR(cudaMemcpy(device_storage.d_values[0],vals_h, 
					sizeof(sa_t)*num_elements, cudaMemcpyHostToDevice));

	// Marker kernel in profiling stream
	util::FlushKernel<void><<<1,1>>>();

	// Start cuda timing record
	timer.Start();

	// Call the sorting API routine
	sorting_enactor.template Sort<radix_sort::LARGE_SIZE>
		(device_storage, num_elements, 1024);

	// End cuda timing record
	timer.Stop();
	elapsed += (double) timer.ElapsedMillis();

	double avg_runtime = elapsed;
//	double throughput = ((double)num_elements)/avg_runtime/1000.0/1000.0;
//	printf(", %f GPU ms, %f x10^9 elts/sec\n", 
//			avg_runtime, throughput);

	// Copy out data
	H_ERR(cudaMemcpy(keys_h, 
				device_storage.d_keys[device_storage.selector], 
				sizeof(comp_t)*num_elements, cudaMemcpyDeviceToHost));
	H_ERR(cudaMemcpy(vals_h, 
				device_storage.d_values[device_storage.selector], 
				sizeof(sa_t)*num_elements, cudaMemcpyDeviceToHost));
}


void 
radix_sort_sync(
	comp_t *keys_h,
	comp_t **keys_d,
	long num_elements
){

	GpuTimer timer;
	double elapsed = 0;
	radix_sort::Enactor sorting_enactor;
	
	util::DoubleBuffer<comp_t> device_storage;
	device_storage.d_keys[0]=keys_d[0];
	device_storage.d_keys[1]=keys_d[1];
	
	//using the spare neighbors array for sorting
	H_ERR(cudaMemcpy(device_storage.d_keys[0],keys_h, 
					sizeof(comp_t)*num_elements, cudaMemcpyHostToDevice));

	// Marker kernel in profiling stream
	util::FlushKernel<void><<<1,1>>>();

	// Start cuda timing record
	timer.Start();

	// Call the sorting API routine
	sorting_enactor.template Sort<radix_sort::LARGE_SIZE>
		(device_storage, num_elements, 1024);

	// End cuda timing record
	timer.Stop();
	elapsed += (double) timer.ElapsedMillis();

	double avg_runtime = elapsed;
	double throughput = ((double)num_elements)/avg_runtime/1000.0/1000.0;
	printf(", %f GPU ms, %f x10^9 elts/sec\n", 
			avg_runtime, throughput);

	// Copy out data
	H_ERR(cudaMemcpy(keys_h, 
				device_storage.d_keys[device_storage.selector], 
				sizeof(comp_t)*num_elements, cudaMemcpyDeviceToHost));
}
