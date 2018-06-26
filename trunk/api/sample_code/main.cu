#include <iostream>
#include "graph.h"
//#include <thrust/sort.h>
//#include <thrust/execution_policy.h>

#include "gpu_radix_sort.cuh"

int main(int args, char **argv)
{
	std::cout<<"Input: ./exe beg csr weight\n";
	if(args!=4){std::cout<<"Wrong input\n"; return -1;}
	
	const char *beg_file=argv[1];
	const char *csr_file=argv[2];
	const char *weight_file=argv[3];
	
    typedef int vertex_t;
    typedef long index_t;

	//template <file_vertex_t, file_index_t, file_weight_t
	//new_vertex_t, new_index_t, new_weight_t>
	graph<int, long, int, vertex_t, index_t, int>
	*ginst = new graph
	<int, long, int, vertex_t, index_t, int>
	(beg_file,csr_file,weight_file);
   
    vertex_t **sort_container_d;
    gpu_allocator(sort_container_d, args, argv);
    //host_allocator(sort_container_h); 

    radix_sort_sync(ginst->csr, sort_container_d, ginst->edge_count);

   // double time = wtime();
   // //thrust::sort(thrust::host, ginst->csr, ginst->csr+ginst->edge_count);
   // thrust::sort(thrust::device, ginst->csr, ginst->csr+ginst->edge_count);
   // time = wtime() - time;
   // 
   // for(int i= 0; i < ginst->edge_count - 1; i++)
   //     assert(ginst->csr[i+1] >= ginst->csr[i]);

   // std::cout<<"Sort result correct! Time: "<<time<<" second(s).\n";
   // 

   // vertex_t *adj_list_d;
   // cudaMalloc((void **)&adj_list_d, sizeof(vertex_t)*ginst->edge_count);
   // 
   // 
   // time = wtime();
   // cudaMemcpy(adj_list_d, ginst->csr, sizeof(vertex_t)*ginst->edge_count, cudaMemcpyHostToDevice);
   // cudaDeviceSynchronize();
   // cudaMemcpy(ginst->csr, adj_list_d, sizeof(vertex_t)*ginst->edge_count, cudaMemcpyDeviceToHost);
   // cudaDeviceSynchronize();
   // time = wtime() - time;
   // std::cout<<"Data copy time: "<<time<<" second(s).\n";


    //**
    //You can implement your single threaded graph algorithm here.
    //like BFS, SSSP, PageRank and etc.
    
    //for(int i = 0; i < ginst->vert_count+1; i++)
    //{
    //    int beg = ginst->beg_pos[i];
    //    int end = ginst->beg_pos[i+1];
    //    std::cout<<i<<"'s neighor list: ";
    //    for(int j = beg; j < end; j++)
    //        std::cout<<ginst->csr[j]<<" ";
    //    std::cout<<"\n";
    //} 
    


	return 0;	
}
