#include <iostream>
#include <algorithm>
#include <string>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cpuGraph.h"
#include "globals.cuh"
#include "io.cuh"
#include "order.h"
// #include "join.cuh"
#include "decycle.h"
#include "cuda_helpers.h"
#include "gpu_match.cuh"
#include "memManag.cuh"
#include "res_table.hpp"

#include "structure.cuh"
#include "filter.cuh"

#include "CLI11.hpp"

using std::cout;
using std::endl;

void info(const char *s)
{
  // printf("%s done\n", s);
  // std::cout << s << " done" << std::endl;
}

int main(int argc, char **argv)
{
  CLI::App app{"App description"};

  std::string query_path, data_path;
  uint32_t gpu_num = 0u;
  app.add_option("-q", query_path, "query graph path")->required();
  app.add_option("-d", data_path, "data graph path")->required();
  app.add_option("--gpu", gpu_num, "gpu number");

  CLI11_PARSE(app, argc, argv);

  int device = gpu_num;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

#ifndef NDEBUG
  cout << "Device " << device << ": " << prop.name << endl;
#endif
  cudaSetDevice(device);
  GPU_NUM = gpu_num;

  info(std::string("query: " + query_path).c_str());

  cpuGraph hq_backup;
  gpuGraph dq_backup;
  cpuGraph hq, hg;
  gpuGraph dq, dg;
  hq_backup.isQuery = true;
  hq.isQuery = true;
  hg.isQuery = false;
  // gpuGraph dg;
  readGraphToCPUDynamic(&hq_backup, query_path.c_str());
  auto startTime = std::chrono::steady_clock::now();
  readGraphToCPUDynamic(&hq, query_path.c_str());
  readGraphToCPUDynamic(&hg, data_path.c_str());
  auto endTime = std::chrono::steady_clock::now();
  std::cout << "read to csr (seconds):" << std::endl;
	//秒
	double csr_duration_second = std::chrono::duration<double>(endTime - startTime).count();
	std::cout << csr_duration_second << "秒" << std::endl;
  hq.keep = new bool[hq.num_e * 2];
  memset(hq.keep, false, sizeof(bool) * hq.num_e * 2);
  hq_backup.keep = new bool[hq_backup.num_e * 2];
  memset(hq_backup.keep, false, sizeof(bool) * hq_backup.num_e * 2);

  info("read graph");
 
  copyMeta(&hq_backup, &hg);
  info("copy meta");

  // chage hq into spanning tree.
  decycle(&hq);

  info("decycle");
#ifndef NDEBUG
  hq.Print();
#endif
  GlobalView global = buildGlobalView(hq, hg);
 #ifndef NDEBUG
  printCandidates(global.candidates);
  printAdjacency(global.adjacency);
#endif
  // allocate query graph
  allocateMemGPU(&dq_backup, &hq_backup);
  allocateMemGPU(&dq, &hq);
  info("allocate");
  auto beforeTime = std::chrono::steady_clock::now();
  copyGraphToGPU(&dq_backup, &hq_backup);
  copyGraphToGPU(&dq, &hq);
  info("copy");
  
  auto afterTime = std::chrono::steady_clock::now();
	std::cout << "data to gpu total (seconds):" << std::endl;
	//秒
	double togpu_duration_second = std::chrono::duration<double>(afterTime - beforeTime).count();
	std::cout << togpu_duration_second << "秒" << std::endl;
  std::cout << csr_duration_second + togpu_duration_second<< "秒（csr + toGPU time）" << std::endl;
  vtype *h_u_candidate_vs_ = nullptr;
  vtype *d_u_candidate_vs_ = nullptr;
  vtype *h_v_candidate_us_ = nullptr; // the same as `bitmap_reverse`
  vtype *d_v_candidate_us_ = nullptr; // the same as `d_bitmap_reverse`

  numtype *h_num_u_candidate_vs_ = nullptr;
  numtype *d_num_u_candidate_vs_ = nullptr;
  numtype *h_num_v_candidate_us_ = nullptr;
  numtype *d_num_v_candidate_us_ = nullptr;

  h_u_candidate_vs_ = new vtype[NUM_VQ * MAX_L_FREQ];
  cuchk(cudaMalloc((void **)&d_u_candidate_vs_, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));
  h_v_candidate_us_ = new vtype[NUM_VD];
  cuchk(cudaMalloc((void **)&d_v_candidate_us_, sizeof(vtype) * NUM_VD));
  // memset(h_u_candidate_vs_, -1, sizeof(vtype) * NUM_VQ * MAX_L_FREQ);
  // cuchk(cudaMemset(d_u_candidate_vs_, -1, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));
  // memset(h_v_candidate_us_, -1, sizeof(vtype) * NUM_VD);
  // cuchk(cudaMemset(d_v_candidate_us_, -1, sizeof(vtype) * NUM_VD));

  h_num_u_candidate_vs_ = new numtype[NUM_VQ];
  cuchk(cudaMalloc((void **)&d_num_u_candidate_vs_, sizeof(numtype) * NUM_VQ));
  h_num_v_candidate_us_ = new numtype[NUM_VD];
  cuchk(cudaMalloc((void **)&d_num_v_candidate_us_, NUM_VD * sizeof(numtype)));
  // memset(h_num_u_candidate_vs_, 0, sizeof(numtype) * NUM_VQ);
  // cuchk(cudaMemset((void **)d_num_u_candidate_vs_, 0, sizeof(numtype) * NUM_VQ));
  // memset(h_num_v_candidate_us_, 0, sizeof(numtype) * NUM_VD);
  // cuchk(cudaMemset(d_num_v_candidate_us_, 0, NUM_VD * sizeof(numtype)));

  uint32_t *d_compact_encodings_; // column first, compacted.
  cuchk(cudaMalloc((void **)&d_compact_encodings_, sizeof(uint32_t) * NUM_VQ * COL_LEN));
  uint32_t *h_compact_encodings_ = new uint32_t[NUM_VQ * COL_LEN];
  int *d_enc_pos_u;
  cuchk(cudaMalloc((void **)&d_enc_pos_u, sizeof(int) * NUM_VQ));

  TIME_INIT();
  micro_init();

  double total_update_time = 0;
  double total_filter_time = 0;
  for (int batch_id = 0; batch_id < hg.num_batches; ++batch_id)
  {
    // 1. update data graph
    auto batch = hg.batches[batch_id];
#ifndef NDEBUG
    for (size_t i = 0; i < batch.size(); ++i) {
      if (i > 0) std::cout << ", ";  // 用逗号分隔多个change
      std::cout << "[" << batch[i].src << "->" << batch[i].dst << "]";
  }
#endif
    micro_start();
    updateHostGraph(hg, batch);
    //
    updateGlobalView(global, hq,hg, batch);
#ifndef NDEBUG
  printCandidates(global.candidates);
  printAdjacency(global.adjacency);
#endif
    micro_end();
    micro_print_local("update");
    total_update_time += diff_micro.tv_nsec / 1000.0;
#ifndef NDEBUG
    hg.Print();
#endif

    copyDataMeta(&hg);

    allocateMemGPU(&dg, &hg, batch_id == 0);
    copyGraphToGPU(&dg, &hg, batch_id == 0);

    cuchk(cudaDeviceSynchronize());

    // 2. re-filter

    memset(h_u_candidate_vs_, -1, sizeof(vtype) * NUM_VQ * MAX_L_FREQ);
    cuchk(cudaMemset(d_u_candidate_vs_, -1, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));
    memset(h_v_candidate_us_, -1, sizeof(vtype) * NUM_VD);
    cuchk(cudaMemset(d_v_candidate_us_, -1, sizeof(vtype) * NUM_VD));

    memset(h_num_u_candidate_vs_, 0, sizeof(numtype) * NUM_VQ);
    cuchk(cudaMemset(d_num_u_candidate_vs_, 0, sizeof(numtype) * NUM_VQ));
    memset(h_num_v_candidate_us_, 0, sizeof(numtype) * NUM_VD);
    cuchk(cudaMemset(d_num_v_candidate_us_, 0, NUM_VD * sizeof(numtype)));

    // cluster structures.
    std::vector<cpuCluster> cpu_clusters_;
    // cpuCluster *cpu_clusters_ = nullptr;
    // gpuCluster *gpu_clusters_ = nullptr;
    encodingMeta enc_meta;

    uint32_t *h_encodings_ = nullptr;
    uint32_t *d_encodings_ = nullptr;

    TIME_START();

    micro_start();

#ifndef NDEBUG
    std::cout << "filter start" << std::endl;
#endif
    // auto start = std::chrono::system_clock::now();
    clusterFilter(&hq_backup, &dq_backup, &hq, &hg, &dq, &dg,
                  // cluster related
                  cpu_clusters_,
                  h_encodings_, d_encodings_, &enc_meta,
                  // return
                  h_u_candidate_vs_, h_num_u_candidate_vs_,
                  d_u_candidate_vs_, d_num_u_candidate_vs_,
                  h_v_candidate_us_, h_num_v_candidate_us_,
                  d_v_candidate_us_, d_num_v_candidate_us_);

    micro_end();
    micro_print_local("filter");

    total_filter_time += diff_micro.tv_nsec / 1000.0;

    TIME_END();
    PRINT_LOCAL_TIME("FILTER");

#ifndef NDEBUG
    std::cout << std::dec << std::endl;
    std::cout << "filter done" << std::endl;
    enc_meta.print();

#endif

    /* free */
    // cudaFree(dq.degree_);
    // cudaFree(dq.edgeIDs_);
    // cudaFree(dq.neighbors_);
    // cudaFree(dq.offsets_);
    // cudaFree(dq.vLabels_);
    // cudaFree(d_v_candidate_us_);
    // cudaFree(d_num_v_candidate_us_);

    // delete[] h_u_candidate_vs_;
    // delete[] h_v_candidate_us_;
    // delete[] h_num_v_candidate_us_;
    // delete[] h_encodings_;

    cuchk(cudaMemset(d_compact_encodings_, 0, sizeof(uint32_t) * NUM_VQ * COL_LEN));
    cuchk(cudaMemcpy(d_enc_pos_u, enc_meta.enc_pos_of_u_, sizeof(int) * NUM_VQ, cudaMemcpyHostToDevice));
    
    compact<<<GRID_DIM, BLOCK_DIM>>>(d_compact_encodings_, d_encodings_, d_enc_pos_u);
    cuchk(cudaDeviceSynchronize());
    cuchk(cudaMemcpy(h_compact_encodings_, d_compact_encodings_, sizeof(uint32_t) * NUM_VQ * COL_LEN, cudaMemcpyDeviceToHost));

    ResTable res_table(MAX_RES);
    res_table.cur_level = 2;

    // Order
    OrderCPU order_obj;
    vtype src, dst; // 数据图中的顶点
    std::map<std::pair<vtype, vtype>, int> initial_edge_2_order_id;
    std::vector<vtype> temp_vec(2);
    int order_id = 0;
    for (int i = 0; i < hg.batch_size[batch_id]; ++i)
    {
      if (hg.batches[batch_id][i].add == false)
        continue;
      src = hg.batches[batch_id][i].src;
      dst = hg.batches[batch_id][i].dst;
      
      std::vector<vtype> src_can_u, dst_can_u;
      for (vtype u = 0; u < NUM_VQ; ++u)
      {
        if (h_compact_encodings_[u * COL_LEN + src / BLK_SIZE] & (1 << (src % BLK_SIZE)))
          src_can_u.push_back(u);
        if (h_compact_encodings_[u * COL_LEN + dst / BLK_SIZE] & (1 << (dst % BLK_SIZE)))
          dst_can_u.push_back(u);
        // 上述代码是找到更新的数据顶点映射到的查询图顶点
      }

      // std::cout << "src: " << src << " dst: " << dst << std::endl;
      // for (auto u : src_can_u)
      //   std::cout << u << " ";
      // std::cout << std::endl;
      // for (auto u : dst_can_u)
      //   std::cout << u << " ";
      // std::cout << std::endl;
      // 遍历当前更新边映射到的查询图顶点
      for (auto u : src_can_u)
      {
        for (auto v : dst_can_u)
        {
          if (is_neighbor(&hq_backup, u, v))
          {
            // std::cout << "u: " << u << " v: " << v << " is neighbor." << std::endl;
            if (initial_edge_2_order_id.find(std::make_pair(u, v)) == initial_edge_2_order_id.end())
            { // 对查询边进行编号
              int x = initial_edge_2_order_id.size();
              initial_edge_2_order_id[std::make_pair(u, v)] = x;
            }
            // if (initial_edge_2_order_id.find(std::make_pair(u, v)) != initial_edge_2_order_id.end())
            // {
            order_id = initial_edge_2_order_id[std::make_pair(u, v)];
            temp_vec[0] = src;
            temp_vec[1] = dst;
            // }
            // else
            // {
            //   order_id = initial_edge_2_order_id[std::make_pair(v, u)];
            //   temp_vec[0] = v;
            //   temp_vec[1] = u;
            // }
            // std::cout << "u: " << u << " v: " << v << std::endl;
            bool res = res_table.insert(order_id, temp_vec);
            if (!res)
            {
              std::cout << "insert failed.\n"
                        << "current size: " << res_table.size
                        << std::endl;
              exit(1);
            }
          }
        }
      }
    }// kv_pair 是 k <src,dst> v orderid
    // for (auto kv_pair : initial_edge_2_order_id)
    // {
    //   order_obj.roots.push_back(kv_pair.first.first);
    //   order_obj.sub_roots.push_back(kv_pair.first.second);
    // }
    std::vector<LocalView> local_views;
    for (const auto& [edge, order_id] : initial_edge_2_order_id)
    {
        order_obj.roots.push_back(edge.first);
        order_obj.sub_roots.push_back(edge.second);
        order_obj.computeMatchingOrder(&hq, edge);
        local_views.push_back(buildLocalView(hq, global, {src, dst}, edge));
    } 

    std::vector<GPULocalView> host_gpu_views;
    GPULocalView* d_gpu_views = transferAllLocalViewsToGPU(local_views,host_gpu_views);
    
  return 0;
}