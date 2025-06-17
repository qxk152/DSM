#ifndef GPUGRAPH_H
#define GPUGRAPH_H

#include <cinttypes>

#include "cpuGraph.h"
#include "globals.cuh"

class gpuGraph
{
public:
  uint32_t *degree_; // arr
  vltype *vLabels_;  // arr
  // eltype *eLabels; // arr

  // CSR
  uint32_t *offsets_; // arr
  vtype *neighbors_;  // arr
  etype *edgeIDs_;    // arr
  // vtype *src_vtx;     // arr

  gpuGraph();
  ~gpuGraph();
};

struct GPULocalView {
    vtype* d_vertices;        // 设备上的顶点顺序数组
    vtype* d_candidates;      // 设备上的候选集数组 (CSR 格式)
    numtype* d_offsets;       // 设备上的偏移数组 (CSR 格式)
    numtype num_vertices;     // 顶点数量
    numtype total_candidates; // 候选总数
};
GPULocalView transferLocalViewToGPU(const LocalView& local_view);
GPULocalView* transferAllLocalViewsToGPU(const std::vector<LocalView>& local_views, 
                                         std::vector<GPULocalView>& host_gpu_views);
#endif