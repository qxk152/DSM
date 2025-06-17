#include "gpuGraph.h"

gpuGraph::gpuGraph()
{
  // elCount = 0;
  degree_ = nullptr;
  vLabels_ = nullptr;
  // eLabels = nullptr;
  offsets_ = nullptr;
  neighbors_ = nullptr;
  edgeIDs_ = nullptr;
  // src_vtx = nullptr;
}
// 将单个 LocalView 转换为 GPULocalView
GPULocalView transferLocalViewToGPU(const LocalView& local_view) {
    const int num_vertices = local_view.vertices.size();
    
    // 准备主机端数据
    std::vector<vtype> h_vertices = local_view.vertices;
    std::vector<numtype> h_offsets(num_vertices + 1);
    std::vector<vtype> h_candidates;
    
    // 计算偏移量
    h_offsets[0] = 0;
    for (int i = 0; i < num_vertices; ++i) {
        vtype u = local_view.vertices[i];
        const auto& candidates = local_view.candidate_map.at(u);
        h_offsets[i + 1] = h_offsets[i] + candidates.size();
        h_candidates.insert(h_candidates.end(), candidates.begin(), candidates.end());
    }
    
    // 创建 GPU 视图
    GPULocalView gpu_view;
    gpu_view.num_vertices = num_vertices;
    gpu_view.total_candidates = h_candidates.size();
    
    // 分配并传输顶点数组
    cudaMalloc(&gpu_view.d_vertices, sizeof(vtype) * num_vertices);
    cudaMemcpy(gpu_view.d_vertices, h_vertices.data(), 
               sizeof(vtype) * num_vertices, cudaMemcpyHostToDevice);
    
    // 分配并传输候选集数组
    cudaMalloc(&gpu_view.d_candidates, sizeof(vtype) * gpu_view.total_candidates);
    cudaMemcpy(gpu_view.d_candidates, h_candidates.data(), 
               sizeof(vtype) * gpu_view.total_candidates, cudaMemcpyHostToDevice);
    
    // 分配并传输偏移数组
    cudaMalloc(&gpu_view.d_offsets, sizeof(numtype) * (num_vertices + 1));
    cudaMemcpy(gpu_view.d_offsets, h_offsets.data(),
               sizeof(numtype) * (num_vertices + 1), cudaMemcpyHostToDevice);
    
    return gpu_view;
}

// 将整个 LocalView 数组传输到 GPU
GPULocalView* transferAllLocalViewsToGPU(const std::vector<LocalView>& local_views, 
                                         std::vector<GPULocalView>& host_gpu_views) {
    // 1. 为每个 LocalView 创建 GPU 表示
    host_gpu_views.clear();
    for (const auto& local_view : local_views) {
        host_gpu_views.push_back(transferLocalViewToGPU(local_view));
    }
    
    // 2. 在设备上分配 GPULocalView 数组
    GPULocalView* d_gpu_views;
    cudaMalloc(&d_gpu_views, sizeof(GPULocalView) * host_gpu_views.size());
    
    // 3. 将主机上的 GPULocalView 数组复制到设备
    cudaMemcpy(d_gpu_views, host_gpu_views.data(), 
               sizeof(GPULocalView) * host_gpu_views.size(), 
               cudaMemcpyHostToDevice);
    
    return d_gpu_views;
}
gpuGraph::~gpuGraph()
{
  // if (degree != nullptr)
  //   delete[] degree;
  // if (vLabels != nullptr)
  //   delete[] vLabels;
  // // if (eLabels != nullptr)
  // //   delete[] eLabels;
  // if (vertexIDs_ != nullptr)
  //   delete[] vertexIDs_;
  // if (offsets_ != nullptr)
  //   delete[] offsets_;
  // if (neighbors_ != nullptr)
  //   delete[] neighbors_;
  // if (edgeIDs_ != nullptr)
  //   delete[] edgeIDs_;
}