#ifndef CPUGRAPH_H
#define CPUGRAPH_H

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "defs.h"
#include <optional> 
class change
{
public:
  bool add = true; // false for edge deletion
  vtype src, dst,labelChange;
};



class cpuGraph
{
public:
  numtype num_v;
  numtype num_e;
  numtype largest_l;
  degtype maxDegree;

  degtype *indeg_;  // in degree. for CSC, unused for now.
  degtype *outdeg_; // outdegree. for CSR
  vltype *vLabels_; // size = num_v
  numtype maxLabelFreq;

  // CSR
  vtype *vertexIDs_; // size = num_v
  offtype *offsets_; // size = num_v + 1
  vtype *neighbors_; // size = num_e * 2
  etype *edgeIDs_;   // size = num_e * 2
  // vtype *src_vtx;    // size = num_e * 2

  bool isQuery;
  bool *keep;

  std::map<std::pair<vtype, vtype>, etype> vve;
  std::map<etype, std::pair<vtype, vtype>> evv; // e is `eid`, for an undirected edge, eids are different.
  
  // for dynamic graphs
  int num_batches = 0;
  std::vector<int> batch_size;
  std::vector<std::vector<change>> batches;
  


public:
  cpuGraph();
  ~cpuGraph();

  void Print();

  offtype get_u_off(vtype u);
  
  
  
};
// 全局候选视图数据结构
struct GlobalView {
    std::vector<std::vector<vtype>> candidates;  // 各顶点候选集
    std::unordered_map<vtype, std::unordered_set<vtype>> adjacency;  
    // 反向索引（从数据顶点映射到查询顶点）
    std::vector<std::unordered_set<vtype>> dataVertexToQuery;
    
};

  // 局部候选视图数据结构
struct LocalView {
  std::vector<vtype> vertices;  // 所有需要处理的查询图顶点
  std::unordered_map<vtype, std::vector<vtype>> candidate_map;  // 查询图顶点候选集
  // 邻接关系：存储每个候选顶点的有效邻居
  std::unordered_map<vtype, std::unordered_set<vtype>> adjacency;  
};

class GraphUtils
{
public:
  uint8_t eidx_[MAX_VQ * MAX_VQ]; // actually it is a 2-d array.
  uint16_t nbrbits_[MAX_VQ];

public:
  void Set(const cpuGraph &g);
};
GlobalView buildGlobalView(const cpuGraph& query, const cpuGraph& data);
LocalView buildLocalView(const cpuGraph& QG, const GlobalView& global,
                        const std::pair<vtype, vtype>& data_edge,
                        const std::pair<vtype, vtype>& query_edge);
bool areAdjacent(const std::unordered_map<vtype, std::unordered_set<vtype>>& adj,vtype v0, vtype v1); 
void updateHostGraph(cpuGraph &graph, std::vector<change> &batch);
void updateGlobalView(GlobalView& globalView, 
                     const cpuGraph& queryGraph,
                     const cpuGraph& dataGraph,
                     const std::vector<change>& batchChanges);
void printCandidates(const std::vector<std::vector<vtype>>& candidates);

void printAdjacency(const std::unordered_map<vtype, std::unordered_set<vtype>>& adjacency);
#endif