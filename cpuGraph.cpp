#include "cpuGraph.h"
#include "globals.cuh"

#include <cstddef>
#include <iostream>
#include <cstring>
#include <set>

cpuGraph::cpuGraph() : vve()
{
	num_v = 0;
	num_e = 0;
	largest_l = 0;
	// elCount = 0;
	maxDegree = 0;

	outdeg_ = nullptr;
	vLabels_ = nullptr;
	maxLabelFreq = 0;
	// eLabels = nullptr;

	vertexIDs_ = nullptr;
	offsets_ = nullptr;
	neighbors_ = nullptr;
	edgeIDs_ = nullptr;
	isQuery = true;
	keep = nullptr;
}



cpuGraph::~cpuGraph()
{
	if (outdeg_ != nullptr)
		delete[] outdeg_;
	if (vLabels_ != nullptr)
		delete[] vLabels_;
	if (vertexIDs_ != nullptr)
		delete[] vertexIDs_;
	if (offsets_ != nullptr)
		delete[] offsets_;
	if (neighbors_ != nullptr)
		delete[] neighbors_;
	if (edgeIDs_ != nullptr)
		delete[] edgeIDs_;
	if (keep != nullptr)
		delete[] keep;
}

GlobalView buildGlobalView(const cpuGraph& query, const cpuGraph& data) {
    GlobalView global;
    global.candidates.resize(query.num_v);
    for (vtype i = 0; i < data.num_v; ++i) {
    	global.adjacency.emplace(i, std::unordered_set<vtype>{});  // 显式插入
	}

    // 第一层过滤：标签匹配且度数过滤
    for (vtype u_q = 0; u_q < query.num_v; ++u_q) {
        const auto& q_label = query.vLabels_[u_q];
        const auto& q_outdeg = query.outdeg_[u_q];
        
        for (vtype v_d = 0; v_d < data.num_v; ++v_d) {
            if (data.vLabels_[v_d] == q_label && data.outdeg_[v_d] >= q_outdeg) {
                global.candidates[u_q].push_back(v_d);
            }
        }
        std::sort(global.candidates[u_q].begin(), global.candidates[u_q].end());
    }

    // 预处理：建立数据图顶点到查询顶点的映射
    std::vector<std::unordered_set<vtype>> data_to_query(data.num_v);
    for (vtype u_q = 0; u_q < query.num_v; ++u_q) {
        for (const auto& v_d : global.candidates[u_q]) {
            data_to_query[v_d].insert(u_q);
        }
    }
    global.dataVertexToQuery = std::move(data_to_query); 
    // 构建邻接关系
	for (vtype u_d = 0; u_d < data.num_v; ++u_d) {
		// 获取u_d对应的所有查询顶点候选（即哪些查询顶点的候选包含u_d）
		const auto& related_queries = global.dataVertexToQuery[u_d];
		
		// 遍历数据图中u_d的所有出边
		for (offtype j = data.offsets_[u_d]; j < data.offsets_[u_d+1]; ++j) {
			vtype v_d = data.neighbors_[j];
			
			// 需要找到满足以下条件的查询边：
			// 1. 存在查询顶点u_q ∈ related_queries
			// 2. 存在查询边u_q -> v_q
			// 3. v_d ∈ candidates[v_q]
			std::unordered_set<vtype> valid_queries;
			
			// 遍历u_d关联的所有查询顶点
			for (vtype u_q : related_queries) {
				// 遍历u_q在查询图中的所有邻居v_q
				for (offtype k = query.offsets_[u_q]; k < query.offsets_[u_q+1]; ++k) {
					vtype v_q = query.neighbors_[k];
					
					// 检查v_d是否是v_q的候选
					if (std::binary_search(global.candidates[v_q].begin(),
										global.candidates[v_q].end(),
										v_d)) {
						// 记录满足条件的候选对
						valid_queries.insert(v_q);
					}
				}
			}
			
			// 将有效邻接关系存入数据结构
			if (!valid_queries.empty()) {
				global.adjacency[u_d].insert(v_d); // 记录u_d到v_d的边满足查询模式
			}
		}
	}

    return global;
}

// 检查 v0 和 v1 是否连接
bool areAdjacent(const std::unordered_map<vtype, std::unordered_set<vtype>>& adj,
                vtype v0, vtype v1) {
    // 检查 v0 -> v1
    auto it_v0 = adj.find(v0);
    if (it_v0 == adj.end()) return false;  // v0 不存在
    
    // 使用 find() 检查 v1 是否在 v0 的邻接表中
    if (it_v0->second.find(v1) == it_v0->second.end()) {
        return false;  // v0 和 v1 不相连
    }
    
    return true;
}

// 局部候选视图
LocalView buildLocalView(const cpuGraph& QG, const GlobalView& global,
                        const std::pair<vtype, vtype>& data_edge,
                        const std::pair<vtype, vtype>& query_edge) {
    LocalView local;
    std::unordered_set<vtype> related_vertices;
	const uint32_t u0 = query_edge.first;
	const uint32_t u1 = query_edge.second;
	const uint32_t v0 = data_edge.first;
	const uint32_t v1 = data_edge.second;
    // Stet1：验证变关系
	if(!areAdjacent(global.adjacency,v0,v1)){
		return local;  // 显式构造空值
	}
 	// 固定核心顶点映射
    // local.candidate_map[u0] = {v0};
    // local.candidate_map[u1] = {v1};
    // Step 2
    // 分类顶点
    std::vector<vtype> connected_both;   // 与u1和u2都相连
    std::vector<vtype> connected_either; // 与u1或u2的一个相连
    std::vector<vtype> connected_none;  // 都不相连
    
     // 分类顶点
    for (vtype u = 0; u < QG.num_v; ++u) {
        if (u == u0 || u == u1) continue;

        bool conn_u0 = false;
        bool conn_u1 = false;
        
        // 检查查询图中的连接关系
        for (offtype i = QG.offsets_[u]; i < QG.offsets_[u+1]; ++i) {
            vtype nbr = QG.neighbors_[i];
            if (nbr == u0) conn_u0 = true;
            if (nbr == u1) conn_u1 = true;
        }

        if (conn_u0 && conn_u1) {
            connected_both.push_back(u);
        } else if (conn_u0 || conn_u1) {
            connected_either.push_back(u);
        } else {
            connected_none.push_back(u);
        }
    }

    // 处理与u0和u1都相连的顶点
    for (vtype u : connected_both) {
        std::vector<vtype> candidates;
        for (vtype v : global.candidates[u]) {
            // 必须同时与v0和v1相连
            bool valid = (global.adjacency.at(v).count(v0) > 0) && 
                         (global.adjacency.at(v).count(v1) > 0);
            if (valid) candidates.push_back(v);
        }
        if (!candidates.empty()) {
            local.candidate_map[u] = candidates;
        }
    }

	 // 处理与u0或u1相连的顶点
	for (vtype u : connected_either) {
		std::vector<vtype> candidates;
		
		// 检查当前顶点在查询图中与u0/u1的连接情况
		bool conn_u0 = false;
		bool conn_u1 = false;
		for (offtype i = QG.offsets_[u]; i < QG.offsets_[u+1]; ++i) {
			vtype nbr = QG.neighbors_[i];
			if (nbr == u0) conn_u0 = true;
			if (nbr == u1) conn_u1 = true;
		}

		for (vtype v : global.candidates[u]) {
			bool valid = true;
			// 如果与u0相连，必须与v0相连
			if (conn_u0 && global.adjacency.at(v).count(v0) == 0) {
				valid = false;
			}
			// 如果与u1相连，必须与v1相连
			if (conn_u1 && global.adjacency.at(v).count(v1) == 0) {
				valid = false;
			}
			
			if (valid) {
				candidates.push_back(v);
			}
		}
		
		if (!candidates.empty()) {
			local.candidate_map[u] = candidates;
		}
	}

    // 处理与u0和u1都不相连的顶点（保留全局候选）
    for (vtype u : connected_none) {
  
		if (u < global.candidates.size() && 
			!global.candidates[u].empty() &&
			local.candidate_map.count(u) == 0) {
			
			try {
				local.candidate_map[u] = global.candidates[u];
			} catch (const std::exception& e) {
				std::cerr << "Error assigning candidates for vertex " << u 
						<< ": " << e.what() << std::endl;
			}
		}
    }

	// step3：构建邻接关系
    // ----------------------------------------------------------
    for (const auto& [u, cands] : local.candidate_map) {
        for (vtype v : cands) {
            std::unordered_set<vtype> valid_neighbors;
            for (offtype i = QG.offsets_[u]; i < QG.offsets_[u+1]; ++i) {
                vtype q_nbr = QG.neighbors_[i];
                if (local.candidate_map.count(q_nbr)) {
                    for (vtype v_nbr : global.adjacency.at(v)) {
                        if (std::binary_search(local.candidate_map[q_nbr].begin(),
                                            local.candidate_map[q_nbr].end(),
                                            v_nbr)) {
                            valid_neighbors.insert(v_nbr);
                        }
                    }
                }
            }
            local.adjacency[v] = valid_neighbors;
        }
    }

    // 构建顶点列表
    local.vertices.reserve(local.candidate_map.size());
    for (const auto& [u, _] : local.candidate_map) {
        local.vertices.push_back(u);
    }

    return local;

}

void cpuGraph::Print()
{
	std::cout << "============================\n";
	std::cout << "num_v: " << num_v << std::endl;
	std::cout << "num_e: " << num_e << std::endl;
	std::cout << "largest_l: " << largest_l << std::endl;
	std::cout << "maxDegree: " << maxDegree << std::endl;

	std::cout << "outdeg_: \n";
	for (int i = 0; i < num_v; ++i)
		std::cout << outdeg_[i] << " \n"[i == num_v - 1];
	std::cout << "vLabels: \n";
	for (int i = 0; i < num_v; ++i)
		std::cout << vLabels_[i] << " \n"[i == num_v - 1];
	std::cout << "maxLabelFreq: " << maxLabelFreq << std::endl;

	std::cout << "vertexIDs_: \n";
	for (int i = 0; i < num_v; ++i)
		std::cout << vertexIDs_[i] << " \n"[i == num_v - 1];
	std::cout << "offsets_: \n";
	for (int i = 0; i < num_v + 1; ++i)
		std::cout << offsets_[i] << " \n"[i == num_v];
	std::cout << "neighbors_: \n";
	for (int i = 0; i < num_e * 2; ++i)
		std::cout << neighbors_[i] << " \n"[i == num_e * 2 - 1];
	std::cout << "edgeIDs_: \n";
	for (int i = 0; i < num_e * 2; ++i)
		std::cout << edgeIDs_[i] << " \n"[i == num_e * 2 - 1];
	std::cout << "============================" << std::endl;
}


bool hasCommonNeighbor(const std::vector<vtype>& candidates,
                      const cpuGraph& dataGraph, 
                      vtype dv) 
{
    const vtype* nbrs = dataGraph.neighbors_ + dataGraph.offsets_[dv];
    const size_t n = dataGraph.offsets_[dv+1] - dataGraph.offsets_[dv];
    
    for (vtype cand : candidates) {
        if (std::binary_search(nbrs, nbrs + n, cand)) {
            return true;
        }
    }
    return false;
}
// 更新全局候选视图
void updateGlobalView(GlobalView& global, 
                     const cpuGraph& queryGraph,
                     const cpuGraph& dataGraph,
                     const std::vector<change>& batchChanges) 
{
    // 阶段1：处理顶点级变更
    std::unordered_set<vtype> affectedVertices;
    
    // 收集受影响的顶点（映射图顶点对应超边）
    std::unordered_set<vtype> affectedDataVertices;
    for (const auto& chg : batchChanges) {
        affectedDataVertices.insert(chg.src);
        affectedDataVertices.insert(chg.dst);
    }

     // 阶段2：反向映射到查询顶点
	std::unordered_set<vtype> affectedQueryVertices;

	for (vtype dv : affectedDataVertices) {
		// 边界检查（确保dv在合法范围内）
		if (dv >= global.dataVertexToQuery.size()) continue;
		
		const auto& query_vertices = global.dataVertexToQuery[dv];
		affectedQueryVertices.insert(query_vertices.begin(), query_vertices.end());
	}
	
	// 阶段3：候选集增量更新
    for (vtype qu = 0; qu < queryGraph.num_v; ++qu) {
        if (!affectedQueryVertices.count(qu)) continue;

        std::vector<vtype> newCandidates;
        vltype qLabel = queryGraph.vLabels_[qu];
        degtype qDegree = queryGraph.outdeg_[qu];
        newCandidates.reserve(global.candidates[qu].size());

        // 双层过滤
        for (vtype dv = 0; dv < dataGraph.num_v; ++dv) {
            // 快速过滤：标签和度数
            if (dataGraph.vLabels_[dv] != qLabel || 
                dataGraph.outdeg_[dv] < qDegree) {
                continue;
            }
            
            // 结构验证（检查邻居约束）
            bool valid = true;
            for (offtype i = queryGraph.offsets_[qu]; i < queryGraph.offsets_[qu+1]; ++i) {
                vtype q_nbr = queryGraph.neighbors_[i];
                if (!hasCommonNeighbor(global.candidates[q_nbr], 
                                     dataGraph, dv)) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) newCandidates.push_back(dv);
        }

        // 更新候选集
        global.candidates[qu] = std::move(newCandidates);
    }

	// 阶段4：邻接关系更新 ===
    for (const auto& chg : batchChanges) {
        const vtype u = chg.src, v = chg.dst;
        
        if (chg.add) {
            // 添加双向邻接关系
            global.adjacency[u].insert(v);
            global.adjacency[v].insert(u);
        } else {
            // 删除双向邻接关系
            global.adjacency[u].erase(v);
            global.adjacency[v].erase(u);
        }
    }

	// 阶段5：维护反向索引
    for (vtype dv : affectedDataVertices) {
        if (dv >= global.dataVertexToQuery.size()) continue;
        
        global.dataVertexToQuery[dv].clear();
        for (vtype qu = 0; qu < queryGraph.num_v; ++qu) {
            if (std::binary_search(global.candidates[qu].begin(), 
                                 global.candidates[qu].end(), dv)) {
                global.dataVertexToQuery[dv].insert(qu);
            }
        }
    }
    
}


// void updateHostGraph(cpuGraph &graph, std::vector<change> &batch)
// {

// 	int sz = batch.size();
// 	for (int i = 0; i < sz; ++i)
// 	{
// 		change c = batch[i];
// 		std::swap(c.src, c.dst);
// 		batch.push_back(c);
// 	}

// 	std::sort(batch.begin(), batch.end(),
// 						[](change &a, change &b)
// 						{
// 		if (a.src != b.src)
// 			return a.src < b.src;
// 		return a.dst < b.dst; });

// 	std::vector<vtype> new_nbrs;
// 	new_nbrs.assign(graph.neighbors_, graph.neighbors_ + graph.num_e * 2);

// 	bool a_flip = true;
// 	bool d_flip = true;
// 	for (int i = 0; i < batch.size(); ++i)
// 	{
// 		vtype &src = batch[i].src;
// 		vtype &dst = batch[i].dst;
// 		if (batch[i].add)
// 		{
// 			int start = graph.offsets_[src];
// 			int end = graph.offsets_[src + 1];
// 			auto it = std::lower_bound(new_nbrs.begin() + start, new_nbrs.begin() + end, dst);

// 			if (it == new_nbrs.begin() + end || *it != dst)
// 			{
// 				int pos = it - new_nbrs.begin();
// 				new_nbrs.insert(it, dst);
// 				if (a_flip)
// 				{
// 					graph.num_e++;
// 					a_flip = false;
// 				}
// 				else
// 				{
// 					a_flip = true;
// 				}
// 				graph.outdeg_[src]++;
// 				for (int j = src + 1; j <= graph.num_v; ++j)
// 					graph.offsets_[j]++;
// 			}
// 		}
// 		else
// 		{
// 			int start = graph.offsets_[src];
// 			int end = graph.offsets_[src + 1];
// 			auto it = std::lower_bound(new_nbrs.begin() + start, new_nbrs.begin() + end, dst);

// 			if (it != new_nbrs.begin() + end && *it == dst)
// 			{
// 				int pos = it - new_nbrs.begin();
// 				new_nbrs.erase(it);
// 				if (d_flip)
// 				{
// 					graph.num_e--;
// 					d_flip = false;
// 				}
// 				else
// 				{
// 					d_flip = true;
// 				}
// 				graph.outdeg_[src]--;
// 				for (int j = src + 1; j <= graph.num_v; ++j)
// 					graph.offsets_[j]--;
// 			}
// 		}
// 	}

// 	for (int i = 0; i < graph.num_v; ++i)
// 		graph.maxDegree = std::max(graph.maxDegree, graph.outdeg_[i]);

// 	delete[] graph.neighbors_;
// 	graph.neighbors_ = new vtype[graph.num_e * 2];
// 	memcpy(graph.neighbors_, new_nbrs.data(), sizeof(vtype) * graph.num_e * 2);
// }
void printAdjacency(const std::unordered_map<vtype, std::unordered_set<vtype>>& adjacency) {
    std::cout << "==== Adjacency ====" << std::endl;
    for (const auto& [u_d, neighbors] : adjacency) {
        std::cout << "Data vertex " << u_d << " neighbors: [";
        for (const auto& v_d : neighbors) {
            std::cout << v_d << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "===================" << std::endl;
}

void printCandidates(const std::vector<std::vector<vtype>>& candidates) {
    std::cout << "==== Candidates ====" << std::endl;
    for (size_t u_q = 0; u_q < candidates.size(); ++u_q) {
        std::cout << "Query vertex " << u_q << " candidates: [";
        for (const auto& v_d : candidates[u_q]) {
            std::cout << v_d << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "====================" << std::endl;
}
void updateHostGraph(cpuGraph &graph, std::vector<change> &batch)
{
	// 复制原始 batch 以保留原始请求
	std::vector<change> temp_batch = batch;
	temp_batch.reserve(batch.size() * 2);

	// 生成反向边
	for (const auto c : batch)
	{
		change rev_c = c;
		std::swap(rev_c.src, rev_c.dst);
		temp_batch.push_back(rev_c);
	}

	// 按 src 和 dst 排序 保证同一个顶点的操作在一起
	std::sort(temp_batch.begin(), temp_batch.end(),
						[](const change &a, const change &b)
						{
							if (a.src != b.src)
								return a.src < b.src;
							return a.dst < b.dst;
						});

	// 初始化临时邻接表
	std::vector<std::set<int>> temp_adj(graph.num_v);
	for (int i = 0; i < graph.num_v; ++i)
	{
		int start = graph.offsets_[i];
		int end = graph.offsets_[i + 1];

		temp_adj[i].insert(graph.neighbors_ + start, graph.neighbors_ + end);

		// for (int j = start; j < end; ++j)
		// {
		// 	temp_adj[i].insert(graph.neighbors_[j]);
		// }
	}

	// 处理每条边
	for (const auto &c : temp_batch)
	{
		if (c.add)
		{
			graph.vLabels_[c.src] = c.labelChange;
			temp_adj[c.src].insert(c.dst);
			// temp_adj[c.dst].insert(c.src);
		}
		else
		{
			temp_adj[c.src].erase(c.dst);
			// temp_adj[c.dst].erase(c.src);
		}
	}

	// 生成新的 neighbors 和 offsets
	std::vector<int> new_neighbors;
	std::vector<int> new_offsets(graph.num_v + 1, 0);
	int current_offset = 0;

	for (int i = 0; i < graph.num_v; ++i)
	{
		new_offsets[i] = current_offset;
		for (auto neighbor : temp_adj[i])
		{
			new_neighbors.push_back(neighbor);
		}
		current_offset += temp_adj[i].size();
	}
	new_offsets[graph.num_v] = current_offset;

	// 更新图的属性
	graph.num_e = new_neighbors.size() / 2; // 假设图是无向的
	graph.neighbors_ = new vtype[new_neighbors.size()];
	memcpy(graph.neighbors_, new_neighbors.data(), sizeof(uint32_t) * new_neighbors.size());
	
	// delete[] graph.offsets_;
	// graph.offsets_ = new vtype[graph.num_v + 1];
	memcpy(graph.offsets_, new_offsets.data(), sizeof(int) * (graph.num_v + 1));

	// graph.outdeg_.resize(graph.num_v);
	graph.maxDegree = 0;
	for (int i = 0; i < graph.num_v; ++i)
	{
		graph.outdeg_[i] = temp_adj[i].size();
		graph.maxDegree = std::max(graph.maxDegree, graph.outdeg_[i]);
	}
}