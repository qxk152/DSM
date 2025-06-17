#include "defs.h"
#include "gpu_match.cuh"
#include "cuda_helpers.h"
#include "order.h"
#include "join.cuh"
#include "memManag.cuh"
#include "res_table.hpp"

#include <cooperative_groups.h>

#include <cstdio>
#include <ctime>
#include <chrono>

__global__ void
warmup()
{
  int a = IDX * 1;
}

__device__ void
intersect(Arg_t *__restrict__ arg)
{
  __shared__ offtype off[WARP_PER_BLOCK];
  __shared__ offtype off_ed[WARP_PER_BLOCK];
  __shared__ bool found[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ vtype v[WARP_PER_BLOCK][WARP_SIZE];

  __shared__ int cnt[WARP_PER_BLOCK];

  if (LID == 0)
  {
    cnt[WID] = 0;
    off[WID] = 0;
    off_ed[WID] = arg->set1_size;
  }
  __syncwarp();
  // vtype v;
  // bool found = false;

  while (off[WID] < off_ed[WID])
  {
    // off[WID] + lid = off[WID] + LID;
    v[WID][LID] = UINT32_MAX;
    if (off[WID] + LID < off_ed[WID])
      v[WID][LID] = arg->set1[off[WID] + LID];
    // else
    // v[WID][LID] = UINT32_MAX;
    __syncwarp();
    found[WID][LID] = false;
    if (v[WID][LID] != UINT32_MAX)
    {
      int res = lower_bound(arg->set2, arg->set2_size, v[WID][LID]);
      if (res != UINT32_MAX && arg->set2[res] == v[WID][LID])
        found[WID][LID] = true;
      // #pragma unroll 8
      //       for (int i = 0; i < arg->set2_size; ++i)
      //       {
      //         // found[WID][LID] |= (arg->set2[i] == v[WID][LID]);
      //         if (arg->set2[i] == v[WID][LID])
      //         {
      //           found[WID][LID] = true;
      //           break;
      //         }
      //         else if (arg->set2[i] > v[WID][LID])
      //         {
      //           break;
      //         }
      //       }
    }
    __syncwarp();
    if (found[WID][LID])
    {
      int mask = __activemask();
      int size = __popc(mask);
      int rank = __popc(mask & (FULL_MASK >> (31 - LID))) - 1;
      int pos = cnt[WID];
      // if (pos + rank > C_NUM_CAN_UB)
      // printf("pos + rank = %d\n", pos + rank);
      arg->res[pos + rank] = v[WID][LID];

      if (rank == 0)
        cnt[WID] += size;
    }
    __syncwarp();
    if (LID == 0)
      off[WID] += warpSize;
    __syncwarp();
  }
  __syncwarp();
  if (LID == 0)
    arg->res_size[0] = cnt[WID];
  __syncwarp();
}

__device__ void
get_new_v(
    degtype *__restrict__ d_degs_, offtype *__restrict__ d_offsets_, vtype *__restrict__ d_nbrs_,
    CallStack *__restrict__ stk, int *__restrict__ cur,
    StealingArgs *__restrict__ stealing_args, long long &start_clk,
    OrderGPU *__restrict__ order_obj,

    int start_level,

    uint32_t *__restrict__ compact_encodings_,
    int num_blocks,

    vtype *__restrict__ initial_task_table_, numtype num_initial_task_table_rows,

    bool *__restrict__ ret, vtype *__restrict__ cur_res, int *order_id)
{
  if (stk->level == start_level)
  {
    __shared__ int next_pos[WARP_PER_BLOCK];
    // __shared__ vtype u[WARP_PER_BLOCK][WARP_SIZE];
    // int next_pos;
    if (LID == 0)
      next_pos[WID] = atomicAdd(cur, 1);
    __syncwarp();
    // next_pos = __shfl_sync(FULL_MASK, next_pos, 0);

    if (next_pos[WID] >= num_initial_task_table_rows)
    {
      if (LID == 0)
        *ret = false;
      __syncwarp();
      return;
    }

    {
      if (LID == 0)
      {
        *order_id = initial_task_table_[next_pos[WID] * (start_level + 1) + 0];
      }
      // int l = LID; // one lane -- one level
      else if (LID <= start_level + 1)
      { //对于其他 lane（1 到 start_level+1），每个线程读取对应位置的候选顶点，并将结果存入局部当前结果数组 cur_res，同时将调用栈中对应层的迭代器初始化为 0，并设置候选数为 1。
        // u[WID][LID] = order_obj->v_orders_[order_id * C_NUM_VQ + l];
        // u[WID][LID] = v_order[l];
        stk->iter[LID] = 0;
        cur_res[LID - 1] = initial_task_table_[next_pos[WID] * (start_level + 1) + LID];
        // cur_res[u[WID][LID]] = initial_task_table_[next_pos[WID] * C_NUM_VQ + u[WID][LID]];
        // stk->candidates_[l * C_NUM_CAN_UB + 0] = initial_task_table_[next_pos * C_NUM_VQ + u];
        stk->num_candidates_[LID] = 1;
      }
    }
    __syncwarp();
  }
  else // level < C_NUM_VQ - 1
  {
    __shared__ vtype u[WARP_PER_BLOCK];
    __shared__ int cnt[WARP_PER_BLOCK];
    __shared__ Arg_t arg[WARP_PER_BLOCK];
    __shared__ vtype mapped_vs_[WARP_PER_BLOCK][WARP_SIZE];

    __shared__ int min_i[WARP_PER_BLOCK];
    __shared__ int min_nbrs[WARP_PER_BLOCK];
    __shared__ bool flag[WARP_PER_BLOCK][MAX_VQ];

    if (LID < C_NUM_VQ)
      flag[WID][LID] = false;
    __syncwarp();

    if (LID == 0)
      u[WID] = order_obj->v_orders_[(*order_id) * C_NUM_VQ + stk->level];
    // u[WID] = v_order[stk->level];
    __syncwarp();
    // vtype u = v_order[stk->level];

    // if (LID == 0)
    //   *cnt = 0;
    // __syncwarp();

    // if (LID == 0)
    // printf("num_bn: %d\n", order_obj->num_backward_neighbors_[u]);
    // __syncwarp();

    if (LID < order_obj->num_backward_neighbors_[(*order_id) * C_NUM_VQ + u[WID]])
    {
      vtype u_back = order_obj->backward_neighbors_[(*order_id) * C_NUM_VQ * C_NUM_VQ + u[WID] * C_NUM_VQ + LID];
      // int l_back = order_obj->u2l_[u_back];
      // vtype v_back = cur_res[u_back];
      // vtype v_back = stk->candidates_[l_back * C_NUM_CAN_UB + stk->iter[l_back]];
      // mapped_vs_[WID][LID] = cur_res[u_back];
      mapped_vs_[WID][LID] = cur_res[order_obj->u2ls_[(*order_id) * C_NUM_VQ + u_back]];
    }
    __syncwarp();

    if (LID == 0)
    {
      if (order_obj->num_backward_neighbors_[(*order_id) * C_NUM_VQ + u[WID]] == 1)
      {
        min_i[WID] = 0;
      }
      else
      {
        min_i[WID] = 0;
        min_nbrs[WID] = d_degs_[mapped_vs_[WID][0]];
        for (int i = 1; i < order_obj->num_backward_neighbors_[(*order_id) * C_NUM_VQ + u[WID]]; ++i)
        {
          if (d_degs_[mapped_vs_[WID][i]] < min_nbrs[WID])
          {
            min_i[WID] = i;
            min_nbrs[WID] = d_degs_[mapped_vs_[WID][i]];
          }
        }
      }
    }
    __syncwarp();

    if (LID == 0)
    {
      // printf("mapped_vs_[0] = %d\n", mapped_vs_[0]);
      arg[WID].res = d_nbrs_ + d_offsets_[mapped_vs_[WID][min_i[WID]]];
      arg[WID].res_size = stk->num_candidates_ + stk->level;
      arg[WID].res_size[0] = d_degs_[mapped_vs_[WID][min_i[WID]]];
      flag[WID][min_i[WID]] = true;
    }
    __syncwarp();
    for (int i = 1; i < order_obj->num_backward_neighbors_[(*order_id) * C_NUM_VQ + u[WID]]; ++i)
    {
      if (LID == 0)
      {
        min_nbrs[WID] = UINT32_MAX;
        for (int j = 0; j < order_obj->num_backward_neighbors_[(*order_id) * C_NUM_VQ + u[WID]]; ++j)
        {
          if (flag[WID][j])
            continue;
          if (d_degs_[mapped_vs_[WID][j]] < min_nbrs[WID])
          {
            min_i[WID] = j;
            min_nbrs[WID] = d_degs_[mapped_vs_[WID][j]];
          }
        }

        arg[WID].set1 = arg[WID].res;
        arg[WID].set1_size = arg[WID].res_size[0];
        arg[WID].set2 = d_nbrs_ + d_offsets_[mapped_vs_[WID][min_i[WID]]];
        arg[WID].set2_size = d_degs_[mapped_vs_[WID][min_i[WID]]];
        // arg[WID].res = can + (stk->level - start_level - 1) * C_NUM_CAN_UB;
        arg[WID].res = stk->candidates_ + stk->level * C_NUM_CAN_UB;
        arg[WID].res_size = stk->num_candidates_ + stk->level;
        flag[WID][min_i[WID]] = true;
      }
      __syncwarp();
      intersect(&arg[WID]);
    }

    if (LID == 0)
      cnt[WID] = 0;
    __syncwarp();

    __shared__ offtype off[WARP_PER_BLOCK];
    __shared__ offtype off_ed[WARP_PER_BLOCK];
    __shared__ bool dup[WARP_PER_BLOCK][WARP_SIZE];
    __shared__ vtype v[WARP_PER_BLOCK][WARP_SIZE];

    if (LID == 0)
    {
      off[WID] = 0;
      off_ed[WID] = arg[WID].res_size[0];
    }
    __syncwarp();
    // vtype v;
    // bool dup = false;
    while (off[WID] < off_ed[WID])
    {
      int my_off = off[WID] + LID;
      if (my_off < off_ed[WID])
        v[WID][LID] = arg[WID].res[my_off];
      else
        v[WID][LID] = UINT32_MAX;
      __syncwarp();
      dup[WID][LID] = false;
      if (v[WID][LID] != UINT32_MAX)
        for (int l = 0; l < stk->level; ++l)
        {
          if (v[WID][LID] == cur_res[l])
          // if (v[WID][LID] == cur_res[v_order[l]])
          {
            dup[WID][LID] = true;
            break;
          }
          // if (v[WID][LID] == stk->candidates_[l * C_NUM_CAN_UB + stk->iter[l]])
          // {
          //   dup = true;
          //   break;
          // }
        }
      __syncwarp();
      if (v[WID][LID] != UINT32_MAX &&
          !dup[WID][LID] &&
          (compact_encodings_[u[WID] * C_COL_LEN + v[WID][LID] / BLK_SIZE] & (1 << (v[WID][LID] % BLK_SIZE))))
      {
        int mask = __activemask();
        int size = __popc(mask);
        int rank = __popc(mask & (FULL_MASK >> (31 - LID))) - 1;
        int my_pos = cnt[WID] + rank;
        // can[(stk->level - start_level - 1) * C_NUM_CAN_UB + my_pos] = v[WID][LID];
        stk->candidates_[stk->level * C_NUM_CAN_UB + my_pos] = v[WID][LID];
        if (rank == 0)
          cnt[WID] += size;
      }
      __syncwarp();
      if (LID == 0)
        off[WID] += warpSize;
      __syncwarp();
    }
    __syncwarp();
    if (LID == 0)
    {
      if (cnt[WID] == 0)
        *ret = false;
      else
      {
        stk->iter[stk->level] = 0;
        stk->num_candidates_[stk->level] = cnt[WID];
        // cur_res[u[WID]] = can[(stk->level - start_level - 1) * C_NUM_CAN_UB + 0];
        // cur_res[u[WID]] = stk->candidates_[stk->level * C_NUM_CAN_UB + 0];
        cur_res[stk->level] = stk->candidates_[stk->level * C_NUM_CAN_UB + 0];
      }
    }
    __syncwarp();
  }
}

__device__ void
steal(
    CallStack *__restrict__ stk, StealingArgs *__restrict__ stealing_args, bool *__restrict__ __restrict__ ret, long long &start_clk, vtype *v_order,
    int *queue_arr_,
    vtype *__restrict__ cur_res, int *order_id)
{
  __shared__ bool flag[WARP_PER_BLOCK];

  if (LID == 0)
  {
    stk->stealed_task = false;
    flag[WID] = stealing_args->queue->dequeue(queue_arr_, C_STOP_LEVEL);
  }
  __syncwarp();

  if (flag[WID])
  {
    if (LID < C_STOP_LEVEL && LID > 0)
    {
      if (queue_arr_[LID] != DeletionMarker<int>::val - 1)
      {
        stk->iter[LID] = 0;
        stk->num_candidates_[LID] = 1;
        // stk->candidates_[LID * C_NUM_CAN_UB + 0] = queue_arr_[LID];
        // cur_res[v_order[LID]] = queue_arr_[LID];
        cur_res[LID] = queue_arr_[LID];
        atomicMax(&stk->level, LID);
      }
    }
    else if (LID == 0)
    {
      *order_id = queue_arr_[0];
    }
    __syncwarp();
    if (LID == 0)
    {
      stk->iter[stk->level + 1] = 0;
      stk->num_candidates_[stk->level + 1] = 0;
      stk->stealed_task = true;
    }
    __syncwarp();
  }
  else
  {
    if (LID == 0)
      *ret = false;
    __syncwarp();
  }
  if (LID == 0)
    start_clk = clock64();
  __syncwarp();
  // start_clk = __shfl_sync(FULL_MASK, start_clk, 0);

  // if (LID == 0)
  // {
  //   stk->stealed_task = false;
  //   bool flag = stealing_args->queue->dequeue(queue_arr_, C_STOP_LEVEL);
  //   if (flag)
  //   {
  //     for (int i = 0; i < C_STOP_LEVEL; ++i)
  //     {
  //       int val = queue_arr_[i];
  //       if (val != DeletionMarker<int>::val - 1)
  //       {
  //         stk->iter[i] = 0;
  //         stk->num_candidates_[i] = 1;
  //         // stk->candidates_[i * C_NUM_CAN_UB + 0] = val;
  //         cur_res[v_order[i]] = val;
  //         stk->level = i;
  //       }
  //       else
  //       {
  //         stk->iter[i] = 0;
  //         stk->num_candidates_[i] = 0;
  //         break;
  //       }
  //     }
  //     stk->stealed_task = true;
  //   }
  //   else
  //   {
  //     *ret = false;
  //   }
  // }
  // __syncwarp();
  // if (LID == 0)
  //   start_clk = clock64();
  // start_clk = __shfl_sync(FULL_MASK, start_clk, 0);
}

__device__ void
match(
    degtype *__restrict__ d_degs_, offtype *__restrict__ d_offsets_, vtype *__restrict__ d_nbrs_,
    CallStack *__restrict__ stk, int *__restrict__ cur, unsigned long long *__restrict__ count,
    StealingArgs *__restrict__ stealing_args, long long &start_clk,
    OrderGPU *__restrict__ order_obj, int start_level, int *__restrict__ queue_arr_,
    uint32_t *__restrict__ compact_encodings_, int num_blocks,
    vtype *__restrict__ initial_task_table_, numtype num_initial_task_table_rows,

    bool *ret, vtype *cur_res)
{ // initial_task_table_ = d_u_candidate_vs_
  // uint32_t &level = stk->level;
  // can = candidates[wid]

  __shared__ int s_order_id[WARP_PER_BLOCK];

  while (true)
  { //首先检查调用栈当前层是否处于起始层
    if (stk->level == start_level)
    {
      if (LID == 0)
        *ret = true;
      __syncwarp();

      // steal(stk, stealing_args, ret, start_clk, v_order, queue_arr_, cur_res);
      steal(stk, stealing_args, ret, start_clk, order_obj->v_orders_, queue_arr_, cur_res, &s_order_id[WID]);
      if (*ret == false) // nothing to steal
      {
        if (LID == 0)
          *ret = true;
        __syncwarp();
        get_new_v(
            d_degs_, d_offsets_, d_nbrs_,
            stk, cur, stealing_args, start_clk,
            order_obj, start_level,
            compact_encodings_, num_blocks,
            initial_task_table_, num_initial_task_table_rows,
            ret, cur_res, &s_order_id[WID]);
        __syncwarp();
      }
      else
      {
        if (LID == 0)
        {
          *ret = true;
        }
        __syncwarp();
      }
      if (*ret == false) // no more candidates
      {
        break;
      }
      else
      {
        if (LID == 0)
        {
          stk->level++;
          stk->iter[stk->level] = 0;
          stk->num_candidates_[stk->level] = 0;
        }
        __syncwarp();
      }
    }
    else if (stk->level == C_NUM_VQ - 1) // final, intersect and return
    {
      __shared__ Arg_t arg[WARP_PER_BLOCK];
      __shared__ vtype mapped_vs_[WARP_PER_BLOCK][WARP_SIZE];
      __shared__ int cnt[WARP_PER_BLOCK];

      __shared__ int min_i[WARP_PER_BLOCK];
      __shared__ int min_nbrs[WARP_PER_BLOCK];
      __shared__ bool flag[WARP_PER_BLOCK][MAX_VQ];

      if (LID < C_NUM_VQ)
        flag[WID][LID] = false;
      __syncwarp();

      vtype u = order_obj->v_orders_[s_order_id[WID] * C_NUM_VQ + stk->level];
      // vtype u = v_order[stk->level];

      if (LID < order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + u])
      {
        vtype u_back = order_obj->backward_neighbors_[s_order_id[WID] * C_NUM_VQ * C_NUM_VQ + u * C_NUM_VQ + LID];
        // vtype u_back = order_obj->backward_neighbors_[u * C_NUM_VQ + LID];
        // int l_back = order_obj->u2l_[u_back];
        // vtype v_back = cur_res[u_back];
        // vtype v_back = stk->candidates_[l_back * C_NUM_CAN_UB + stk->iter[l_back]];
        // mapped_vs_[WID][LID] = cur_res[u_back];
        mapped_vs_[WID][LID] = cur_res[order_obj->u2ls_[s_order_id[WID] * C_NUM_VQ + u_back]];
      }
      __syncwarp();

      if (LID == 0)
      {
        if (order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + u] == 1)
        {
          min_i[WID] = 0;
        }
        else
        {
          min_i[WID] = 0;
          min_nbrs[WID] = d_degs_[mapped_vs_[WID][0]];
          for (int i = 1; i < order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + u]; ++i)
          {
            if (d_degs_[mapped_vs_[WID][i]] < min_nbrs[WID])
            {
              min_i[WID] = i;
              min_nbrs[WID] = d_degs_[mapped_vs_[WID][i]];
            }
          }
        }
      }
      __syncwarp();

      if (LID == 0)
      {
        arg[WID].res = d_nbrs_ + d_offsets_[mapped_vs_[WID][min_i[WID]]];
        arg[WID].res_size = stk->num_candidates_ + stk->level;
        arg[WID].res_size[0] = d_degs_[mapped_vs_[WID][min_i[WID]]];
        flag[WID][min_i[WID]] = true;
      }
      __syncwarp();
      for (int i = 1; i < order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + u]; ++i)
      {
        if (LID == 0)
        {
          min_nbrs[WID] = UINT32_MAX;
          for (int j = 0; j < order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + u]; ++j)
          {
            if (flag[WID][j])
              continue;
            if (d_degs_[mapped_vs_[WID][j]] < min_nbrs[WID])
            {
              min_i[WID] = j;
              min_nbrs[WID] = d_degs_[mapped_vs_[WID][j]];
            }
          }

          arg[WID].set1 = arg[WID].res;
          arg[WID].set1_size = arg[WID].res_size[0];
          arg[WID].set2 = d_nbrs_ + d_offsets_[mapped_vs_[WID][min_i[WID]]];
          arg[WID].set2_size = d_degs_[mapped_vs_[WID][min_i[WID]]];
          // arg[WID].res = can_wid + (stk->level - start_level - 1) * C_NUM_CAN_UB;
          arg[WID].res = stk->candidates_ + stk->level * C_NUM_CAN_UB;
          arg[WID].res_size = stk->num_candidates_ + stk->level;
          flag[WID][min_i[WID]] = true;
        }
        __syncwarp();
        intersect(&arg[WID]);
      }

      if (LID == 0)
        cnt[WID] = 0;
      __syncwarp();
      // int pos = d_enc_pos_u_[u];

      __shared__ offtype off_ed[WARP_PER_BLOCK];
      __shared__ offtype off[WARP_PER_BLOCK];
      __shared__ bool dup[WARP_PER_BLOCK][WARP_SIZE];
      __shared__ vtype v[WARP_PER_BLOCK][WARP_SIZE];

      off[WID] = 0;
      off_ed[WID] = arg[WID].res_size[0];
      // vtype v;
      dup[WID][LID] = false;
      while (off[WID] < off_ed[WID])
      {
        if (off[WID] + LID < off_ed[WID])
          v[WID][LID] = arg[WID].res[off[WID] + LID];
        else
          v[WID][LID] = UINT32_MAX;
        __syncwarp();
        dup[WID][LID] = false;
        if (v[WID][LID] != UINT32_MAX)
          for (int l = 0; l < stk->level; ++l)
          {
            if (v[WID][LID] == cur_res[l])
            {
              dup[WID][LID] = true;
              break;
            }

            // dup[WID][LID] |= (v[WID][LID] == cur_res[v_order[l]]);
            // if (dup[WID][LID])
            //   break;
            // if (v[WID][LID] == cur_res[v_order[l]])
            // dup[WID][LID] = true;
            // if (v == stk->candidates_[l * C_NUM_CAN_UB + stk->iter[l]])
            // dup[WID][LID] = true;
          }
        __syncwarp();
        if (v[WID][LID] != UINT32_MAX &&
            !dup[WID][LID] &&
            (compact_encodings_[u * C_COL_LEN + v[WID][LID] / BLK_SIZE] & (1 << (v[WID][LID] % BLK_SIZE))))
        // if (v != UINT32_MAX && !dup[WID][LID] && (compact_encodings_[v * num_blocks + pos / BLK_SIZE] & (1 << (pos % BLK_SIZE))))
        {
          int mask = __activemask();
          int size = __popc(mask);
          int rank = __popc(mask & (FULL_MASK >> (31 - LID))) - 1;
          int my_pos = cnt[WID] + rank;
          // can_wid[(stk->level - start_level - 1) * C_NUM_CAN_UB + my_pos] = v[WID][LID];
          stk->candidates_[stk->level * C_NUM_CAN_UB + my_pos] = v[WID][LID];
          if (rank == 0)
            cnt[WID] += size;
        }
        __syncwarp();
        if (LID == 0)
          off[WID] += warpSize;
        __syncwarp();
      }
      if (LID == 0)
      {
        *count += cnt[WID];
        stk->level--;
        stk->iter[stk->level]++;
        // cur_res[v_order[stk->level]] = can_wid[(stk->level - start_level - 1) * C_NUM_CAN_UB + stk->iter[stk->level]];
        // cur_res[order_obj->v_orders_[order_id * C_NUM_VQ + stk->level]] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
        cur_res[stk->level] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
        // cur_res[v_order[stk->level]] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
      }
      __syncwarp();
    }
    else // intermediate levels
    {
      __shared__ int is_timeout[WARP_PER_BLOCK];
      // int is_timeout;
      if (LID == 0)
        is_timeout[WID] = stk->level == C_STOP_LEVEL - 1 && ELAPSED_TIME(start_clk) > TIMEOUT && !stk->stealed_task;
      __syncwarp();
      // is_timeout = __shfl_sync(FULL_MASK, is_timeout, 0);

      if (stk->num_candidates_[stk->level] == 0) // top-down, get new candidates and keep going down. Or return.
      {
        if (LID == 0)
          *ret = true;
        __syncwarp();
        get_new_v(
            d_degs_, d_offsets_, d_nbrs_,
            stk, cur, stealing_args, start_clk,
            order_obj, start_level,
            compact_encodings_, num_blocks,
            initial_task_table_, num_initial_task_table_rows,
            ret, cur_res, &s_order_id[WID]);
        if (*ret == false) // no candidates, return
        {
          if (LID == 0)
          {
            stk->iter[stk->level] = 0;
            stk->level--;
            stk->iter[stk->level]++;
            // cur_res[v_order[stk->level]] = can_wid[(stk->level - start_level - 1) * C_NUM_CAN_UB + stk->iter[stk->level]];
            // cur_res[order_obj->v_orders_[stk->level]] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
            cur_res[stk->level] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
          }
          __syncwarp();
        }
        else // candidates found, go down.
        {
          if (LID == 0)
          {
            stk->iter[stk->level] = 0;
            stk->level++;
            stk->iter[stk->level] = 0;
            stk->num_candidates_[stk->level] = 0;
          }
          __syncwarp();
        }
      }
      else if (stk->iter[stk->level] == stk->num_candidates_[stk->level]) // end of this level, return.
      {
        if (LID == 0)
        {
          stk->iter[stk->level] = 0;
          stk->num_candidates_[stk->level] = 0;

          stk->level--;
          stk->iter[stk->level]++;
          // cur_res[v_order[stk->level]] = can_wid[(stk->level - start_level - 1) * C_NUM_CAN_UB + stk->iter[stk->level]];
          // cur_res[v_order[stk->level]] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
          // cur_res[order_obj->v_orders_[order_id * C_NUM_VQ + stk->level]] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
          cur_res[stk->level] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
        }
        __syncwarp();
        if (stk->level == start_level)
        {
          if (LID == 0)
            start_clk = clock64();
          __syncwarp();
          // start_clk = __shfl_sync(FULL_MASK, start_clk, 0);
        }
      }
      else // not end, map next v, then keep going down.
      {
        if (!is_timeout[WID])
        {
          if (LID == 0)
          {
            stk->level++;
            stk->iter[stk->level] = 0;
            stk->num_candidates_[stk->level] = 0;
          }
          __syncwarp();
        }
        else // timeout, split task into queue.
        {
          __shared__ bool enqueue_succ[WARP_PER_BLOCK];
          // int enqueue_succ = false;
          if (LID == 0)
            queue_arr_[0] = s_order_id[WID];
          else if (LID < C_STOP_LEVEL)
          {
            // queue_arr_[LID] = cur_res[v_order[LID]];
            queue_arr_[LID] = cur_res[LID - 1];
          }
          __syncwarp();
          if (LID == 0)
          {
            enqueue_succ[WID] = false;
            // for (int i = 0; i < C_STOP_LEVEL - 1; ++i)
            // {
            //   queue_arr_[i] = cur_res[v_order[i]];
            //   // queue_arr_[i] = stk->candidates_[i * C_NUM_CAN_UB + 0];

            //   // queue_arr_[i] = cur_res[v_order[i]];
            //   // if (stk->num_candidates_[i])
            //   //   // queue_arr_[i] = cur_res[v_order[i]];
            //   //   queue_arr_[i] = stk->candidates_[i * C_NUM_CAN_UB + stk->iter[i]];
            //   // else
            //   //   queue_arr_[i] = DeletionMarker<int>::val - 1;
            // }
            queue_arr_[C_STOP_LEVEL] = DeletionMarker<int>::val - 1;
            // queue_arr_[C_STOP_LEVEL - 1] = DeletionMarker<int>::val - 1;
#pragma unroll 2
            for (; stk->iter[stk->level] < stk->num_candidates_[stk->level]; ++stk->iter[stk->level])
            {
              // queue_arr_[C_STOP_LEVEL - 1] = can_wid[(stk->level - start_level - 1) * C_NUM_CAN_UB + stk->iter[stk->level]];
              // queue_arr_[C_STOP_LEVEL - 1] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
              queue_arr_[C_STOP_LEVEL] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
              enqueue_succ[WID] = stealing_args->queue->enqueue(queue_arr_, C_STOP_LEVEL + 1);
              if (!enqueue_succ[WID])
                break;
            }
          }
          __syncwarp();
          // enqueue_succ = __shfl_sync(FULL_MASK, enqueue_succ, 0);
          if (enqueue_succ[WID])
          {
            stk->num_candidates_[stk->level] = 0;
            stk->iter[stk->level] = 0;
            if (stk->level > start_level)
            {
              if (LID == 0)
              {
                stk->level--;
                stk->iter[stk->level]++;
                // cur_res[v_order[stk->level]] = can_wid[(stk->level - start_level - 1) * C_NUM_CAN_UB + stk->iter[stk->level]];
                cur_res[stk->level] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
                // cur_res[order_obj->v_orders_[order_id * C_NUM_VQ + stk->level]] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
                // cur_res[v_order[stk->level]] = stk->candidates_[stk->level * C_NUM_CAN_UB + stk->iter[stk->level]];
              }
              __syncwarp();
            }
          }
          else
          {
            if (LID == 0)
              start_clk = clock64();
            __syncwarp();
            // start_clk = __shfl_sync(FULL_MASK, start_clk, 0);
          }
        }
      }
    }
  }
  __syncwarp();
}

__global__ void
parallel_match_kernel(
    degtype *__restrict__ d_degs_, offtype *__restrict__ d_offsets_, vtype *__restrict__ d_nbrs_,
    CallStack *__restrict__ call_stack, int *__restrict__ cur, unsigned long long *__restrict__ res,
    Queue *__restrict__ queue, OrderGPU *__restrict__ order_obj, int num_orders,

    int start_level,

    uint32_t *__restrict__ compact_encodings_,
    int num_blocks,
    vtype *__restrict__ d_u_candidate_vs_, numtype *__restrict__ d_num_u_candidate_vs_,

    vtype *__restrict__ initial_task_table_, numtype num_initial_task_table_rows)
{ // initial_task_table_ = d_u_candidate_vs_
  queue->init();

  // __shared__ OrderGPU s_order_obj(num_orders);
  __shared__ CallStack stk[WARP_PER_BLOCK];

  __shared__ unsigned long long s_count[WARP_PER_BLOCK]; // total count of results this warp
  __shared__ vtype cur_res[WARP_PER_BLOCK][MAX_VQ];      // current result, this warp
  // __shared__ vtype s_v_order[MAX_VQ];                    // shared across this block

  // for match()
  __shared__ bool ret[WARP_PER_BLOCK]; // return value of get_new_v(), this warp

  // for steal.queue
  __shared__ int queue_arr_[WARP_PER_BLOCK][MAX_VQ];
  __shared__ StealingArgs s_stealing_args;
  __shared__ long long start_clk[WARP_PER_BLOCK];

  s_stealing_args.queue = queue;

  // if (TID == 0)
  // {
  // s_order_obj = *order_obj;
  // }

  // if (TID < C_NUM_VQ)
  // {
  // s_v_order[TID] = s_order_obj.v_order_[TID];
  // }
  __syncthreads();

  if (LID == 0)
  {
    stk[WID].candidates_ = call_stack[WID_G].candidates_;
    // stk[WID] = call_stack[WID_G];
    stk[WID].level = start_level;
    stk[WID].num_candidates_[start_level] = 0;
    s_count[WID] = 0;
  }
  __syncwarp();

  // long long st = clock64();
  if (LID == 0)
    start_clk[WID] = clock64();
  __syncwarp();
  match(
      d_degs_, d_offsets_, d_nbrs_,
      &stk[WID], cur,
      &s_count[WID], &s_stealing_args,
      start_clk[WID], order_obj,

      start_level, queue_arr_[WID],

      compact_encodings_,
      num_blocks,

      initial_task_table_, num_initial_task_table_rows,

      &ret[WID], cur_res[WID]);
  __syncwarp();
  // long long ed = clock64();

  if (LID == 31)
  {
    res[WID_G] = s_count[WID];
    // printf("bid: %d, WID: %d, time: %lf(ms)\n", BID, WID_G, (ed - start_clk) / (1.0 * CLOCK_RATE));
    // printf("WID: %d, count: %lu\n", WID_G, s_count[WID]);
  }
  __syncwarp();
}
//d_res_table_old 存储{0, 5, 6, 1, 5, 6, 2, 5, 6, 3, 5, 6} num_res_old存储 结果为4
__global__ void
vertexJoinBFS(
    offtype *d_offsets_, vtype *d_neighbors_, degtype *d_degree_,
    OrderGPU *order_obj,  
    int level, // size of one row = level + 1
    vtype *intersect_temp_storage, numtype *num_intersect_temp_storage,

    uint32_t *d_encodings_, numtype num_blocks, int *d_enc_pos_u_,
    vtype *d_res_table_old, numtype num_res_old,
    vtype *d_res_table, numtype *num_res_new,
    int *exceed)
{
  // int tid = threadIdx.x;
  // int bid = blockIdx.x;
  // int idx = tid + bid * blockDim.x;
  // int wid = tid >> 5;
  // int lid = tid & 31;
  // int wid_g = idx >> 5;

  __shared__ int s_row[WARP_PER_BLOCK];
  __shared__ vtype s_cur_res[WARP_PER_BLOCK][MAX_VQ];
  __shared__ vtype mapped_vs_[WARP_PER_BLOCK][MAX_VQ];
  __shared__ Arg_t arg[WARP_PER_BLOCK];
  __shared__ int warp_pos[WARP_PER_BLOCK];
  // __shared__ int s_v_order_[MAX_VQ];
  // __shared__ int s_num_bn[MAX_VQ];
  // __shared__ vtype s_bn[MAX_VQ][MAX_VQ];

  __shared__ int s_order_id[WARP_PER_BLOCK];
  __shared__ vtype s_u[WARP_PER_BLOCK];
  __shared__ int num_warps;
  __shared__ int block_iter_cnt[WARP_PER_BLOCK];
  // __shared__ int s_pos_u[MAX_VQ];
  // __shared__ int s_orders_[32][MAX_VQ];

  __shared__ int min_i[WARP_PER_BLOCK];
  __shared__ int min_nbrs[WARP_PER_BLOCK];
  __shared__ bool flag[WARP_PER_BLOCK][MAX_VQ];

  // if (LID < C_NUM_VQ)
  //   flag[WID][LID] = false;
  // __syncwarp();

  // int block_iter_cnt = 0;

  if (LID == 0) //LID 是线程在warp中的id WID是当前的warp的id  初始化当前warp在第几层
    block_iter_cnt[WID] = 0;
  __syncwarp();

  if (TID == 0)
    num_warps = blockDim.x / warpSize * gridDim.x;
  __syncthreads();
  
  
  if (TID < C_NUM_VQ)
  {
    // s_pos_u[TID] = d_enc_pos_u_[TID];
    // s_num_bn[TID] = order_obj->num_backward_neighbors_[TID];
    // s_v_order_[TID] = order_obj->v_orders_[TID];
  }
  __syncthreads();
  // if (TID < C_NUM_VQ * C_NUM_VQ)
  //   s_bn[TID / C_NUM_VQ][TID % C_NUM_VQ] = order_obj->backward_neighbors_[TID];
  // __syncthreads();

  // vtype u = s_v_order_[level];
  // 所有 warp 以一个步长为 num_warps 的方式分摊所有匹配结果
  //block_iter_cnt 记录warp迭代了几次
  
  while (WID_G + block_iter_cnt[WID] * num_warps < num_res_old)
  {
    if (LID < C_NUM_VQ)
      flag[WID][LID] = false;
    __syncwarp();
    if (LID == 0)
    {
      s_row[WID] = WID_G + block_iter_cnt[WID] * num_warps; // 当前warp处理哪个orderID
      s_order_id[WID] = d_res_table_old[s_row[WID] * (level + 1)];
      s_u[WID] = order_obj->v_orders_[s_order_id[WID] * C_NUM_VQ + level]; // 这是要匹配的查询顶点
    }
    __syncwarp();
    if (LID < level)
      s_cur_res[WID][LID] = d_res_table_old[s_row[WID] * (level + 1) + 1 + LID];
    __syncwarp();

    // if (LID < s_num_bn[u])
    if (LID < order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + s_u[WID]])
    { //如果查询顶点有 2 个后向邻居，那么只有 LID 为 0 和 1 的线程参与处理。 数组中取出当前查询顶点对应的第 LID 个后向邻居。
      vtype u_back = order_obj->backward_neighbors_[s_order_id[WID] * C_NUM_VQ * C_NUM_VQ + s_u[WID] * C_NUM_VQ + LID];
      mapped_vs_[WID][LID] = s_cur_res[WID][order_obj->u2ls_[s_order_id[WID] * C_NUM_VQ + u_back]];
      // 假设 s_cur_res[WID] 存储的是 {5, 6}，那么如果LID= 0，就表示后向邻居 u_back 对应的是更新边的起始顶点 5；如果返回 1，则对应更新边的顶点 6。
      // mapped_vs_[WID][LID] = v_back;
    }
    __syncwarp();
    
    if (LID == 0)
    {
      min_i[WID] = 0;
      // if (order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + level] == 1)
      // {
      // min_i[WID] = 0;
      // }
      if (order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + s_u[WID]] > 1)
      {
        // min_i[WID] = 0;
        min_nbrs[WID] = d_degree_[mapped_vs_[WID][0]];
        for (int i = 1; i < order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + s_u[WID]]; ++i)
        {
          if (d_degree_[mapped_vs_[WID][i]] < min_nbrs[WID])
          {
            min_i[WID] = i;
            min_nbrs[WID] = d_degree_[mapped_vs_[WID][i]]; // 数据图顶点的度
          }
        }
      }
    } // 每一个warp中的第一个线程选择一个度最小的顶点
    __syncwarp();
   
    if (LID == 0)
    {
      // printf("mapped_vs_[0] = %d\n", mapped_vs_[0]); 就是表示当前 warp 中选择的那个后向邻居对应的顶点值
      arg[WID].res = d_neighbors_ + d_offsets_[mapped_vs_[WID][min_i[WID]]];  // 存储的是映射的最小度顶点的令居集合
      arg[WID].res_size = num_intersect_temp_storage + WID_G;
      arg[WID].res_size[0] = d_degree_[mapped_vs_[WID][min_i[WID]]];
      flag[WID][min_i[WID]] = true;
    }
    __syncwarp();
    for (int i = 1; i < order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + s_u[WID]]; ++i)
    {
      if (LID == 0)
      {
        min_nbrs[WID] = UINT32_MAX;
        for (int j = 0; j < order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + s_u[WID]]; ++j)
        {
          if (flag[WID][j])
            continue;
          if (d_degree_[mapped_vs_[WID][j]] < min_nbrs[WID])
          {
            min_i[WID] = j;
            min_nbrs[WID] = d_degree_[mapped_vs_[WID][j]];
          }
        }

        arg[WID].set1 = arg[WID].res;
        arg[WID].set1_size = arg[WID].res_size[0];
        arg[WID].set2 = d_neighbors_ + d_offsets_[mapped_vs_[WID][min_i[WID]]]; // 是两个最小度后向邻居的邻居候选集取交集
        arg[WID].set2_size = d_degree_[mapped_vs_[WID][min_i[WID]]];
        // arg[WID].res = can + (stk->level - start_level - 1) * C_NUM_CAN_UB;
        arg[WID].res = intersect_temp_storage + WID_G * C_NUM_CAN_UB;
        arg[WID].res_size = num_intersect_temp_storage + WID_G;
        flag[WID][min_i[WID]] = true;
      }
      __syncwarp();
      intersect(&arg[WID]);
    }
    // if (LID == 0)
    // {
    //   arg[WID].res = d_neighbors_ + d_offsets_[mapped_vs_[WID][0]];
    //   arg[WID].res_size = num_intersect_temp_storage + WID_G;
    //   arg[WID].res_size[0] = d_degree_[mapped_vs_[WID][0]];
    // }
    // __syncwarp();
    // for (int i = 1; i < order_obj->num_backward_neighbors_[s_order_id[WID] * C_NUM_VQ + level]; ++i)
    // // for (int i = 1; i < s_num_bn[u]; ++i)
    // {
    //   if (LID == 0)
    //   {
    //     arg[WID].set1 = arg[WID].res;
    //     arg[WID].set1_size = arg[WID].res_size[0];
    //     arg[WID].set2 = d_neighbors_ + d_offsets_[mapped_vs_[WID][i]];
    //     arg[WID].set2_size = d_degree_[mapped_vs_[WID][i]];
    //     arg[WID].res = intersect_temp_storage + WID_G * C_NUM_CAN_UB;
    //     arg[WID].res_size = num_intersect_temp_storage + WID_G;
    //   }
    //   __syncwarp();
    //   intersect(&arg[WID]);
    // }
    // int pos = s_pos_u[s_u[WID]];

    offtype off = 0;
    offtype off_ed = arg[WID].res_size[0];
    vtype v;
    bool dup = false;
    
    while (off < off_ed)
    {
      int my_off = off + LID;
      if (my_off < off_ed)
        v = arg[WID].res[my_off];
      else
        v = UINT32_MAX;
      __syncwarp();
      dup = false;
      if (v != UINT32_MAX)
        for (int l = 0; l < level; ++l)
        {
          // if (v == s_cur_res[WID][order_obj->v_orders_[s_order_id[WID] * C_NUM_VQ + l]])
          // if (v == s_cur_res[WID][s_v_order_[l]])
          if (v == s_cur_res[WID][l])
          {
            dup = true; //重复
            break;
          }
        }
      __syncwarp();
      if (v != UINT32_MAX &&
          !dup &&
          // (d_encodings_[v * num_blocks + s_pos_u[s_u[WID]] / BLK_SIZE] & (1 << (s_pos_u[s_u[WID]] % BLK_SIZE)))
          d_encodings_[s_u[WID] * C_COL_LEN + v / BLK_SIZE] & (1 << (v % BLK_SIZE)))
      {
        int mask = __activemask();// 获取当前活跃线程的掩码
        int size = __popc(mask);// 当前活跃线程数
        int rank = __popc(mask & (FULL_MASK >> (31 - LID))) - 1;// 计算当前线程在活跃线程中的排名
        if (rank == 0)
        { //原子分配当前warp的结果起始位置
          warp_pos[WID] = atomicAdd(num_res_new, size);
          if (warp_pos[WID] + size >= MAX_RES)
          {
            *exceed = 1;
          }
        }
        __syncwarp(mask);
        if (*exceed == 1)
          return;
        int my_pos = warp_pos[WID] + rank;
        // 需要前面的多一个所以要加2
        d_res_table[my_pos * (level + 2)] = s_order_id[WID];
        for (int i = 0; i < level; ++i)
          d_res_table[my_pos * (level + 2) + i + 1] = s_cur_res[WID][i];
        d_res_table[my_pos * (level + 2) + level + 1] = v;
        // for (int i = 0; i < C_NUM_VQ; ++i)
        // d_res_table[my_pos * C_NUM_VQ + i] = d_res_table_old[s_row[WID] * C_NUM_VQ + i];
        // d_res_table[my_pos * C_NUM_VQ + s_u[WID]] = v;
      }
      __syncwarp();
      off += warpSize;
    }
    if (LID == 0)
      block_iter_cnt[WID]++;
    __syncwarp();
  }
}

void parallelMatch(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,
    OrderCPU *h_order_obj,

    uint32_t *d_compact_encodings_,
    encodingMeta *enc_meta,
    uint32_t *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    numtype *h_num_u_candidate_vs_,

    ResTable *res_table,std::vector<LocalView> local_views)
{
  // gpuGraph *real_dg;
  // cuchk(cudaMalloc((void **)&real_dq, sizeof(gpuGraph)));
  // cuchk(cudaMalloc((void **)&real_dg, sizeof(gpuGraph)));

  // cuchk(cudaMemcpy(real_dq, dq, sizeof(gpuGraph), cudaMemcpyHostToDevice));
  // cuchk(cudaMemcpy(real_dg, dg, sizeof(gpuGraph), cudaMemcpyHostToDevice));

  // std::cout << "free memory: " << getFreeGlobalMemory(GPU_NUM) << std::endl;
  // std::cout << "MAX_L_FREQ: " << MAX_L_FREQ << std::endl;
  // std::cout << "NUM_VQ: " << NUM_VQ << std::endl;

  NUM_CAN_UB = 0;
  for (int i = 0; i < NUM_VQ; ++i)
    NUM_CAN_UB = std::max(NUM_CAN_UB, h_num_u_candidate_vs_[i]);
  NUM_CAN_UB = std::min(NUM_CAN_UB, MAX_DATA_DEGREE);
  cudaMemcpyToSymbol(C_NUM_CAN_UB, &NUM_CAN_UB, sizeof(uint32_t));
  // std::cout << "NUM_CAN_UB: " << NUM_CAN_UB << std::endl;

  /*--- tdfs ---*/
  CallStack *callstack_gpu;
  std::vector<CallStack> stk(NWARPS_TOTAL);

  vtype *candidate_space;
  cuchk(cudaMalloc((void **)&candidate_space, sizeof(vtype) * NUM_VQ * NUM_CAN_UB * NWARPS_TOTAL));

  for (int i = 0; i < NWARPS_TOTAL; i++)
  {
    auto &s = stk[i];
    s.candidates_ = candidate_space + i * NUM_CAN_UB * NUM_VQ;
    memset(s.iter, 0, sizeof(vtype) * MAX_VQ);
    memset(s.num_candidates_, 0, sizeof(numtype) * MAX_VQ);
    // memset(s.map_res_, 0, sizeof(s.map_res_));
  }
  cuchk(cudaMalloc(&callstack_gpu, NWARPS_TOTAL * sizeof(CallStack)));
  cuchk(cudaMemcpy(callstack_gpu, stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice));

  int *cur;
  cuchk(cudaMalloc((void **)&cur, sizeof(int)));
  cuchk(cudaMemset(cur, 0, sizeof(int)));

  unsigned long long *d_res;
  cuchk(cudaMalloc((void **)&d_res, sizeof(unsigned long long) * NWARPS_TOTAL));
  cuchk(cudaMemset(d_res, 0, sizeof(unsigned long long) * NWARPS_TOTAL));

  numtype &num_orders = h_order_obj->num_orders;
  OrderGPU order_gpu_temp(num_orders);
  cuchk(cudaMemcpy(order_gpu_temp.num_orders, &num_orders, sizeof(numtype), cudaMemcpyHostToDevice));
  // order_gpu_temp.num_orders = num_orders;
  cuchk(cudaMemcpy(order_gpu_temp.roots_, h_order_obj->roots.data(), sizeof(vtype) * num_orders, cudaMemcpyHostToDevice));
  for (int i = 0; i < num_orders; ++i)
  {
    cuchk(cudaMemcpy(order_gpu_temp.v_orders_ + i * NUM_VQ, h_order_obj->v_orders[i].data(), sizeof(vtype) * NUM_VQ, cudaMemcpyHostToDevice));
    cuchk(cudaMemcpy(order_gpu_temp.u2ls_ + i * NUM_VQ, h_order_obj->u2ls[i].data(), sizeof(int) * NUM_VQ, cudaMemcpyHostToDevice));
    cuchk(cudaMemcpy(order_gpu_temp.num_backward_neighbors_ + i * NUM_VQ, h_order_obj->num_backward_neighbors[i].data(), sizeof(numtype) * NUM_VQ, cudaMemcpyHostToDevice));
  }
  // cuchk(cudaMemcpy(order_gpu_temp.v_orders_, h_order_obj->v_orders.data(), sizeof(vtype) * num_orders * NUM_VQ, cudaMemcpyHostToDevice));
  // cuchk(cudaMemcpy(order_gpu_temp.u2ls_, h_order_obj->u2ls.data(), sizeof(int) * num_orders * NUM_VQ, cudaMemcpyHostToDevice));
  // cuchk(cudaMemcpy(order_gpu_temp.num_backward_neighbors_, h_order_obj->num_backward_neighbors.data(), sizeof(numtype) * num_orders * NUM_VQ, cudaMemcpyHostToDevice));
  offtype off = 0;
  for (int i = 0; i < num_orders; ++i)
  {
    for (int j = 0; j < NUM_VQ; ++j)
    {
      cuchk(cudaMemcpy(order_gpu_temp.backward_neighbors_ + off, h_order_obj->backward_neighbors[i][j].data(), sizeof(vtype) * h_order_obj->num_backward_neighbors[i][j], cudaMemcpyHostToDevice));
      off += NUM_VQ;
    }
  }

  // cuchk(cudaMemcpy(order_gpu_temp.root_u, &h_order_obj->root_u, sizeof(vtype), cudaMemcpyHostToDevice));
  // cuchk(cudaMemcpy(order_gpu_temp.v_order_, h_order_obj->v_order_, sizeof(vtype) * NUM_VQ, cudaMemcpyHostToDevice));
  // cuchk(cudaMemcpy(order_gpu_temp.u2l_, h_order_obj->u2l_, sizeof(int) * NUM_VQ, cudaMemcpyHostToDevice));
  // cuchk(cudaMemcpy(order_gpu_temp.e_order_, h_order_obj->e_order_, sizeof(etype) * NUM_EQ, cudaMemcpyHostToDevice));
  // cuchk(cudaMemcpy(order_gpu_temp.shared_neighbors_with_, h_order_obj->shared_neighbors_with_, sizeof(vtype) * NUM_VQ, cudaMemcpyHostToDevice));
  // cuchk(cudaMemcpy(order_gpu_temp.num_backward_neighbors_, h_order_obj->num_backward_neighbors_, sizeof(numtype) * NUM_VQ, cudaMemcpyHostToDevice));
  // offtype off = 0;
  // for (int i = 0; i < NUM_VQ; ++i)
  // {
  // cuchk(cudaMemcpy(order_gpu_temp.backward_neighbors_ + off, h_order_obj->backward_neighbors_[i], sizeof(vtype) * NUM_VQ, cudaMemcpyHostToDevice));
  // off += NUM_VQ;
  // }

  /*--- order, encoding ---*/
  OrderGPU *real_order_gpu;
  cuchk(cudaMalloc((void **)&real_order_gpu, sizeof(OrderGPU)));
  cuchk(cudaMemcpy(real_order_gpu, &order_gpu_temp, sizeof(OrderGPU), cudaMemcpyHostToDevice));

  numtype &num_blocks = enc_meta->num_blocks;

  // vtype u = h_order_obj->v_order_[0];

  vtype *d_res_table_old_;
  // unsigned long long h_num_res_old = h_num_u_candidate_vs_[u];
  unsigned long long h_num_res_old;
  cuchk(cudaMalloc((void **)&d_res_table_old_, sizeof(vtype) * NUM_VQ * MAX_RES));

  vtype *d_res_table_;
  unsigned long long h_num_res_new = 0;
  cuchk(cudaMalloc((void **)&d_res_table_, sizeof(vtype) * NUM_VQ * MAX_RES));

  int start_level = 0; // at `start_level`, tasks are done, just fetch from table. real match begins from `start_level + 1`.

  vtype *d_intersect_temp_storage;
  cuchk(cudaMalloc((void **)&d_intersect_temp_storage, sizeof(vtype) * NUM_CAN_UB * NWARPS_TOTAL));
  numtype *d_num_intersect_temp_storage;
  cuchk(cudaMalloc((void **)&d_num_intersect_temp_storage, sizeof(numtype) * NWARPS_TOTAL));
  cuchk(cudaMemset(d_num_intersect_temp_storage, 0, sizeof(numtype) * NWARPS_TOTAL));

  warmup<<<GRID_DIM, BLOCK_DIM>>>();
  cuchk(cudaDeviceSynchronize());

  TIME_INIT();
  TIME_START();

  // struct timespec time_st;
  // struct timespec time_ed;

  micro_init();
  micro_start();

  // clock_gettime(CLOCK_REALTIME, &time_st);
  // dim3 fj_block = BLOCK_DIM;
  // dim3 fj_grid = (h_num_res_old - 1) / fj_block.x + 1;
  // firstJoinKernel<<<fj_grid, fj_block>>>(u, d_u_candidate_vs_, h_num_u_candidate_vs_[u], d_res_table_old_);
  // cuchk(cudaDeviceSynchronize());
  // std::cout << "first join done" << std::endl;

  // cudaFree(d_u_candidate_vs_);
  // cudaFree(d_num_u_candidate_vs_);

  int level = 2;
  cuchk(cudaMemcpy(d_res_table_old_, res_table->res_table, sizeof(vtype) * res_table->size, cudaMemcpyHostToDevice));
  h_num_res_old = res_table->size / (level + 1); // 4 有多少组不同的匹配

#ifndef NDEBUG
  std::cout << "res_table: " << std::endl;
  for (int i = 0; i < res_table->size; ++i)
  {
    std::cout << res_table->res_table[i] << " ";
  }
#endif

 

  offtype v_off = 2;

  int *d_enc_pos_u_;
  // cuchk(cudaMalloc((void **)&d_enc_pos_u_, sizeof(int) * NUM_VQ));
  // cuchk(cudaMemcpy(d_enc_pos_u_, enc_meta->enc_pos_of_u_, sizeof(int) * NUM_VQ, cudaMemcpyHostToDevice));

  numtype *d_num_res_new;
  cuchk(cudaMalloc((void **)&d_num_res_new, sizeof(numtype)));
  cuchk(cudaMemset(d_num_res_new, 0, sizeof(numtype)));

  int *exceed;
  cuchk(cudaMalloc((void **)&exceed, sizeof(int)));
  cuchk(cudaMemset(exceed, 0, sizeof(int)));

  int h_exceed; //BFS
  while (v_off < NUM_VQ)
  {
    // printf("-------");
    //printf("当前BFS层数为%d\n",v_off);
    vertexJoinBFS<<<GRID_DIM, BLOCK_DIM>>>(
        dg->offsets_, dg->neighbors_, dg->degree_,
        real_order_gpu,
        v_off,
        d_intersect_temp_storage, d_num_intersect_temp_storage,

        d_compact_encodings_, num_blocks, d_enc_pos_u_,
        d_res_table_old_, h_num_res_old,
        d_res_table_, d_num_res_new,
        exceed);
    cuchk(cudaDeviceSynchronize());
    //printf("当前BFS层数%djieshu\n",v_off);
    cuchk(cudaMemcpy(&h_exceed, exceed, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_exceed == 1)
    {
      std::cout << "exceed" << std::endl;
      break;
    }

    // auto num_res_backup = h_num_res_old;
    cuchk(cudaMemcpy(&h_num_res_old, d_num_res_new, sizeof(numtype), cudaMemcpyDeviceToHost));
    std::swap(d_res_table_old_, d_res_table_);
    cuchk(cudaMemset(d_num_res_new, 0, sizeof(numtype)));

    v_off++;
  }

  micro_end();
  micro_print_local("bfs");
  total_match_time_us += diff_micro.tv_nsec / 1000.0;

  TIME_END();
  total_match_time_ms += kernel_time;

  // clock_gettime(CLOCK_REALTIME, &time_ed);

  // std::cout << "bfs: " << time_ed.tv_nsec - time_st.tv_nsec << "(ns)" << std::endl;
  // TIME_END();
  // PRINT_LOCAL_TIME("BFS");

  cuchk(cudaFree(d_res_table_));
  if (v_off == NUM_VQ)
  {
    std::cout << "res: " << h_num_res_old << std::endl;
    std::cout << std::endl;
    return;
  }
  // std::cout << "u: " << h_order_obj->v_order_[start_level] << " num_rows: " << h_num_res_old << std::endl;

#ifndef NDEBUG
  std::cout << "prepare done, entering match kernel" << std::endl;
#endif

  start_level = v_off;
  if (start_level >= NUM_VQ - 1)
  {
    // std::cout << "res: " << h_num_res_old << std::endl;
    return;
  }

  // STOP_LEVEL = std::max((uint32_t)start_level + 1, NUM_VQ / 2);
  // start_level--;
  STOP_LEVEL = start_level + 2;
  cuchk(cudaMemcpyToSymbol(C_STOP_LEVEL, &STOP_LEVEL, sizeof(uint32_t)));

  int *gpu_timeout_queue_space;
  cuchk(cudaMalloc(&gpu_timeout_queue_space, sizeof(int) * TIMEOUT_QUEUE_CAP * STOP_LEVEL));
  // vtype *gpu_timeout_candidate_queue_space;
  // cudaMalloc(&gpu_timeout_candidate_queue_space, sizeof(vtype) * TIMEOUT_QUEUE_CAP * WARP_SIZE);
  // cudaMemset(gpu_timeout_candidate_queue_space, UINT32_MAX, sizeof(vtype) * TIMEOUT_QUEUE_CAP * WARP_SIZE);
  Queue *gpu_timeout_queue;
  cudaMallocManaged(&gpu_timeout_queue, sizeof(Queue));
  gpu_timeout_queue->queue_ = gpu_timeout_queue_space;
  // gpu_timeout_queue->candidate_queue = gpu_timeout_candidate_queue_space;
  gpu_timeout_queue->size_ = TIMEOUT_QUEUE_CAP * (STOP_LEVEL);
  gpu_timeout_queue->resetQueue();
  cudaMemPrefetchAsync(gpu_timeout_queue, sizeof(Queue), GPU_NUM);
  cudaDeviceSynchronize();

  // size_t shared_used = 0;
  // shared_used += sizeof(OrderGPU);
  // shared_used += sizeof(gpuGraph);
  // shared_used += sizeof(CallStack) * WARP_PER_BLOCK;
  // shared_used += sizeof(unsigned long long) * WARP_PER_BLOCK;
  // shared_used += sizeof(int) * MAX_VQ;
  // shared_used += sizeof(vtype) * WARP_PER_BLOCK * MAX_VQ;
  // shared_used += sizeof(vtype) * MAX_VQ;
  // shared_used += sizeof(Arg_t) * WARP_PER_BLOCK;
  // shared_used += sizeof(bool) * WARP_PER_BLOCK;
  // shared_used += sizeof(vtype) * WARP_PER_BLOCK * MAX_VQ;
  // shared_used += sizeof(int) * WARP_PER_BLOCK;
  // shared_used += sizeof(int) * WARP_PER_BLOCK * MAX_VQ;
  // shared_used += sizeof(StealingArgs);
  // std::cout << shared_used * 1.0 / 1024 << "KB" << std::endl;
  // std::cout << "totally: " << shared_used * 1.0 / 1024 * GRID_DIM << "KB" << std::endl;

  // TIME_INIT();
  TIME_START();
  micro_start();

  // clock_gettime(CLOCK_REALTIME, &time_st);

  parallel_match_kernel<<<GRID_DIM, BLOCK_DIM>>>(
      dg->degree_, dg->offsets_, dg->neighbors_,
      callstack_gpu, cur, d_res,
      gpu_timeout_queue, real_order_gpu, h_order_obj->num_orders,

      start_level,

      d_compact_encodings_,
      enc_meta->num_blocks,
      d_u_candidate_vs_, d_num_u_candidate_vs_,

      d_res_table_old_, h_num_res_old);
  cuchk(cudaDeviceSynchronize());

  micro_end();
  micro_print_local("dfs");
  total_match_time_us += diff_micro.tv_nsec / 1000.0;
  // clock_gettime(CLOCK_REALTIME, &time_ed);

  // std::cout << "dfs: " << time_ed.tv_nsec - time_st.tv_nsec << "(ns)" << std::endl;

  TIME_END();
  PRINT_LOCAL_TIME("DFS_JOIN");
  total_match_time_ms += kernel_time;
  // PRINT_TOTAL_TIME("HYBRID_JOIN");
  // std::cout << "Parallel_match_kernel done" << std::endl;

  unsigned long long *h_res = new unsigned long long[NWARPS_TOTAL];
  cuchk(cudaMemcpy(h_res, d_res, sizeof(unsigned long long) * NWARPS_TOTAL, cudaMemcpyDeviceToHost));
  unsigned long long res = 0;
  for (int i = 0; i < NWARPS_TOTAL; ++i)
    res += h_res[i];
  std::cout << "res: " << res << std::endl;
  std::cout << std::endl;
}