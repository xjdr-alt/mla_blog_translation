
# DeepSeek-V2 High-performance Inference Optimization Notes: MLA Optimization

## Preface

Recently, the DeepSeek-V2 model released by Iluvatar has gained widespread attention in academia and industry. As a 236B-parameter MoE large model, DeepSeek-V2 activates only 21B parameters per token through its unique DeepSeekMoE architecture design. Moreover, by using the newly proposed MLA mechanism to replace the traditional MHA and MQA attention mechanisms, it greatly reduces the size of the KV Cache during inference. As a result, DeepSeek-V2 achieves model performance comparable to GPT-4 at a lower inference cost.

The MLA mechanism is a core innovation in DeepSeek-V2. As researchers in the field of computer systems, we naturally dare not make arbitrary comments on the algorithmic design of MLA from an AI/ML perspective. However, from a systems perspective, MLA is undoubtedly an excellent design. In recent years, one major reason for the persistently high inference cost of large models is the low utilization of GPU computing power. With the emergence of dedicated circuits such as Tensor Cores, the computing power of modern high-performance GPUs far exceeds their memory bandwidth. To ensure that the computing units of the GPU are not idle and achieve good utilization of computing resources (i.e., Memory Fetch Utilization, or MFU), each byte of data read into the GPU often needs to participate in hundreds of computations. However, due to various constraints, the workloads of large model inference are often unable to provide such high computational intensity. The parameters read into the GPU are discarded and replaced by the next parameter before participating in enough computations, resulting in memory bandwidth becoming the performance bottleneck of the entire inference process. One major obstacle is the space occupied by the KV Cache: GPU memory is often very limited, and a larger KV Cache leads to a smaller number of requests that can be processed simultaneously, i.e., a smaller batch size. Works such as vLLM optimize the memory utilization of the KV Cache from this perspective to improve the efficiency of the inference process. On the other hand, for traditional MHA or GQA operators, during the attention computation, all data read from the KV Cache only participates in one or a few computations, resulting in very low MFU for these operators. Moreover, since each request has its own KV Cache, this problem cannot be solved by increasing the batch size. The MLA operator, judging from its computational characteristics, solves both of these problems simultaneously: on one hand, it greatly reduces the size of the KV Cache through low-rank compression; on the other hand, the multi-head attention mechanism after MLA decompression provides high computational intensity, which helps fully utilize the GPU's computing resources. It is evident that the MLA operator is an attention mechanism tailored to the characteristics of modern GPU hardware, achieving a rebalancing of storage and computation to fully leverage the strengths of modern GPUs.

The released code of DeepSeek-V2 does not heavily optimize the MLA operator. We attempted to reproduce some of the optimization points that may be involved in the MLA operator during the inference stage (specifically, the decoding stage of inference) and conducted evaluations and analyses.

The address of all the code involved in this article: https://github.com/madsys-dev/deepseekv2-profile

## Computational Process of the MLA Module

Given an input vector $h_t \in \mathbb{R}^{B \times L \times 5120}$, where $B$ is the batch size and $L$ is the sequence length, the computation process of MLA is as follows.

### Q Vector

In DeepSeek-V2, the Q vector also adopts low-rank compression. First, the input vector is projected to a 1536-dimensional low-dimensional space:
$$ c_t^Q = W^{DQ} h_t \in \mathbb{R}^{B \times L \times 1536} $$
Then, it is projected to the $\mathbb{R}^{H \times 128}$ multi-head vector space (where $H=128$ is the number of heads), obtaining the first part of the Q vector:
$$ q_t^C = W^{UQ} c_t^Q \in \mathbb{R}^{B \times L \times H \times 128} $$
Next, it is projected to $\mathbb{R}^{H \times 64}$ and uses RoPE to embed position information, obtaining the second part of the Q vector:
$$ q_t^R = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{B \times L \times H \times 64} $$
The two parts are concatenated to obtain the final Q vector:
$$ q_t = [q_t^C, q_t^R] \in \mathbb{R}^{B \times L \times H \times 192} $$

### KV Vector

When computing the KV vector, the input vector first needs to be projected to a 512-dimensional joint compressed representation:
$$ c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{B \times L \times 512} $$

Similar to the computation process of the Q vector, the first part of the K vector is obtained by projecting $c_t^{KV}$ through decompression to the $\mathbb{R}^{H \times 128}$ multi-head vector space:
$$ k_t^C = W^{UK} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$
The second part of K is obtained by projecting the input vector to a 64-dimensional vector space and applying RoPE to embed position information:
$$ k_t^R = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{B \times L \times 64} $$
Unlike Q, the complete K is obtained by broadcasting the second part of K to each head and concatenating it with the first part:
$$ k_t = \begin{bmatrix}
    k_{t,1}^C & k_t^R \\ 
    k_{t,2}^C & k_t^R \\
    \vdots & \vdots \\
    \end{bmatrix} \in \mathbb{R}^{B \times L \times H \times 192} $$
That is, the RoPE part of each head is exactly the same.

The computation of the V vector is relatively simple, directly decompressing $c_t^{KV}$ to $\mathbb{R}^{H \times 128}$:
$$ v_t = W^{UV} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$

### Attention Computation

The attention computation process is no different from traditional MHA. First, the attention score is computed:
$$ a = \mathrm{softmax}\left(\frac{q_t^\top k_t + \mathrm{Mask}}{\sqrt{192}}\right) = 
\mathrm{softmax}\left(\frac{{q_t^C}^\top k_t^C + {q_t^R}^\top k_t^R + \mathrm{Mask}}{\sqrt{128 + 64}} \right)
\in \mathbb{R}^{B \times L \times H \times L} $$
The weighted sum of V is computed, and all heads are flattened to obtain the Attention output:
$$ o = a \cdot v_t \in \mathbb{R}^{B \times L \times H \times 128} \cong \mathbb{R}^{B \times L \times 16384} $$
After projection with another matrix, the final output of MLA is obtained:
$$ u = W^O o \in \mathbb{R}^{B \times L \times 5120} $$

## MLA Analysis of Open Source Code

```python
def forward(...):
    bsz, q_len, _ = hidden_states.size()
    
    # Compute Q: first reduce dimension then increase dimension
    # Compared to directly using a matrix of size [5120, 24576], 
    # the low-rank decomposition [5120, 1536] * [1536, 24576] greatly reduces storage space and computation
    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    # Split rope and non-rope parts
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    
    # Compute KV
    # An optimized MLA KVCache implementation only needs to cache this compressed_kv, but it is actually expanded later
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    # Here compressed_kv corresponds to c_t^{KV} in the formula
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    # Expand MLA to the standard MHA form
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )
    # Since kv_b_proj packs W^{UK} and W^{UV} together, separate them
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    ...
    # Add rope to the parts that need rope
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    
    # Update and concatenate historical KVCache
    # It can be seen that the expanded MHA KVCache is stored here
    # where q_head_dim equals qk_nope_head_dim + qk_rope_head_dim
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # The subsequent code is the standard MHA code, no need to elaborate
    ...
```

## Implementation Optimization of the MLA Module

### KV Caching

In the decoding process of the original transformer model, the KV vectors corresponding to all tokens need to be computed in each iteration, which often incurs significant overhead. In fact, the values of these KV vectors are the same in each iteration; therefore, we can adopt a "space-for-time" strategy and cache the values of the KV vectors from previous iterations, so that in subsequent iterations, there is no need to recompute the KV vectors, greatly reducing the computation in the model inference process.

However, in traditional Attention operators such as MHA, this space-for-time strategy often overcorrects the problem. Since the KV cache occupies a large space and the data in the KV cache only participates in one computation in each iteration, although the computation is reduced after using the KV cache, the memory occupation and memory bandwidth requirements increase sharply, becoming a new bottleneck limiting the efficiency of large model inference. The design of MLA greatly reduces the occupation of the KV cache by sharing the compressed KV representation among multiple heads. On the other hand, since the Compressed KV participates in computations in each head, the 128 heads of DeepSeek-V2 can provide sufficient computational intensity, thus significantly improving the MFU of the Attention part.

In the open-source version, the MLA operator caches the complete KV Cache, losing the above-mentioned advantages of MLA. We tried to change it to cache the compressed KV Cache and compare it with caching the complete KV Cache. Of course, we also cache the RoPE-processed k_pe into the KV Cache here.

```python
# CacheCompressed
def forward(self, hidden_states_q: torch.Tensor, q_position_ids: torch.LongTensor, compressed_kv: torch.Tensor):
    ...
    kv_seq_len = compressed_kv.size(1)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, kv_seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = self.kv_b_proj(compressed_kv) \
        .view(bsz, kv_seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim) \
        .transpose(1, 2)
    
    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    ... 
```

The KV Cache space occupation and computation amount of the two implementations are shown in the table below:

| Implementation Version | Cache Size per Token per Layer | Computation Amount per Token per Layer |
| :---: | :---: | :---: |
| CacheDecompressed (CD) | 81.92 kB | 0.08 MFLOP |
| CacheCompressed (CC) | 1.152 kB | 33.64 MFLOP |

As can be seen, although the CacheDecompressed strategy can save almost all floating-point computations, its memory occupation reaches 81.92kB per token. This makes it very easy for the bottleneck of CacheDecompressed to be on memory capacity and memory bandwidth. In contrast, the memory occupation of CacheCompressed is reduced by about 98.6%. Therefore, we can expect the CacheCompressed strategy to leverage the various hardware capabilities of the GPU in a more balanced manner and provide a larger batch size, thereby reducing inference costs.

We tested the above implementations on A100-PCIe-40G (Compute80 architecture) and GeForce RTX 4080 (Compute89 architecture), respectively. For a single request, the performance of each implementation is shown in the figure below:

![](data/caching-B1.png)

The performance of CacheDecompressed is significantly better than CacheCompressed. This indicates that the CacheCompressed strategy needs further optimization to reduce the computation per token in order to achieve better performance.

When Batch Size=32, the performance of each implementation is shown in the figure below:

![](data/caching-B32.png)

The test results are basically the same as when querying a single request.

### Projection Absorption

The above analysis and experimental results show that compared to caching the complete KV Cache, caching the compressed KV Cache leads to a significant performance degradation. Another important issue is that the current CacheDecompressed implementation does not actually alleviate the problem of the KV Cache being too large, because when computing MLA, the decompressed complete KV Cache still needs to be stored, which is likely to cause OOM crashes.

Fortunately, the DeepSeek-V2 paper proposes that the KVdecompression matrix can be absorbed into the Q-projection and Out-projection, so that the final Attention result can be computed directly without decompressing the KV Cache.
For the absorption of K, in the formula for computing the Attention Score, the non-RoPE part can be expanded as follows:
$$
{q_t^C}^\top k_t^C = (W^{UQ} c_t^Q)^{\top} W^{UK} c_t^{KV} = {c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK} c_t^{KV} = ({c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK}) c_t^{KV} 
$$
That is, by applying the associative law of matrix multiplication, it can be changed to computing $({c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK})$, avoiding the decompression of the complete K matrix. Furthermore, in the original version of the decompression process, since the key of each token needs to be multiplied with $W^{UK}$ to obtain it, the computation amount is relatively large; after matrix absorption, $W^{UK}$ only needs to be multiplied with the single vector $q_t^C$, which also greatly reduces the floating-point computation.

For the absorption of V, the situation is slightly more complex. For clarity of expression, we use the Einstein summation convention to describe this process:
```python
v_t = einsum('hdc,blc->blhd', W_UV, c_t_KV) # (1)
o   = einsum('bqhl,blhd->bqhd', a, v_t)     # (2)
u   = einsum('hdD,bhqd->bhD', W_o, o)       # (3)

# Combine the above three equations to get the overall computation process
u   = einsum('hdc,blc,bqhl,hdD->bhD', W_UV, c_t_KV, a, W_o)

# Use the associative law to change the computation order
o_  = einsum('bhql,blc->bhqc', a, c_t_KV) # (4)
o   = einsum('bhqc,hdc->bhqd', o_, W_UV)  # (5)
u   = einsum('hdD,bhqd->bhD', W_o, o)     # (6)
```

The specific code implementation is as follows:
```python
# Absorbed_CacheCompressed
def forward(hidden_states_q: torch.Tensor, q_position_ids: torch.LongTensor, compressed_kv: torch.Tensor):
    ...
    kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
    q_absorb = kv_b_proj[:, :self.qk_nope_head_dim,:]
    out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]
    
    cos, sin = self.rotary_emb(q_pe)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, q_position_ids)
    
    qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, qk_head_dim)
    # Here the computation order of q_nope is changed
    query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
    query_states[:, :, :, self.kv_lora_rank :] = q_pe
    
    ...

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(q_nope.dtype)
    # Here the computation order of attn_output is changed
    attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
    attn_output = torch.einsum('bhqc,hdc->bhqd', attn_output, out_absorb)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
    attn_output = self.o_proj(attn_output)
```

#### Move Elision
However, this is not enough to fully unleash the power of MLA. In the original code, query_states and key_states are obtained by concatenating the RoPE and non-RoPE parts:
```python
def forward(...):
    ...
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, kv_seq_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    ...
```
When we adopt the above optimization, this concatenation process will generate a large amount of useless data copying and broadcasting, while also occupying a lot of memory space leading to OOM. To address this, we adopt the MoveElision optimization strategy, which omits the concatenation of the RoPE and non-RoPE parts here, and instead directly computes the Attention Score of the two parts separately and adds them together (considering $q_t^\top k_t = {q_t^C}^\top k_t^C + {q_t^R}^\top k_t^R$):
```python
# Absorbed_CacheCompressed_MoveElision
def forward(...):
    ...
    # qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, qk_head_dim)
    # query_states[:, :, :, : self.kv_lora_rank] = torch.einsum('hdc,bhid->bhic', q_absorb, q_nope)
    # query_states[:, :, :, self.kv_lora_rank :] = q_pe

    # key_states = k_pe.new_empty(bsz, self.num_heads, kv_seq_len, qk_head_dim)
    # key_states[:, :, :, : self.kv_lora_rank] = compressed_kv.unsqueeze(1)
    # key_states[:, :, :, self.kv_lora_rank :] = k_pe

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

    attn_weights = torch.matmul(q_pe, k_pe.transpose(2, 3)) + torch.einsum('bhqc,blc->bhql', q_nope, compressed_kv)
    attn_weights *= self.softmax_scale
    ...
```

Thus, we obtained the following four optimized implementations:

| Implementation Version | Cache Size per Token per Layer | Computation Amount per Token per Layer |
| :---: | :---: | :---: |
| CacheDecompressed (CD) | 81.92 kB | 0.08 MFLOP |
| CacheCompressed (CC) | 1.152 kB | 33.64 MFLOP |
| Absorbed_CacheCompressed (A_CC) | 1.152 kB | 0.28 MFLOP |
| Absorbed_CacheCompressed_MoveElision (A_CC_ME) | 1.152 kB | 0.28 MFLOP |

The test results on A100-PCIe-40G and GeForce RTX 4080 are shown below, which are completely consistent with the theoretical analysis.

![](data/absorption-B1-annotated.png)

![](data/absorption-B32-annotated.png)

It is worth noting that when the MoveElision strategy is adopted, due to the reduction in memory occupation, the batch size and sequence length that can be processed are significantly increased, fully demonstrating the advantage of MLA's compressed representation.

#### Materializing Projection Matrices?

The DeepSeek-V2 paper states:
> ..., we can absorb $W^{UK}$ into $W^{UQ}$, and $W^{UV}$ into $W^O$.

However, we don't seem to have the necessity to further change the order, preprocess the model parameters, multiply $W^{UK}$ with $W^{UQ}$, and multiply $W^{UV}$ with $W^O$. This is because the result of multiplying $W^{UK}$ with $W^{UQ}$ can be viewed as $H$ low-rank (not exceeding 128) matrices of size $1536 \times 512$, and the result of multiplying $W^{UV}$ with $W^O$ can be viewed as $H$ low-rank matrices of size $5120 \times 512$. Compared to performing projection with these particularly large low-rank matrices, it is obviously more advantageous to multiply them successively according to the low-rank decomposition form. Therefore, we believe that this optimization step is not very necessary.

We implemented this optimized version (AM_CC_ME) and conducted tests. The test results can validate our viewpoint.

![](data/am-B1-annotated.png)

![](data/am-B32-annotated.png)

The performance after this optimization is significantly worse than the original version, especially when the sequence length is small and the computation time of these projections dominates.

## Future Optimizations

The current code implementation is based on matrix multiplication, so it requires computing the complete attention score matrix during the computation process. For further optimization, we can consider approaches similar to FlashAttention, i.e., reading the entire KV-pair at once for computation. Since the K and V of MLA share the same compressed representation (in fact, the optimized MLA implementation above is very similar to MQA satisfying $K=V$), this can further reduce memory reads and increase computational intensity.
