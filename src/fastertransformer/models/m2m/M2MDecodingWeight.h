/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/models/m2m/M2MDecoderLayerWeight.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct M2MDecodingWeight {

    M2MDecodingWeight() = default;
    M2MDecodingWeight(const size_t                head_num,
                       const size_t                size_per_head,
                       const size_t                d_model,
                       const size_t                inter_size,
                       const size_t                vocab_size,
                       const size_t                num_layer,
                       const size_t                mem_d_model,
                       const size_t                num_bucket_or_max_seq_len,
                       const size_t                tensor_para_size,
                       const size_t                tensor_para_rank,
                       const size_t                pipeline_para_size,
                       const size_t                pipeline_para_rank,
                       const bool                  use_gated_activation_para    = false,
                       const PositionEmbeddingType position_embedding_type_para = PositionEmbeddingType::absolute);
    ~M2MDecodingWeight();
    M2MDecodingWeight(const M2MDecodingWeight& other);
    M2MDecodingWeight& operator=(const M2MDecodingWeight& other);

    std::vector<M2MDecoderLayerWeight<T>*> decoder_layer_weights;
    const T*                                pre_decoder_embedding_table             = nullptr;
    LayerNormWeight<T>                      post_decoder_layernorm;
    DenseWeight<T> post_decoder_embedding;  // Megatron embedding is weight + bias, so prefer to use a separate weight
                                            // class to store
    const T*                                sinusoidal_position_embedding = nullptr;
    bool use_gated_activation = false;
    // 0 = relative_position_embedding,  1 = absolute_position_embedding
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::absolute;

    void loadModel(std::string dir_path);
    void resizeLayer(const int num_layer);

private:
    void setWeightPtr();
    void mallocWeights();
    bool isValidLayerParallelId(int l);
    void initialize();

    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    size_t inter_size_;
    size_t vocab_size_;
    size_t num_layer_;
    size_t mem_d_model_;
    // refer to num_buckt if using relative position embedding
    // refer to max_seq_len if using absolute position embedding
    size_t num_bucket_or_max_seq_len_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;
    bool   is_maintain_buffer_ = false;

    int real_weights_num_;

    // 5: [0] word embedding weight [1] word embedding 2 weight
    // [2] post-LN weight [3] post-LN bias [4] embed_positions (precalculated sinusoidal)
    const static int weights_num_ = 5;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer
