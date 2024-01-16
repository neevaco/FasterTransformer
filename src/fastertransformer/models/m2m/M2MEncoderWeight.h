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
#include "src/fastertransformer/models/m2m/M2MEncoderLayerWeight.h"

namespace fastertransformer {

template<typename T>
struct M2MEncoderWeight {

    M2MEncoderWeight() = default;
    M2MEncoderWeight(const size_t                head_num,
                      const size_t                size_per_head,
                      const size_t                d_model,
                      const size_t                inter_size,
                      const size_t                vocab_size,
                      const size_t                num_layer,
                      const size_t                num_bucket_or_max_seq_len,
                      const size_t                tensor_para_size,
                      const size_t                tensor_para_rank,
                      const size_t                pipeline_para_size,
                      const size_t                pipeline_para_rank,
                      const bool                  use_gated_activation_para = false,
                      const PositionEmbeddingType pe_type                   = PositionEmbeddingType::absolute);
    ~M2MEncoderWeight();
    M2MEncoderWeight(const M2MEncoderWeight& other);
    M2MEncoderWeight& operator=(const M2MEncoderWeight& other);

    std::vector<M2MEncoderLayerWeight<T>*> m2m_encoder_layer_weights;
    LayerNormWeight<T>                      post_transformer_layernorm_weights;
    T*                                      embedding_table                         = nullptr;
    bool                                    use_gated_activation                    = false;
    PositionEmbeddingType                   position_embedding_type                 = PositionEmbeddingType::absolute;

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
    // refer to num_buckt if using relative position embedding
    // refer to max_seq_len if using absolute position embedding
    size_t num_bucket_or_max_seq_len_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;

    bool is_maintain_buffer = false;

    int real_weights_num_;

    // 3: [0] word embedding weight [1] post-LN weight [2] post-LN bias
    const static int weights_num_ = 3;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer