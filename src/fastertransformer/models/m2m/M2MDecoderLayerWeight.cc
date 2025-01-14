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

#include "src/fastertransformer/models/m2m/M2MDecoderLayerWeight.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
M2MDecoderLayerWeight<T>::M2MDecoderLayerWeight(const size_t head_num,
                                                  const size_t size_per_head,
                                                  const size_t d_model,
                                                  const size_t inter_size,
                                                  const size_t mem_d_model,
                                                  const size_t tensor_para_size,
                                                  const size_t tensor_para_rank,
                                                  const bool   use_gated_activation):
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    inter_size_(inter_size),
    mem_d_model_(mem_d_model),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    use_gated_activation_(use_gated_activation)
{
    // 11: self-attn QKV (fused), self-attn O, self-attn LN, cross-attn
    // Q, K, V, O, cross-attn LN, FC1, FC2, LN
    real_weights_num_ = (11 + (use_gated_activation ? 1 : 0)) * 2;

    FT_LOG_DEBUG("M2MDecoderLayerWeight " + std::string(__func__) + " start");

    initialize();
    mallocWeights();
    setWeightPtr();

    FT_LOG_DEBUG("M2MDecoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void M2MDecoderLayerWeight<T>::initialize()
{
    FT_LOG_DEBUG("M2MDecoderLayerWeight " + std::string(__func__) + " start");

    weights_size[0] = d_model_;
    weights_size[1] = d_model_ * 3 * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[2] = (head_num_ / tensor_para_size_) * size_per_head_ * d_model_;
    weights_size[3] = d_model_;
    weights_size[4] = d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[5] = mem_d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[6] = mem_d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[7] = (head_num_ / tensor_para_size_) * size_per_head_ * d_model_;
    weights_size[8] = d_model_;
    if (use_gated_activation_) {
        weights_size[9]  = d_model_ * (inter_size_ / tensor_para_size_);
        weights_size[10] = d_model_ * (inter_size_ / tensor_para_size_);  // for gated activation
        weights_size[11] = (inter_size_ / tensor_para_size_) * d_model_;
    }
    else {
        weights_size[9]  = d_model_ * (inter_size_ / tensor_para_size_);
        weights_size[10] = (inter_size_ / tensor_para_size_) * d_model_;
    }

    if (use_gated_activation_) {
        weights_size[12] = d_model_;
        weights_size[13] = 3 * (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[14] = d_model_;
        weights_size[15] = d_model_;
        weights_size[16] = (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[17] = (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[18] = (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[19] = d_model_;
        weights_size[20] = d_model_;
        weights_size[21] = (inter_size_ / tensor_para_size_);
        weights_size[22] = (inter_size_ / tensor_para_size_);  // for gated activation
        weights_size[23] = d_model_;
    }
    else {
        weights_size[11] = d_model_;
        weights_size[12] = 3 * (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[13] = d_model_;
        weights_size[14] = d_model_;
        weights_size[15] = (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[16] = (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[17] = (head_num_ / tensor_para_size_) * size_per_head_;
        weights_size[18] = d_model_;
        weights_size[19] = d_model_;
        weights_size[20] = (inter_size_ / tensor_para_size_);
        weights_size[21] = d_model_;
    }

    FT_LOG_DEBUG("M2MDecoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
M2MDecoderLayerWeight<T>::~M2MDecoderLayerWeight()
{
    FT_LOG_DEBUG("M2MDecoderLayerWeight " + std::string(__func__) + " start");

    if (is_maintain_buffer_ == true) {
        for (int i = 0; i < weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        layernorm_weights.gamma                               = nullptr;
        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attn_layernorm_weights.gamma                     = nullptr;

        cross_attention_weights.query_weight.kernel            = nullptr;
        cross_attention_weights.key_weight.kernel              = nullptr;
        cross_attention_weights.value_weight.kernel            = nullptr;
        cross_attention_weights.attention_output_weight.kernel = nullptr;
        cross_attn_layernorm_weights.gamma                     = nullptr;

        ffn_weights.intermediate_weight.kernel  = nullptr;
        ffn_weights.intermediate_weight2.kernel = nullptr;
        ffn_weights.output_weight.kernel        = nullptr;

        layernorm_weights.beta                              = nullptr;
        self_attention_weights.query_weight.bias            = nullptr;
        self_attention_weights.attention_output_weight.bias = nullptr;
        self_attn_layernorm_weights.beta                    = nullptr;

        cross_attention_weights.query_weight.bias            = nullptr;
        cross_attention_weights.key_weight.bias              = nullptr;
        cross_attention_weights.value_weight.bias            = nullptr;
        cross_attention_weights.attention_output_weight.bias = nullptr;
        cross_attn_layernorm_weights.beta                    = nullptr;

        ffn_weights.intermediate_weight.bias  = nullptr;
        ffn_weights.intermediate_weight2.bias = nullptr;
        ffn_weights.output_weight.bias        = nullptr;
        is_maintain_buffer_                   = false;
    }

    FT_LOG_DEBUG("M2MDecoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
M2MDecoderLayerWeight<T>::M2MDecoderLayerWeight(const M2MDecoderLayerWeight& other):
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    d_model_(other.d_model_),
    inter_size_(other.inter_size_),
    mem_d_model_(other.mem_d_model_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    use_gated_activation_(other.use_gated_activation_),
    real_weights_num_(other.real_weights_num_)
{

    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();
}

template<typename T>
M2MDecoderLayerWeight<T>& M2MDecoderLayerWeight<T>::operator=(const M2MDecoderLayerWeight& other)
{
    head_num_             = other.head_num_;
    size_per_head_        = other.size_per_head_;
    d_model_              = other.d_model_;
    inter_size_           = other.inter_size_;
    mem_d_model_          = other.mem_d_model_;
    tensor_para_size_     = other.tensor_para_size_;
    tensor_para_rank_     = other.tensor_para_rank_;
    use_gated_activation_ = other.use_gated_activation_;
    real_weights_num_     = other.real_weights_num_;

    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    setWeightPtr();

    return *this;
}

template<typename T>
void M2MDecoderLayerWeight<T>::setWeightPtr()
{
    layernorm_weights.gamma                               = weights_ptr[0];
    self_attention_weights.query_weight.kernel            = weights_ptr[1];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[2];
    self_attn_layernorm_weights.gamma                     = weights_ptr[3];

    cross_attention_weights.query_weight.kernel            = weights_ptr[4];
    cross_attention_weights.key_weight.kernel              = weights_ptr[5];
    cross_attention_weights.value_weight.kernel            = weights_ptr[6];
    cross_attention_weights.attention_output_weight.kernel = weights_ptr[7];
    cross_attn_layernorm_weights.gamma                     = weights_ptr[8];

    if (use_gated_activation_) {
        ffn_weights.intermediate_weight.kernel  = weights_ptr[9];
        ffn_weights.intermediate_weight2.kernel = weights_ptr[10];
        ffn_weights.output_weight.kernel        = weights_ptr[11];
    }
    else {
        ffn_weights.intermediate_weight.kernel = weights_ptr[9];
        ffn_weights.output_weight.kernel       = weights_ptr[10];
    }
    if (use_gated_activation_) {
        layernorm_weights.beta                              = weights_ptr[12];
        self_attention_weights.query_weight.bias            = weights_ptr[13];
        self_attention_weights.attention_output_weight.bias = weights_ptr[14];
        self_attn_layernorm_weights.beta                    = weights_ptr[15];

        cross_attention_weights.query_weight.bias            = weights_ptr[16];
        cross_attention_weights.key_weight.bias              = weights_ptr[17];
        cross_attention_weights.value_weight.bias            = weights_ptr[18];
        cross_attention_weights.attention_output_weight.bias = weights_ptr[19];
        cross_attn_layernorm_weights.beta                    = weights_ptr[20];

        ffn_weights.intermediate_weight.bias  = weights_ptr[21];
        ffn_weights.intermediate_weight2.bias = weights_ptr[22];
        ffn_weights.output_weight.bias        = weights_ptr[23];
    }
    else {
        layernorm_weights.beta                              = weights_ptr[11];
        self_attention_weights.query_weight.bias            = weights_ptr[12];
        self_attention_weights.attention_output_weight.bias = weights_ptr[13];
        self_attn_layernorm_weights.beta                    = weights_ptr[14];

        cross_attention_weights.query_weight.bias            = weights_ptr[15];
        cross_attention_weights.key_weight.bias              = weights_ptr[16];
        cross_attention_weights.value_weight.bias            = weights_ptr[17];
        cross_attention_weights.attention_output_weight.bias = weights_ptr[18];
        cross_attn_layernorm_weights.beta                    = weights_ptr[19];

        ffn_weights.intermediate_weight.bias = weights_ptr[20];
        ffn_weights.output_weight.bias       = weights_ptr[21];
    }

}

template<typename T>
void M2MDecoderLayerWeight<T>::mallocWeights()
{
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    is_maintain_buffer_ = true;
}

template<typename T>
void M2MDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG("M2MDecoderLayerWeight " + std::string(__func__) + " start");

    const auto tp_rank = std::to_string(tensor_para_rank_);

    loadWeightFromBin<T>(weights_ptr[0],
                         {weights_size[0]},
                         dir_path + "layer.SelfAttention.final_layer_norm.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[1],
                         {weights_size[1]},
                         dir_path + "layer.SelfAttention.qkv.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[2],
                         {weights_size[2]},
                         dir_path + "layer.SelfAttention.out_proj.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {weights_size[3]},
                         dir_path + "layer.SelfAttention.attn_layer_norm.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[4],
                         {weights_size[4]},
                         dir_path + "layer.CrossAttention.q.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5],
                         {weights_size[5]},
                         dir_path + "layer.CrossAttention.k.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[6],
                         {weights_size[6]},
                         dir_path + "layer.CrossAttention.v.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[7],
                         {weights_size[7]},
                         dir_path + "layer.CrossAttention.out_proj.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[8],
                        {weights_size[8]},
                        dir_path + "layer.CrossAttention.attn_layer_norm.weight.bin",
                        model_file_type);

    loadWeightFromBin<T>(weights_ptr[9],
                        {weights_size[9]},
                        dir_path + "layer.SelfAttention.fc1.weight." + tp_rank + ".bin",
                        model_file_type);
    loadWeightFromBin<T>(weights_ptr[10],
                        {weights_size[10]},
                        dir_path + "layer.SelfAttention.fc2.weight." + tp_rank + ".bin",
                        model_file_type);

    loadWeightFromBin<T>(weights_ptr[11],
                        {weights_size[11]},
                        dir_path + "layer.SelfAttention.final_layer_norm.bias.bin",
                        model_file_type);
    loadWeightFromBin<T>(weights_ptr[12],
                        {weights_size[12]},
                        dir_path + "layer.SelfAttention.qkv.bias." + tp_rank + ".bin",
                        model_file_type);
    loadWeightFromBin<T>(weights_ptr[13],
                        {weights_size[13]},
                        dir_path + "layer.SelfAttention.out_proj.bias." + tp_rank + ".bin",
                        model_file_type);
    loadWeightFromBin<T>(weights_ptr[14],
                        {weights_size[14]},
                        dir_path + "layer.SelfAttention.attn_layer_norm.bias.bin",
                        model_file_type);

    loadWeightFromBin<T>(weights_ptr[15],
                        {weights_size[15]},
                        dir_path + "layer.CrossAttention.q.bias." + tp_rank + ".bin",
                        model_file_type);
    loadWeightFromBin<T>(weights_ptr[16],
                        {weights_size[16]},
                        dir_path + "layer.CrossAttention.k.bias." + tp_rank + ".bin",
                        model_file_type);
    loadWeightFromBin<T>(weights_ptr[17],
                        {weights_size[17]},
                        dir_path + "layer.CrossAttention.v.bias." + tp_rank + ".bin",
                        model_file_type);
    loadWeightFromBin<T>(weights_ptr[18],
                        {weights_size[18]},
                        dir_path + "layer.CrossAttention.out_proj.bias." + tp_rank + ".bin",
                        model_file_type);
    loadWeightFromBin<T>(weights_ptr[19],
                        {weights_size[19]},
                        dir_path + "layer.CrossAttention.attn_layer_norm.bias.bin",
                        model_file_type);

    loadWeightFromBin<T>(weights_ptr[20],
                        {weights_size[20]},
                        dir_path + "layer.SelfAttention.fc1.bias." + tp_rank + ".bin",
                        model_file_type);
    loadWeightFromBin<T>(weights_ptr[21],
                        {weights_size[21]},
                        dir_path + "layer.SelfAttention.fc2.bias." + tp_rank + ".bin",
                        model_file_type);
  

    FT_LOG_DEBUG("M2MDecoderLayerWeight " + std::string(__func__) + " end");
}


template struct M2MDecoderLayerWeight<float>;
template struct M2MDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct M2MDecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
