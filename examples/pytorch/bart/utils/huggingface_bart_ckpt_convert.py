# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import configparser
import multiprocessing
from datetime import datetime
import logging
from pathlib import Path

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../../3rdparty/transformers/src/")

from transformers import BartForConditionalGeneration

import numpy as np
import torch  # pytype: disable=import-error

LOGGER = logging.getLogger(__name__)


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def fuse_decoder_qkv(model, factor, saved_dir, np_weight_data_type):
    model_dict = {}
    for name, param in model.named_parameters():
        if name.find("decoder") != -1 and name.find("SelfAttention") != -1:
            model_dict[name] = param

    for i in range(model.decoder.config.num_layers):
        shape = model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T.shape
        qkv = torch.cat([model_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"].T,
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"].T,
                         model_dict[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"].T], dim=-1)

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        qkv = qkv.cpu().detach().numpy().astype(np_weight_data_type)

        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"decoder.block.{i}.layer.0.SelfAttention.qkv.weight.{j}.bin"
            split_vals[j].tofile(saved_path.as_posix())


def split_and_convert_process(key, val, factor, saved_dir):
    if val.ndim == 2:
        val = val.transpose(1, 0)
    LOGGER.debug(f"key: {key}, val.shape: {val.shape}")

    if key.find("encoder.embed_positions.weight") != -1:
        saved_path = saved_dir / "encoder.embed_positions.weight.bin"
        val[2:, :].tofile(saved_path.as_posix())
    elif key.find("encoder.embed_tokens.weight") != -1:
        saved_path = saved_dir / "encoder.embed_tokens.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("encoder.layernorm_embedding.weight") != -1:
        saved_path = saved_dir / "encoder.final_layer_norm.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("encoder.layernorm_embedding.bias") != -1:
        saved_path = saved_dir / "encoder.final_layer_norm.bias.bin"
        val.tofile(saved_path.as_posix())
    elif (
        key.find("self_attn.k_proj.weight") != -1
        or key.find("self_attn.v_proj.weight") != -1
        or key.find("self_attn.q_proj.weight") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        qkv = key.split('self_attn.')[1][:1]
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.{qkv}.weight.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif (
        key.find("self_attn.k_proj.bias") != -1
        or key.find("self_attn.v_proj.bias") != -1
        or key.find("self_attn.q_proj.bias") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        qkv = key.split('self_attn.')[1][:1]
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.{qkv}.bias.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("self_attn.out_proj.weight") != -1:
        split_vals = np.split(val, factor, axis=0)
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.out_proj.weight.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("self_attn.out_proj.bias") != -1:
        split_vals = np.split(val, factor, axis=0)
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        for j in range(factor):
            saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.out_proj.bias.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
    elif key.find("self_attn_layer_norm.weight") != -1:
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.attn_layer_norm.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("self_attn_layer_norm.bias") != -1:
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.self_attn')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.attn_layer_norm.bias.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("fc1.weight") != -1:
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.fc1.')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.fc1.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("fc1.bias") != -1:
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.fc1.')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.fc1.bias.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("fc2.weight") != -1:
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.fc2.')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.fc2.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("fc2.bias") != -1:
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.fc2.')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.fc2.bias.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("final_layer_norm.weight") != -1:
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.final_layer_norm.')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.final_layer_norm.weight.bin"
        val.tofile(saved_path.as_posix())
    elif key.find("final_layer_norm.bias") != -1:
        if key.find("encoder") != -1:
            prefix = "encoder"
        else:
            prefix = "decoder"
        layer = int(key.split('layers.')[1].split('.final_layer_norm.')[0])
        saved_path = saved_dir / f"{prefix}.{layer}.layer.SelfAttention.final_layer_norm.bias.bin"
        val.tofile(saved_path.as_posix())
    # elif (
    #         key.find("SelfAttention.o.weight") != -1
    #         or key.find("EncDecAttention.o.weight") != -1
    #         or key.find("DenseReluDense.wo.weight") != -1
    # ):
    #     split_vals = np.split(val, factor, axis=0)
    #     for j in range(factor):
    #         saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
    #         split_vals[j].tofile(saved_path.as_posix())

    # elif (
    #         key.find("DenseReluDense.wi.weight") != -1
    #         or (key.find("encoder") != -1 and (
    #         key.find("SelfAttention.q.weight") != -1
    #         or key.find("SelfAttention.k.weight") != -1
    #         or key.find("SelfAttention.v.weight") != -1
    # )
    #         )
    #         or key.find("EncDecAttention.q.weight") != -1
    #         or key.find("EncDecAttention.k.weight") != -1
    #         or key.find("EncDecAttention.v.weight") != -1
    # ):
    #     split_vals = np.split(val, factor, axis=-1)
    #     for j in range(factor):
    #         saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
    #         split_vals[j].tofile(saved_path.as_posix())
    # elif (
    #         key.find("DenseReluDense.wi_0.weight") != -1
    #         or key.find("DenseReluDense.wi_1.weight") != -1
    # ):
    #     # For gated activation.
    #     if key.find("DenseReluDense.wi_0.weight") != -1:
    #         saved_key = key.replace("wi_0", "wi")
    #     elif key.find("DenseReluDense.wi_1.weight") != -1:
    #         saved_key = key.replace("wi_1", "wi2")
    #     split_vals = np.split(val, factor, axis=-1)
    #     for j in range(factor):
    #         saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
    #         split_vals[j].tofile(saved_path.as_posix())
    # elif key.find("relative_attention_bias") != -1:
    #     split_vals = np.split(val, factor, axis=0)
    #     for j in range(factor):
    #         saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
    #         split_vals[j].tofile(saved_path.as_posix())
    # elif (
    #         key.find("decoder") != -1 and
    #         (
    #                 key.find("SelfAttention.q.weight") != -1
    #                 or key.find("SelfAttention.k.weight") != -1
    #                 or key.find("SelfAttention.v.weight") != -1
    #         )
    # ):
    #     pass
    elif key.find("encoder.embed_tokens.weight") != -1 or \
            key.find("decoder.embed_tokens.weight") != -1:
        LOGGER.warning(f"Not save {key}, using shared.weight directly.")
    else:
        LOGGER.warning(f"Not save '{key}' with shape {val.shape}")


def convert_checkpoint(args):
    saved_dir = Path(args.saved_dir) / f"{args.inference_tensor_para_size:d}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    bart_model = BartForConditionalGeneration.from_pretrained(args.in_file)
    hf_config = vars(bart_model.config)
    config = configparser.ConfigParser()

    config["encoder"] = {}
    config["encoder"]["num_heads"] = str(hf_config["encoder_attention_heads"])
    config["encoder"]["d_kv"] = str(hf_config["d_model"] // hf_config["encoder_attention_heads"])
    config["encoder"]["d_model"] = str(hf_config["d_model"])
    config["encoder"]["d_ff"] = str(hf_config["encoder_ffn_dim"])
    config["encoder"]["num_layers"] = str(hf_config["encoder_layers"])
    config["encoder"]["vocab_size"] = str(hf_config["vocab_size"])
    config["encoder"]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])

    config["decoder"] = {}
    config["decoder"]["num_heads"] = str(hf_config["decoder_attention_heads"])
    config["decoder"]["d_kv"] = str(hf_config["d_model"] // hf_config["decoder_attention_heads"])
    config["decoder"]["d_model"] = str(hf_config["d_model"])
    config["decoder"]["d_ff"] = str(hf_config["decoder_ffn_dim"])
    config["decoder"]["num_layers"] = str(hf_config["decoder_layers"])
    config["decoder"]["vocab_size"] = str(hf_config["vocab_size"])
    config["decoder"]["max_pos_seq_len"] = str(hf_config["max_position_embeddings"])

    with open((saved_dir / "config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)
    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    i_gpu_num = args.inference_tensor_para_size

    pool = multiprocessing.Pool(args.processes)
    pool.starmap_async(split_and_convert_process,
                       [(name, param.cpu().detach().numpy().astype(np_weight_data_type), i_gpu_num, saved_dir)
                        for name, param in bart_model.state_dict().items()])

    pool.close()
    pool.join()

    # fuse_decoder_qkv(bart_model, i_gpu_num, saved_dir, np_weight_data_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file", required=True)
    parser.add_argument("-inference_tensor_para_size", "-i_g", type=int, help="How many gpus for inference",
                        required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 4)",
                        default=4)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
