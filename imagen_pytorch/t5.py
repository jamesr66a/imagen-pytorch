import torch
import transformers
from typing import List
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from einops import rearrange
import time
from functorch.compile import aot_module, ts_compile, default_decompositions
import torch.utils._pytree as pytree
from  torch import _dynamo as dynamo
from functools import partial
import torch._dynamo.backends.nvfuser as nvfuser
from torch import _inductor as inductor
from torch._inductor import compile_fx
from torch._inductor import decomposition
from torch.autograd.profiler import record_function

transformers.logging.set_verbosity_error()

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# config

MAX_LENGTH = 256

DEFAULT_T5_NAME = 'google/t5-v1_1-base'

T5_CONFIGS = {}

# singleton globals

def get_tokenizer(name):
    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()

    if "tokenizer" not in T5_CONFIGS[name]:
        tokenizer = T5Tokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)
        T5_CONFIGS[name]["tokenizer"] = tokenizer
    else:
        tokenizer = T5_CONFIGS[name]["tokenizer"]

    return tokenizer

def compiler_inductor(fx_module, args):
    with open("/tmp/graph-induct", "w+") as f:
        f.write(fx_module.print_readable(print_output=False))
    
    return compile_fx.compile_fx_inner(fx_module, args)

def compiler_nvfuser(fx_module, args):
    with open("/tmp/graph-nvfuser", "w+") as f:
        f.write(fx_module.print_readable(print_output=False))
    
    return partial(nvfuser.prims_executor, executor="nvfuser")(fx_module, args)

def print_fn(fx_module, args):
    with open("/tmp/graph-base", "w+") as f:
        f.write(fx_module.print_readable(print_output=False))
    
    return fx_module

def get_model(name, fuser_backend):
    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()

    if "model" not in T5_CONFIGS[name]:
        print('Loading T5 model...')
        s = time.time()
        model = T5EncoderModel.from_pretrained(name, torch_dtype=torch.bfloat16)
        e = time.time()
        print(f"T5 model loaded in {int(e-s)}s")

        if torch.cuda.is_available():
            model = model.cuda()

        pytree._register_pytree_node(
            transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
            pytree._odict_flatten,
            lambda values, context: transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions(pytree._odict_unflatten(values, context)),
        )
                
        print(f"Using {fuser_backend} fuser backend for T5")
        if fuser_backend == "nvfuser":
            partition_fn = nvfuser.nvprims_fw_bw_partition_fn
            model = aot_module(model, fw_compiler=compiler_nvfuser, bw_compiler=compiler_nvfuser, decompositions=default_decompositions,partition_fn=partition_fn)
        elif fuser_backend == "inductor":
            torch._functorch.config.use_dynamic_shapes = True
            torch._dynamo.config.dynamic_shapes = True
            model = aot_module(model, fw_compiler=compiler_inductor, bw_compiler=compiler_inductor, decompositions=decomposition.fast_random_decomps())
        elif fuser_backend == "passthrough":
            model = aot_module(model, fw_compiler=print_fn, bw_compiler=print_fn, decompositions=default_decompositions)
        elif fuser_backend == "none":
            pass
        else:
            raise ValueError(f"invalid fuser_backend {fuser_backend}")

        T5_CONFIGS[name]["model"] = model
    else:
        model = T5_CONFIGS[name]["model"]

    return model

def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model

# encoding text

def t5_tokenize(
    texts: List[str],
    name = DEFAULT_T5_NAME
):
    tokenizer = get_tokenizer(name)

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids
    attn_mask = encoded.attention_mask.bool()

    return input_ids, attn_mask

def t5_encode_tokenized_text(
    token_ids,
    attn_mask = None,
    pad_id = None,
    name = DEFAULT_T5_NAME,
    batch_size=None,
    fuser_backend="none",
):
    batch_size = batch_size or len(token_ids)
    assert batch_size > 0
    assert len(token_ids) % batch_size == 0

    assert exists(attn_mask) or exists(pad_id)
    
    t5 = get_model(name, fuser_backend)

    attn_mask = default(attn_mask, lambda: (token_ids != pad_id).long())

    assert len(attn_mask) % batch_size == 0

    t5.eval()

    for idx in range(0, len(token_ids), batch_size):
        with torch.no_grad():
            # with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
            token_ids_slice = token_ids[idx:idx+batch_size]
            attn_mask_slice = attn_mask[idx:idx+batch_size]
            with record_function("t5-forward"):
                output = t5(input_ids = token_ids_slice, attention_mask = attn_mask_slice)
            encoded_text = output.last_hidden_state.detach().float()
            # Check if any values are NaN in encoded_text
            # TODO this causes D&H sync
            # assert not torch.isnan(encoded_text).any()
            if idx == 0:
                all_encoded_text = encoded_text
            else:
                # TODO: dumb inefficient implementation. Prefer to preallocate output tensor
                # and write into slices of it
                all_encoded_text = torch.cat((all_encoded_text, encoded_text), dim=0)

    all_encoded_text = all_encoded_text.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.) # just force all embeddings that is padding to be equal to 0.
    return all_encoded_text

def t5_encode_text(
    texts: List[str],
    name = DEFAULT_T5_NAME,
    return_attn_mask = False
):
    token_ids, attn_mask = t5_tokenize(texts, name = name)

    encoded_text = t5_encode_tokenized_text(token_ids, attn_mask = attn_mask, name = name)

    if return_attn_mask:
        attn_mask = attn_mask.bool()
        return encoded_text, attn_mask
    
    return encoded_text
