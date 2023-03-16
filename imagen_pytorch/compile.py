import torch
from torch import nn

from functorch.compile import aot_module, default_decompositions, default_partition
import torch._dynamo.backends.nvfuser as nvfuser
from torch._inductor import compile_fx
from functools import partial
from torch._inductor import decomposition

def compiler_inductor(fx_module, args):
    return compile_fx.compile_fx_inner(fx_module, args)

def compiler_nvfuser(fx_module, args):
    return partial(nvfuser.prims_executor, executor="nvfuser")(fx_module, args)

def compiler_passthrough(fx_module, args):
    return fx_module

def _compile_module(module: torch.nn.Module, fw_compiler, bw_compiler, decompositions = None, partition_fn = default_partition, dynamic_shapes: bool = False, name = None) -> torch.nn.Module:
    if name is None:
        name = type(module).__name__
    aot_m = aot_module(module, fw_compiler=fw_compiler, bw_compiler=bw_compiler, decompositions=decompositions,partition_fn=partition_fn)

    class CompiledModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.aot_module = aot_m

        def forward(self, *args, **kwargs):
            with torch.autograd.profiler.record_function(name):
                torch._functorch.config.use_dynamic_shapes = dynamic_shapes
                torch._dynamo.config.dynamic_shapes = dynamic_shapes

                return self.aot_module(*args, **kwargs)
        
    return CompiledModule()

def compile_module(backend: str, module: torch.nn.Module, dynamic_shapes: bool = False, name = None) -> torch.nn.Module:
    if backend == "nvfuser":
        partition_fn = nvfuser.nvprims_fw_bw_partition_fn
        module = _compile_module(module, fw_compiler=compiler_nvfuser, bw_compiler=compiler_nvfuser, decompositions=default_decompositions,partition_fn=partition_fn, dynamic_shapes=dynamic_shapes, name=name)
    elif backend == "inductor":
        module = _compile_module(module, fw_compiler=compiler_inductor, bw_compiler=compiler_inductor, decompositions=decomposition.fast_random_decomps(), dynamic_shapes=dynamic_shapes, name=name)
    elif backend == "passthrough":
        module = _compile_module(module, fw_compiler=compiler_passthrough, bw_compiler=compiler_passthrough, decompositions=default_decompositions, dynamic_shapes=dynamic_shapes, name=name)
    elif backend == "none":
        pass
    else:
        raise ValueError(f"invalid compiler backend: {backend}")

    return module

def compile_class(module: torch.nn.Module, clazz, backend: str, dynamic_shapes=False, path: str = ""):
    for name, child in module.named_children():
        path = (path + "." + name) if path else name
        if isinstance(child, clazz):
            print("Compiling", path)
            child = compile_module(backend=backend, module=child, dynamic_shapes=dynamic_shapes)
            setattr(module, name, child)
        else:
            compile_class(module=child, clazz=clazz, backend=backend, dynamic_shapes=dynamic_shapes, path=path)
