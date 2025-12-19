from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.server.args import ServerArgs, parse_args as parse_server_args
from minisgl.utils import init_logger

logger = init_logger(__name__, "PearlServerArgs")

@dataclass(frozen=True)
class PearlServerArgs(ServerArgs):
    draft_model_path: str | None = None
    draft_tensor_parallel_size: int = 1
    target_tensor_parallel_size: int = 1
    # PEARL-specific configs
    gamma: int = -1
    max_num_batched_tokens: int = 16384
    use_radix_cache: bool = True

    @property
    def tp_size(self) -> int:
        """Return total tensor-parallel size (draft + target)."""
        return int(self.draft_tensor_parallel_size + self.target_tensor_parallel_size)

    # Backwards-compat aliases for code that expects old names
    @property
    def draft_tp_size(self) -> int:
        return int(self.draft_tensor_parallel_size)

    @property
    def target_tp_size(self) -> int:
        return int(self.target_tensor_parallel_size)
    
    # We override tensor_parallel_size to be the sum, usually.
    # But for PEARLConfig layout, we map specific args.

def parse_args(args: List[str], run_shell: bool = False) -> Tuple[PearlServerArgs, bool]:
    # Reuse minisgl parser logic by calling it first (not optimal as strict parsing might fail)
    # Better: Re-implement parser or add arguments to the parser.
    # Because ServerArgs is frozen, we can't easily modify instance.
    
    # Use standard argparse
    parser = argparse.ArgumentParser(description="Nano-PEARL Server Arguments")
    
    # Add PEARL specific args
    # (PEARL-specific arguments are parsed later to avoid duplicate definitions.)
    
    # Add standard ServerArgs arguments (manually copied or use parse_known_args)
    # We will first parse PEARL-specific args, strip them, and then let minisgl parse the remaining arguments.

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1", dest="server_host")
    parser.add_argument("--port", type=int, default=1919, dest="server_port")
    parser.add_argument("--num-tokenizer", type=int, default=0)
    parser.add_argument("--served-model-name", type=str, default=None,
                        help="Custom model name for API responses (default: auto from model path)")
    parser.add_argument("--max-prefill-length", type=int, default=16384, dest="max_extend_tokens")
    parser.add_argument("--mem-fraction-static", type=float, default=0.9, dest="memory_ratio") # Default naming in vllm/others
    parser.add_argument("--max-running-requests", type=int, default=512, dest="max_running_req")
    parser.add_argument("--context-length", type=int, default=4096, dest="max_seq_len_override") # Approximating

    # PEARL Config (extra args are parsed first and removed before calling minisgl parser)
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--draft-model-path", type=str, required=True)
    extra_parser.add_argument("--draft-tp-size", type=int, default=1)
    extra_parser.add_argument("--target-tp-size", type=int, default=1)
    extra_parser.add_argument("--gamma", type=int, default=-1)
    extra_parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    extra_parser.add_argument("--mem-fraction-static", type=float, dest="memory_ratio", default=0.9,
                              help="Static memory fraction (mapped to memory_ratio)")
    extra_parser.add_argument("--use-radix-cache", action="store_true", default=True,
                              help="Use Radix Tree KV Cache (default: True). Set --no-use-radix-cache for Hash-based.")
    extra_parser.add_argument("--no-use-radix-cache", dest="use_radix_cache", action="store_false",
                              help="Use Hash-based KV Cache instead of Radix Tree")

    # Parse PEARL-specific args first and get remaining args for minisgl
    known, remaining_args = extra_parser.parse_known_args(args)

    # Now parse the remaining args with minisgl's parser
    server_args, run_shell = parse_server_args(remaining_args, run_shell)

    # Merge
    kwargs = server_args.__dict__.copy()
    kwargs.update(known.__dict__)

    # Normalize keys to match PearlServerArgs dataclass
    if "draft_tp_size" in kwargs:
        kwargs["draft_tensor_parallel_size"] = kwargs.pop("draft_tp_size")
    if "target_tp_size" in kwargs:
        kwargs["target_tensor_parallel_size"] = kwargs.pop("target_tp_size")

    pearl_args = PearlServerArgs(**kwargs)
    return pearl_args, run_shell
