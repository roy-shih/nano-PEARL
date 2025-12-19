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
    """Extended ServerArgs with PEARL-specific parameters."""
    draft_model_path: str | None = None
    draft_tp_size: int = 1
    target_tp_size: int = 1
    gamma: int = -1
    max_num_batched_tokens: int = 16384
    use_radix_cache: bool = True

def parse_args(args: List[str], run_shell: bool = False) -> Tuple[PearlServerArgs, bool]:
    """
    Parse arguments for Nano-PEARL server.

    Strategy:
    1. Use minisgl.server.args.parse_args to parse standard server arguments
    2. Parse PEARL-specific arguments (draft model, gamma, etc.) separately
    3. Merge both into PearlServerArgs
    """

    # First, parse standard minisgl server arguments
    server_args, run_shell = parse_server_args(args, run_shell)
    
    extra_parser = argparse.ArgumentParser()
    extra_parser.add_argument("--draft-model-path", type=str, required=True)
    extra_parser.add_argument("--draft-tp-size", type=int, default=1)
    extra_parser.add_argument("--target-tp-size", type=int, default=1)
    extra_parser.add_argument("--gamma", type=int, default=-1)
    extra_parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    extra_parser.add_argument("--use-radix-cache", action="store_true", default=True,
                             help="Use Radix Tree KV Cache (default: True). Set --no-use-radix-cache for Hash-based.")
    extra_parser.add_argument("--no-use-radix-cache", dest="use_radix_cache", action="store_false",
                             help="Use Hash-based KV Cache instead of Radix Tree")
    
    # We need to parse args again to get these values.
    # Note: parse_server_args consumes known args? No, it uses sys.argv[1:] passed to it.
    
    # We should parse ONLY our extra args from the same list.
    known, unknown = extra_parser.parse_known_args(args)
    
    # Merge
    kwargs = server_args.__dict__.copy()
    kwargs.update(known.__dict__)
    
    # Remove keys that shouldn't be in PearlServerArgs if any?
    # PearlServerArgs inherits ServerArgs, so keeping them is fine.
    # But we added new fields.
    
    pearl_args = PearlServerArgs(**kwargs)
    return pearl_args, run_shell
