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
    1. Parse PEARL-specific arguments first using parse_known_args
    2. Pass remaining arguments to minisgl.server.args.parse_args
    3. Merge both into PearlServerArgs
    """

    # First, extract PEARL-specific arguments
    pearl_parser = argparse.ArgumentParser(add_help=False)
    pearl_parser.add_argument("--draft-model-path", type=str, required=True)
    pearl_parser.add_argument("--draft-tp-size", type=int, default=1)
    pearl_parser.add_argument("--target-tp-size", type=int, default=1)
    pearl_parser.add_argument("--gamma", type=int, default=-1)
    pearl_parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    pearl_parser.add_argument("--use-radix-cache", action="store_true", default=True,
                             help="Use Radix Tree KV Cache (default: True). Set --no-use-radix-cache for Hash-based.")
    pearl_parser.add_argument("--no-use-radix-cache", dest="use_radix_cache", action="store_false",
                             help="Use Hash-based KV Cache instead of Radix Tree")

    # Parse PEARL-specific args, leaving the rest for minisgl
    pearl_args_ns, remaining_args = pearl_parser.parse_known_args(args)

    # Add --tensor-parallel-size to remaining args if not already present
    # PEARL uses the sum of draft_tp_size and target_tp_size for total parallelism
    if not any(arg.startswith('--tensor-parallel-size') or arg.startswith('--tp-size') for arg in remaining_args):
        total_tp_size = pearl_args_ns.draft_tp_size + pearl_args_ns.target_tp_size
        remaining_args.extend(['--tensor-parallel-size', str(total_tp_size)])

    # Convert use_radix_cache to --cache-type if not already present
    if not any(arg.startswith('--cache-type') for arg in remaining_args):
        cache_type = 'radix' if pearl_args_ns.use_radix_cache else 'naive'
        remaining_args.extend(['--cache-type', cache_type])

    # Now parse standard minisgl server arguments with remaining args
    server_args, run_shell = parse_server_args(remaining_args, run_shell)

    # Merge both argument namespaces
    kwargs = server_args.__dict__.copy()
    kwargs.update(pearl_args_ns.__dict__)

    # Create PearlServerArgs with merged arguments
    pearl_args = PearlServerArgs(**kwargs)
    return pearl_args, run_shell
