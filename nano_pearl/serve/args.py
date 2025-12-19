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
    
    # We override tensor_parallel_size to be the sum, usually.
    # But for PEARLConfig layout, we map specific args.

def parse_args(args: List[str], run_shell: bool = False) -> Tuple[PearlServerArgs, bool]:
    # Reuse minisgl parser logic by calling it first (not optimal as strict parsing might fail)
    # Better: Re-implement parser or add arguments to the parser.
    # Because ServerArgs is frozen, we can't easily modify instance.
    
    # Use standard argparse
    parser = argparse.ArgumentParser(description="Nano-PEARL Server Arguments")
    
    # Add PEARL specific args
    parser.add_argument("--draft-model-path", type=str, required=True, help="Path to draft model")
    parser.add_argument("--draft-tp-size", type=int, default=1, help="TP size for draft model")
    parser.add_argument("--target-tp-size", type=int, default=1, help="TP size for target model")
    
    # Add standard ServerArgs arguments (manually copied or use parse_known_args)
    # We use parse_known_args to let minisgl parse the rest, 
    # BUT we need to produce a single PearlServerArgs object.
    
    # Let's use parse_known_args with the minisgl parser? No, minisgl parser returns ServerArgs.
    # We will just recreate the parser adding our args.
    
    # Actually, simplifies things: Just use parse_known_args for our unique args, 
    # then pass EVERYTHING to minisgl parse_args, then merge?
    # But minisgl parse_args expects specific args.
    
    # Let's copy the needed args from minisgl/server/args.py effectively.
    # To avoid code duplication, we can assume users pass standard minisgl args + ours.
    
    # Strategy:
    # 1. Parse known args for draft/target stuff.
    # 2. Call minisgl.server.args.parse_args with the remaining args?
    # No, because --model-path is required by minisgl.
    
    # We will redefine the parser to include everything.
    
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--draft-model-path", type=str, required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument("--host", type=str, default="127.0.0.1", dest="server_host")
    parser.add_argument("--port", type=int, default=1919, dest="server_port")
    parser.add_argument("--num-tokenizer", type=int, default=0)
    parser.add_argument("--max-prefill-length", type=int, default=16384, dest="max_extend_tokens")
    parser.add_argument("--mem-fraction-static", type=float, default=0.9, dest="memory_ratio") # Default naming in vllm/others
    parser.add_argument("--max-running-requests", type=int, default=512, dest="max_running_req")
    parser.add_argument("--context-length", type=int, default=4096, dest="max_seq_len_override") # Approximating
    
    # PEARL Config
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gamma", type=int, default=-1)

    # Ignore other flags by using parse_known_args used by this script, 
    # OR we need to be robust.
    
    # Correct Approach:
    # Use minisgl.server.args.parse_args to parse standard structure.
    # Then parse EXTRA args manually.
    # Then construct PearlServerArgs.
    
    server_args, run_shell = parse_server_args(args, run_shell)
    
    extra_parser = argparse.ArgumentParser()
    extra_parser.add_argument("--draft-model-path", type=str, required=True)
    extra_parser.add_argument("--draft-tp-size", type=int, default=1)
    extra_parser.add_argument("--target-tp-size", type=int, default=1)
    extra_parser.add_argument("--gamma", type=int, default=-1)
    extra_parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    
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
