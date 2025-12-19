from __future__ import annotations

import sys
import multiprocessing as mp
from minisgl.utils import init_logger
from minisgl.server.api_server import run_api_server
from minisgl.tokenizer import tokenize_worker

from nano_pearl.serve.args import parse_args
from nano_pearl.serve.pearl_scheduler import run_pearl_scheduler

logger = init_logger(__name__, "PearlLauncher")

def launch_server(run_shell: bool = False) -> None:
    # 1. Save original argv
    original_argv = sys.argv.copy()
    
    # 2. Parse Args (includes PEARL-specific arguments)
    pearl_args, run_shell = parse_args(sys.argv[1:], run_shell)
    
    # 3. ⭐ CRITICAL FIX: Clean sys.argv before calling run_api_server
    #    Remove PEARL-specific arguments that minisgl doesn't recognize
    #    This prevents "unrecognized arguments" error in minisgl's internal parsing
    cleaned_argv = [original_argv[0]]  # Keep program name
    
    # Add only minisgl-compatible arguments
    cleaned_argv.extend(['--model-path', pearl_args.model_path])
    cleaned_argv.extend(['--host', pearl_args.server_host])
    cleaned_argv.extend(['--port', str(pearl_args.server_port)])
    cleaned_argv.extend(['--tp-size', str(pearl_args.tp_size)])
    
    if hasattr(pearl_args, 'tokenizer_path') and pearl_args.tokenizer_path:
        cleaned_argv.extend(['--tokenizer-path', pearl_args.tokenizer_path])
    
    # Replace sys.argv with cleaned version
    sys.argv = cleaned_argv
    logger.info(f"Cleaned sys.argv: {sys.argv}")
    
    def start_subprocess() -> None:
        mp.set_start_method("spawn", force=True)
        
        ack_queue = mp.Queue()
        
        # 1. Start Scheduler (ONE process)
        # Note: pearl_args passed
        mp.Process(
            target=run_pearl_scheduler,
            args=(pearl_args, ack_queue),
            daemon=False,
            name="pearl-scheduler"
        ).start()
        
        # 2. Start Tokenizers (Standard minisgl)
        num_tokenizers = pearl_args.num_tokenizer
        
        # Detokenizer
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": pearl_args.model_path,
                "addr": pearl_args.zmq_detokenizer_addr,
                "backend_addr": pearl_args.zmq_backend_addr,
                "frontend_addr": pearl_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": pearl_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        
        # Tokenizers
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": pearl_args.model_path, # Use target model tokenizer? Or draft? Usually target.
                    "addr": pearl_args.zmq_tokenizer_addr,
                    "backend_addr": pearl_args.zmq_backend_addr,
                    "frontend_addr": pearl_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": pearl_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()
            
        # Wait for ACKs
        # 1 scheduler + n tokenizers + 1 detokenizer
        for _ in range(num_tokenizers + 2):
            msg = ack_queue.get()
            logger.info(msg)
    
    try:        
        run_api_server(pearl_args, start_subprocess, run_shell=run_shell)
    finally:
        # Restore original argv (optional, for cleanup)
        sys.argv = original_argv

if __name__ == "__main__":
    launch_server()
