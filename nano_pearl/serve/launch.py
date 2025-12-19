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
    # 1. Parse Args
    pearl_args, run_shell = parse_args(sys.argv[1:], run_shell)
    
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
            
    run_api_server(pearl_args, start_subprocess, run_shell=run_shell)

if __name__ == "__main__":
    launch_server()
