from __future__ import annotations

import multiprocessing as mp
import time
import pickle
import asyncio
import queue
import torch.distributed as dist

from minisgl.message import (
    BatchBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg
)
from minisgl.server.args import ServerArgs
from minisgl.utils import init_logger, ZmqAsyncPullQueue, ZmqAsyncPushQueue
from minisgl.scheduler.io import SchedulerIOMixin
from minisgl.core import SamplingParams as MiniSamplingParams

from nano_pearl.pearl_config import PEARLConfig
from nano_pearl.pearl_engine.pearl_engine import PEARLEngine
from nano_pearl.layers.sampler import SamplingParams as PearlSamplingParams
from nano_pearl.serve.args import PearlServerArgs

logger = init_logger(__name__, "PearlScheduler")

class PearlSchedulerWrapper:
    def __init__(self, args: PearlServerArgs):
        self.args = args
        
        # Initialize PEARL Config
        self.pearl_config = PEARLConfig(
            draft_model_path=args.draft_model_path,
            target_model_path=args.model_path,
            draft_tensor_parallel_size=args.draft_tp_size,
            target_tensor_parallel_size=args.target_tp_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_running_req,
            gamma=args.gamma,
            # Others use defaults or map from args if needed
        )
        
        # Initialize Engine
        self.engine = PEARLEngine(self.pearl_config)
        
        # IO
        # We manually initialize IO mixin components or just use ZMQ directly
        # because SchedulerIOMixin assumes 'self.config' matches SchedulerConfig + initialization logic
        # We'll just define queues here for clarity.
        
        self.recv_from_tokenizer = ZmqAsyncPullQueue(
            args.zmq_backend_addr,
            create=True, # Backend creates the PULL socket
            decoder=BatchBackendMsg.decoder
        )
        
        self.send_to_detokenizer = ZmqAsyncPushQueue(
            args.zmq_detokenizer_addr,
            create=False,
            encoder=BatchTokenizerMsg.encoder
        )
        
        # Mappings
        self.req_map = {} # uid -> sampling_params (if needed?)
        
    def run_forever(self):
        logger.info("PearlScheduler running...")
        
        while True:
            # 1. Receive Requests
            try:
                # We want non-blocking if we have work doing, blocking if idle?
                # PEARLEngine.step updates state.
                # If we have active requests in engine, we should not block long.
                # But actually PEARLEngine.step takes time.
                
                # We check for new messages.
                # ZmqAsyncPullQueue.get_batch() is async/await? No, it's Sync wrapper?
                # minisgl implementation uses `get()` which is async, or `get_nowait`.
                # Wait, minisgl utils generic queue:
                # `class ZmqAsyncPullQueue`: `async def get(self)`
                
                # Use `asyncio.run` for the loop?
                # Or use the synchronous version if available?
                # minisgl ZmqQueue seems to be Async.
                
                # But `PearlScheduler` is running in a Process which is `target=_run_pearl_scheduler`.
                # If we want to use PEARL synchronous logic, we need to bridge async IO.
                # `minisgl`'s Scheduler uses `receive_msg` generator which wraps async get.
                
                # Let's inspect `minisgl/utils.py` for queue implementation.
                pass
            except Exception as e:
                logger.error(f"Error in loop: {e}")
                
            # To simplify, we run an asyncio loop for IO?
            # But `engine.step()` is blocking sync code.
            # Mixing them is tricky.
            
            # Simple approach: Use asyncio.run(self.io_step()) + engine.step() inside a generic loop?
            # Or just use zmq directly in blocking/non-blocking mode without async wrapper?
            
            # minisgl queue is ZmqAsyncPullQueue.
            # I'll implement a helper to fetch batch synchronously with timeout.
            
            pass

    async def main_loop(self):
        while True:
            # 1. Receive new requests
            # If we assume continuous batching, we check for new reqs every step.
            # zmq queue `get_batch` might wait?
            
            # We construct a batch of messages.
            msgs = []
            
            # Try to get messages. If engine is idle, we wait. If busy, we don't wait (or small wait).
            # How to check if engine is idle?
            # We don't have direct access to engine's internal queue unless we added `finished` check.
            # BUT we control input.
            
            # We can rely on `ZmqAsyncPullQueue`'s behaviour.
            # `minisgl` scheduler uses `blocking = not (self.prefill_manager.runnable ...)`
            
            # Here:
            # blocking = True # for now
            
            # But we want to run engine step.
            
            # We'll use a timeout of 0 if we have work.
            
            # Actually, `minisgl` uses `receive_msg` which yields messages.
            
            # Implementation:
            # Loop:
            #   Process incoming messages (add to engine)
            #   Run engine step
            #   Process output
            
            # Incoming
            try:
                # We need to peek or get with timeout.
                # minisgl Queue doesn't support timeout easily on `get`.
                # But `ZmqAsyncPullQueue` has `_queue` (asyncio.Queue).
                
                # If use asyncio, we can use `asyncio.wait_for`.
                pass
            except:
                pass

def run_pearl_scheduler(args: PearlServerArgs, ack_queue: mp.Queue):
    scheduler = PearlScheduler(args)
    if ack_queue:
        ack_queue.put("Scheduler Ready")
    scheduler.run()

class PearlScheduler:
    def __init__(self, args: PearlServerArgs):
        self.args = args
        self.pearl_config = PEARLConfig(
            draft_model_path=args.draft_model_path,
            target_model_path=args.model_path,
            draft_tensor_parallel_size=args.draft_tp_size,
            target_tensor_parallel_size=args.target_tp_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_running_req,
            gamma=args.gamma,
        )
        self.engine = PEARLEngine(self.pearl_config)
        self.context = None # ZMQ context

    def run(self):
        # We implement a synchronous loop using ZMQ directly to avoid async complexity 
        # mixing with PEARL sync engine, if possible.
        import zmq
        
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind(self.args.zmq_backend_addr)
        
        sender = context.socket(zmq.PUSH)
        # We connect to Detokenizer.
        # Note: Detokenizer binds or connects?
        # `ServerArgs.zmq_detokenizer_addr` is usually `ipc://...`.
        # minisgl `tokenize_worker` binds/connects depending on config.
        # `launch.py` uses `SchedulerIOMixin`.
        # `SchedulerIOMixin` uses `ZmqAsyncPushQueue` with `create=False` -> connects?
        # `ZmqAsyncPushQueue` docs: `create=True` -> bind. `create=False` -> connect.
        # minisgl scheduler `send_to_detokenizer` uses `create=False`.
        # So Scheduler CONNECTS to Detokenizer.
        # Detokenizer (tokenizer_worker) Binds.
        
        sender.connect(self.args.zmq_detokenizer_addr)
        
        poller = zmq.Poller()
        poller.register(receiver, zmq.POLLIN)
        
        logger.info(f"PearlScheduler listening on {self.args.zmq_backend_addr}")
        
        has_running_reqs = False
        
        while True:
            # Poll with timeout
            # If has_running_reqs, timeout=0 (non-blocking).
            # Else timeout=None (blocking).
            
            timeout = 0 if has_running_reqs else None
            socks = dict(poller.poll(timeout))
            
            if receiver in socks and socks[receiver] == zmq.POLLIN:
                # Receive batch
                # Loop to get multiple?
                # Just get one batch msg.
                
                # ZMQ Recv
                raw_msg = receiver.recv()
                # Decode
                msg = BatchBackendMsg.decoder(raw_msg)
                
                if isinstance(msg, ExitMsg):
                    break
                
                if isinstance(msg, BatchBackendMsg):
                    for sub_msg in msg.data:
                        if isinstance(sub_msg, UserMsg):
                            # Convert to PEARL sequence
                            # sub_msg.input_ids (list[int])
                            # sub_msg.sampling_params (minisgl.core.SamplingParams)
                            
                            # Convert Sampling Params
                            p = PearlSamplingParams(
                                n=1, # always 1 for now
                                temperature=sub_msg.sampling_params.temperature,
                                max_tokens=sub_msg.sampling_params.max_tokens,
                                ignore_eos=sub_msg.sampling_params.ignore_eos,
                            )
                            
                            sid = self.engine.add_request(sub_msg.input_ids, p)
                            # We might need to map sid to uid if PEARL generates its own sid?
                            # PEARL Sequence generates its own ID.
                            # We need to map it back to `sub_msg.uid`.
                            
                            if not hasattr(self, 'sid_to_uid'):
                                self.sid_to_uid = {}
                            self.sid_to_uid[sid] = sub_msg.uid
                            
                            has_running_reqs = True
            
            # Run Step if running
            if has_running_reqs:
                output_list = self.engine.step()
                # output_list: [(seq_id, new_tokens, finished)]
                
                if output_list:
                    # Prepare BatchTokenizerMsg
                    replies = []
                    for sid, tokens, finished in output_list:
                        uid = self.sid_to_uid.get(sid)
                        if uid is not None:
                            # Send one DetokenizeMsg PER TOKEN?
                            # Frontend expects streaming per token?
                            # `DetokenizeMsg` has `next_token` (int).
                            # So if we have multiple tokens, we send MULTIPLE messages? 
                            # Or `BatchTokenizerMsg` with multiple entries.
                            
                            for t in tokens:
                                # We treat intermediate tokens as not finished?
                                # Only last token uses `finished` flag?
                                # Actually `finished` flag in `DetokenizeMsg` signals end of stream.
                                
                                # If `tokens` has 5 tokens, and `finished=True`.
                                # We send 4 with finished=False, 1 with finished=True?
                                # Correct.
                                
                                pass
                            
                            for i, t in enumerate(tokens):
                                is_last_token = (i == len(tokens) - 1)
                                is_actually_finished = finished and is_last_token
                                
                                replies.append(DetokenizeMsg(
                                    uid=uid,
                                    next_token=t,
                                    finished=is_actually_finished
                                ))
                                
                            if finished:
                                del self.sid_to_uid[sid]
                    
                    if replies:
                        # Send batch
                        batch_reply = BatchTokenizerMsg(data=replies)
                        sender.send(BatchTokenizerMsg.encoder(batch_reply))
                else:
                    # No output? verify if finished
                    # If empty output but requests are running, it means they are still processing (gamma steps).
                    pass
                
                # Check if we should stop loop (if we can detect emptiness)
                # PEARL doesn't expose `is_finished()` nicely on Engine.
                # But we clean up `sid_to_uid`.
                if not self.sid_to_uid:
                    has_running_reqs = False

        self.engine.exit()
