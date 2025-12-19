from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from transformers import AutoTokenizer, LlamaTokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    # Log ZMQ endpoint information for debugging connectivity
    logger.info(
        "Tokenizer startup: tokenizer_id=%s addr=%s (create=%s) backend_addr=%s frontend_addr=%s",
        tokenizer_id,
        addr,
        create,
        backend_addr,
        frontend_addr,
    )

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            # Info-level logs to help trace incoming requests and UIDs
            uids = [m.uid for m in pending_msg]
            logger.info(f"Tokenizer received {len(pending_msg)} messages uids={uids}")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) == len(pending_msg)
            if len(detokenize_msg) > 0:
                logger.info(f"Tokenizer: processing {len(detokenize_msg)} detokenize messages")
                replies = detokenize_manager.detokenize(detokenize_msg)
                logger.info(f"Tokenizer: detokenized {len(replies)} replies")
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                logger.info(f"Tokenizer -> frontend put {len(batch_output.data if isinstance(batch_output, BatchFrontendMsg) else [batch_output])} replies for uids={[r.uid for r in batch_output.data] if isinstance(batch_output, BatchFrontendMsg) else [batch_output.uid]}")
                send_frontend.put(batch_output)
                logger.info("Tokenizer: sent replies to frontend")

            if len(tokenize_msg) > 0:
                tensors = tokenize_manager.tokenize(tokenize_msg)
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                logger.info(f"Tokenizer -> backend put {len(batch_output.data if isinstance(batch_output, BatchBackendMsg) else [batch_output])} msgs for uids={[m.uid for m in (batch_output.data if isinstance(batch_output, BatchBackendMsg) else [batch_output]) ]}")
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
