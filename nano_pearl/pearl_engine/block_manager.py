import torch
from collections import deque
import xxhash
import numpy as np

from nano_pearl.pearl_engine.sequence import Sequence
from nano_pearl.pearl_engine.radix_manager import RadixCacheManager, RadixCacheHandle


class Block:
    """用於 Hash-based 模式的 Block 類別"""
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    支援兩種 KV Cache 管理模式：
    1. Hash-based: 使用 xxhash 進行前綴比對（原始實作）
    2. Radix Tree: 使用 Radix Tree 進行高效前綴共享（新實作）
    
    透過 use_radix_cache 參數控制使用哪種模式
    """
    
    def __init__(self, num_blocks: int, block_size: int, use_radix_cache: bool = True):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.use_radix_cache = use_radix_cache
        
        if use_radix_cache:
            # Radix Tree 模式
            self.radix_cache = RadixCacheManager(torch.device("cpu"))
            self.free_block_ids: deque[int] = deque(range(num_blocks))
        else:
            # Hash-based 模式
            self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
            self.hash_to_block_id: dict[int, int] = dict()
            self.free_block_ids: deque[int] = deque(range(num_blocks))
            self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """Hash-based 模式的雜湊計算"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def can_allocate(self, seq: Sequence) -> bool:
        if self.use_radix_cache:
            available = len(self.free_block_ids) + self.radix_cache.evictable_size // self.block_size
            return available >= seq.num_blocks
        else:
            return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        if self.use_radix_cache:
            self._allocate_radix(seq)
        else:
            self._allocate_hash(seq)
    
    def _allocate_radix(self, seq: Sequence):
        """Radix Tree 模式的配置"""
        # 1. 前綴比對
        input_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cpu")
        handle, cached_indices = self.radix_cache.match_prefix(input_ids)
        self.radix_cache.lock_handle(handle)
        
        # 2. 提取已快取的 Block IDs
        cached_block_ids = []
        if cached_indices.numel() > 0:
            flat = cached_indices.tolist()
            matched_len = handle.cached_len
            dims = matched_len // self.block_size
            
            if dims > 0:
                valid_indices = flat[:dims * self.block_size]
                cached_block_ids = valid_indices[::self.block_size]
                seq._radix_handle = handle
                seq.num_cached_tokens = dims * self.block_size
            else:
                self.radix_cache.lock_handle(handle, unlock=True)
                seq._radix_handle = None
                seq.num_cached_tokens = 0
        else:
            self.radix_cache.lock_handle(handle, unlock=True)
            seq._radix_handle = None
            seq.num_cached_tokens = 0

        seq.block_table = cached_block_ids[:]
        
        # 3. 配置剩餘 Blocks
        needed_blocks = seq.num_blocks - len(cached_block_ids)
        if needed_blocks > 0:
            new_blocks = self._allocate_blocks_radix(needed_blocks)
            seq.block_table.extend(new_blocks)
    
    def _allocate_hash(self, seq: Sequence):
        """Hash-based 模式的配置"""
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
                
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block_hash(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block_hash(block_id)
                    
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def _allocate_blocks_radix(self, n: int) -> list[int]:
        """Radix 模式的 Block 配置（支援驅逐）"""
        if len(self.free_block_ids) < n:
            to_evict_blocks = n - len(self.free_block_ids)
            evicted_indices = self.radix_cache.evict(to_evict_blocks * self.block_size)
            evicted_blocks = list(set(evicted_indices.tolist()))
            self.free_block_ids.extend(evicted_blocks)

        new_blocks = []
        for _ in range(n):
            new_blocks.append(self.free_block_ids.popleft())
        return new_blocks

    def _allocate_block_hash(self, block_id: int) -> Block:
        """Hash 模式的單一 Block 配置"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block_hash(self, block_id: int) -> Block:
        """Hash 模式的單一 Block 釋放"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def deallocate(self, seq: Sequence):
        if self.use_radix_cache:
            self._deallocate_radix(seq)
        else:
            self._deallocate_hash(seq)
    
    def _deallocate_radix(self, seq: Sequence):
        """Radix 模式的釋放（插入到快取）"""
        # 1. 插入完整 Blocks 到 Radix Tree
        num_full_blocks = len(seq.token_ids) // self.block_size
        if num_full_blocks > 0:
            tokens_to_cache = seq.token_ids[:num_full_blocks * self.block_size]
            input_ids = torch.tensor(tokens_to_cache, dtype=torch.int64, device="cpu")
            
            blocks_to_cache = seq.block_table[:num_full_blocks]
            value_list = [b for b in blocks_to_cache for _ in range(self.block_size)]
            value_tensor = torch.tensor(value_list, dtype=torch.int64, device="cpu")
            
            self.radix_cache.insert_prefix(input_ids, value_tensor)

        # 2. 解鎖 Handle
        if hasattr(seq, '_radix_handle') and seq._radix_handle:
            self.radix_cache.lock_handle(seq._radix_handle, unlock=True)
            seq._radix_handle = None
        
        seq.block_table.clear()
        seq.num_cached_tokens = 0
    
    def _deallocate_hash(self, seq: Sequence):
        """Hash 模式的釋放（減少引用計數）"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block_hash(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def rollback(self, seq: Sequence, n: int) -> None:
        if self.use_radix_cache:
            self._rollback_radix(seq, n)
        else:
            self._rollback_hash(seq, n)
    
    def _rollback_radix(self, seq: Sequence, n: int) -> None:
        """Radix 模式的 Rollback"""
        before_num_blocks = seq.num_blocks
        seq.rollback_tokens(n)
        after_num_blocks = seq.num_blocks
        
        dropped_blocks = seq.block_table[after_num_blocks:]
        seq.block_table = seq.block_table[:after_num_blocks]
        
        if after_num_blocks * self.block_size < seq.num_cached_tokens:
            seq.num_cached_tokens = after_num_blocks * self.block_size
        else:
            self.free_block_ids.extend(dropped_blocks)
    
    def _rollback_hash(self, seq: Sequence, n: int) -> None:
        """Hash 模式的 Rollback"""
        block_table = seq.block_table
        before_num_blocks = seq.num_blocks
        seq.rollback_tokens(n)
        after_num_blocks = seq.num_blocks
        
        if before_num_blocks == after_num_blocks:
            return
            
        for block_id in block_table[after_num_blocks:]:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block_hash(block_id)
        seq.block_table = seq.block_table[:after_num_blocks]

    def can_append(self, seq: Sequence) -> bool:
        if self.use_radix_cache:
            if len(seq) % self.block_size != 0:
                return True
            available = len(self.free_block_ids) + self.radix_cache.evictable_size // self.block_size
            return available >= 1
        else:
            return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if self.use_radix_cache:
            self._may_append_radix(seq)
        else:
            self._may_append_hash(seq)
    
    def _may_append_radix(self, seq: Sequence):
        """Radix 模式的動態擴展"""
        if len(seq.block_table) * self.block_size < len(seq):
            new_blocks = self._allocate_blocks_radix(1)
            seq.block_table.extend(new_blocks)
    
    def _may_append_hash(self, seq: Sequence):
        """Hash 模式的動態擴展"""
        block_table = seq.block_table
        required_blocks = seq.num_blocks
        current_blocks = len(block_table)
        
        if required_blocks > current_blocks:
            assert required_blocks == current_blocks + 1
            block_id = self.free_block_ids[0]
            self._allocate_block_hash(block_id)
            block_table.append(block_id)

            if self.blocks[block_table[-2]].hash == -1:
                token_ids = seq.block(seq.num_blocks-2)
                prefix = self.blocks[block_table[-3]].hash if len(block_table) > 2 else -1
                h = self.compute_hash(token_ids, prefix)
                self.blocks[block_table[-2]].update(h, token_ids)
                self.hash_to_block_id[h] = block_table[-2]
        else:
            if seq.last_block_num_tokens == self.block_size:
                token_ids = seq.block(seq.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                self.blocks[block_table[-1]].update(h, token_ids)
                self.hash_to_block_id[h] = block_table[-1]
    
    def clear(self):
        """清理快取狀態（用於 Scheduler.clear）"""
        if not self.use_radix_cache:
            self.hash_to_block_id.clear()
            for block in self.blocks:
                block.hash = -1
                block.token_ids = []