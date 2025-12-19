import torch
from collections import deque
from nano_pearl.pearl_engine.sequence import Sequence
from nano_pearl.pearl_engine.radix_manager import RadixCacheManager, RadixCacheHandle


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.radix_cache = RadixCacheManager(torch.device("cpu"))
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        
        # RadixManager stores keys (token_ids).
        # We need to maintain mapping?
        # When we append tokens to a block, we need to know its content to insert to Radix later.
        # Sequence object holds token_ids. So we are good.

    def can_allocate(self, seq: Sequence) -> bool:
        # Check free blocks + evictable
        available = len(self.free_block_ids) + self.radix_cache.evictable_size // self.block_size
        return available >= seq.num_blocks

    def _allocate_blocks(self, n: int) -> list[int]:
        if len(self.free_block_ids) < n:
            to_evict_blocks = n - len(self.free_block_ids)
            # Evict from Radix
            # Radix evict size is in KEY LENGTH (tokens).
            # We assume value length == key length logic in our insert.
            # So evicting X tokens frees X indices (which are block_ids repeated).
            # So we need to evict `to_evict_blocks * self.block_size` tokens?
            # Yes, if we stored block_size repeats.
            evicted_indices = self.radix_cache.evict(to_evict_blocks * self.block_size)
            # evicted_indices is tensor of block_ids (repeated).
            # We take unique or stride.
            # Convert to list
            evicted_list = evicted_indices.tolist()
            # De-duplicate: [0, 0, ..., 1, 1, ...]
            # We can pick every block_size-th element.
            # Or simpler: set(evicted_list)? No, order might matter for debug, but generally set is fine.
            # But we might have same block_id used multiple times? No, unique blocks.
            # Using set is safest to get unique block IDs.
            evicted_blocks = list(set(evicted_list))
            self.free_block_ids.extend(evicted_blocks)
            
            if len(self.free_block_ids) < n:
                # Should not happen if math strictly holds, unless fragmentation/partial nodes issue
                # Fallback or Error
                pass

        new_blocks = []
        for _ in range(n):
            new_blocks.append(self.free_block_ids.popleft())
        return new_blocks

    def allocate(self, seq: Sequence):
        # 1. Prefix Matching
        input_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cpu")
        handle, cached_indices = self.radix_cache.match_prefix(input_ids)
        self.radix_cache.lock_handle(handle)
        
        # cached_indices contains block_ids repeated `block_size` times for full blocks.
        # We process to get block_ids.
        # Note: cached_indices might be empty.
        
        cached_block_ids = []
        if cached_indices.numel() > 0:
            flat = cached_indices.tolist()
            # We assume we only cache full blocks, so we can take stride.
            # But what if partial match?
            # Radix match returns matched length.
            # cached_indices corresponds to matched prefix.
            # If matched prefix is 300 tokens, block_size 256.
            # First 256 -> block A. Next 44 -> partial?
            # We enforced "Only Cache Full Blocks" policy in deallocate.
            # So match should be multiple of block_size?
            # Radix tree might have a node matching 300 tokens if we inserted it?
            # If we strictly check alignment:
            matched_len = handle.prefix_len
            
            # Use only full blocks
            dims = matched_len // self.block_size
            if dims > 0:
                # We take first dims * block_size indices
                valid_indices = flat[:dims * self.block_size]
                # Stride to get block ids
                cached_block_ids = valid_indices[::self.block_size]
                
                # Update handle?
                # The handle locks the node which covers `matched_len`.
                # If we only use partial, the lock still holds the whole node. Safe.
                seq._radix_handle = handle
                seq.num_cached_tokens = dims * self.block_size
            else:
                # Less than one block matched
                self.radix_cache.lock_handle(handle, unlock=True)
                seq._radix_handle = None
                seq.num_cached_tokens = 0
        else:
            self.radix_cache.lock_handle(handle, unlock=True)
            seq._radix_handle = None
            seq.num_cached_tokens = 0


        seq.block_table = cached_block_ids[:]
            
        # Allocate remaining blocks
        needed_blocks = seq.num_blocks - len(cached_block_ids)
        if needed_blocks > 0:
            new_blocks = self._allocate_blocks(needed_blocks)
            seq.block_table.extend(new_blocks)
            
            
            # ISSUE: RadixCacheManager expects key/value 1:1 mapping (token to page_mapping?).
            # In paged attention, usually value is list of pages?
            # In sglang/vllm, `value` is usually tensor of block indices matching the key?
            # No, standard block manager: Key = tokens. Value = Block Indices?
            # If I have 1 block for 16 tokens.
            # Key = 16 tokens. Value = 1 block index?
            # RadixManager `set_key_value` asserts lengths match.
            # This implies `value` in `RadixTreeNode` stores something per-token?
            # Or `RadixManager` assumes Value IS tokens?
            
            # Let's check `minisgl` usage.
            # `scheduler.py` -> calls `free_and_cache_finished_req`.
            # `indices` = `page_table[req.table_idx, : req.cached_len]`.
            # This is list of pages.
            # `input_ids` = tokens.
            # `req.cached_len` is number of blocks? Or tokens?
            # `req.cached_len` in PEARL is tokens?
            # In `minisgl`, `cached_len` seems to be number of BLOCKS?
            # Let's verify `minisgl` `req` definition.
            # `minisgl/core.py`: `class Req`.
            pass 
        
        # ... logic to evict ...
        pass
        
    def deallocate(self, seq: Sequence):
        # 1. Insert into Radix (Only Full Blocks)
        num_full_blocks = len(seq.token_ids) // self.block_size
        if num_full_blocks > 0:
            tokens_to_cache = seq.token_ids[:num_full_blocks * self.block_size]
            input_ids = torch.tensor(tokens_to_cache, dtype=torch.int64, device="cpu")
            
            blocks_to_cache = seq.block_table[:num_full_blocks]
            # Construct value tensor: repeat each block_id block_size times
            value_list = [b for b in blocks_to_cache for _ in range(self.block_size)]
            value_tensor = torch.tensor(value_list, dtype=torch.int64, device="cpu")
            
            self.radix_cache.insert_prefix(input_ids, value_tensor)

        # 2. Unlock Handle
        if hasattr(seq, '_radix_handle') and seq._radix_handle:
            self.radix_cache.lock_handle(seq._radix_handle, unlock=True)
            seq._radix_handle = None
            
        # Cleanup
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def rollback(self, seq: Sequence, n: int) -> None:
        before_num_blocks = seq.num_blocks
        seq.rollback_tokens(n)
        after_num_blocks = seq.num_blocks
        
        dropped_blocks = seq.block_table[after_num_blocks:]
        seq.block_table = seq.block_table[:after_num_blocks]
        
        # Check integrity
        if after_num_blocks * self.block_size < seq.num_cached_tokens:
             # We rolled back into cached region!
             # Update num_cached_tokens
             seq.num_cached_tokens = after_num_blocks * self.block_size
             # Dropped blocks were cached (covered by handle).
             # We assume they remain valid in Radix for other requests.
             # We don't free them.
        else:
             # Dropped blocks are strictly tail (private).
             # Free them.
             self.free_block_ids.extend(dropped_blocks)

    def can_append(self, seq: Sequence) -> bool:
        # Check if last block has space OR we can allocate 1 new block
        if len(seq) % self.block_size != 0:
            return True # Space in last block
        # Need new block
        available = len(self.free_block_ids) + self.radix_cache.evictable_size // self.block_size
        return available >= 1

    def may_append(self, seq: Sequence):
        # If last block full, allocate new one
        if len(seq.block_table) * self.block_size < len(seq):
             # Need new block
             new_blocks = self._allocate_blocks(1)
             seq.block_table.extend(new_blocks)