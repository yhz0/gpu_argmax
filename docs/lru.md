Implementing an LRU Eviction Policy for the Dual Solution Pool
This document outlines the motivation and a concrete plan for integrating a Least Recently Used (LRU) eviction policy into the ArgmaxOperation class. The goal is to intelligently manage a fixed-size pool of dual solutions, ensuring that the most relevant solutions are retained as the decomposition algorithm progresses.

1. Motivation and Goal 🧠
The ArgmaxOperation class stores a pool of dual solutions (and their corresponding bases) up to a fixed capacity, MAX_PI. The original implementation would fill this pool and then reject any new solutions.

The Problem: In decomposition algorithms like Benders or Stochastic Decomposition, the quality and relevance of dual solutions change over time.

Early Iterations: When the first-stage solution x is far from optimal, the algorithm generates many dual solutions that are only relevant to that suboptimal region.

Later Iterations: As x improves and converges, the algorithm discovers new dual solutions that are highly relevant to the optimal region.

The original "fill-and-stop" approach risks cluttering the valuable, finite pool with "stale" duals from early iterations, preventing the addition of more useful ones discovered later.

The Goal: Implement an LRU eviction policy. When the pool is full and a new dual solution is generated, the least recently used solution is evicted to make space. In this context, "used" means a dual solution was selected as the winner (the argmax) for at least one scenario in a given iteration. This ensures the pool dynamically adapts to keep the most active and relevant set of duals.

2. Summary of the Discussion
Our discussion evolved to a simple and efficient implementation strategy:

Initial Idea: We started by exploring the replacement of the simple num_pi counter with a more complex system involving a free_indices list to track available slots.

While filling: The next available index is always the current size of the pool (len(self.lru_manager)).

When full: The cachetools.LRU.popitem() method not only removes the LRU item from its tracking but also tells us the index of the slot that was just freed.

This insight led to the "evict and overwrite" strategy. We can immediately reuse the index returned by popitem() for the new data. This is highly efficient as it avoids managing a separate list of free slots and simply overwrites data in place.

3. Final Implementation Plan
This plan requires modifying the class __init__ method, refactoring add_pi, and adding a new helper method to update the usage status of items.

Part A: Key Components in __init__
The constructor will be updated to include the new management tools.

Add:

self.lru_manager = LRUCache(maxsize=self.MAX_PI): The core cache mapping a basis hash to its storage index.

self.index_to_hash_map = {}: A reverse map to find a hash from a storage index, needed for updates.

Remove:

self.hashes: No longer needed. Use if hash in self.lru_manager.

Part B: The add_pi() Workflow
This method will contain the primary logic for adding, evicting, and overwriting a dual solution. The workflow proceeds in these steps:

Pre-computation: Perform initial data validation and calculate a unique hash from the incoming basis vectors.

Duplicate Handling: Within a thread-safe lock, check if the basis hash already exists in the lru_manager. If it does, "touch" the item to mark it as recently used and exit, as no new item needs to be added.

Index Determination:

If the pool is full (len(self.lru_manager) >= self.MAX_PI), evict the least recently used item using lru_manager.popitem(). This action frees up a storage index. Clean up the index_to_hash_map for the evicted item.

If the pool is not full, the storage index will be the next sequential number, determined by the current size of the pool.

Data Storage: Use the determined storage index to write (or overwrite) all data associated with the new dual solution—such as pi, rc, basis vectors, and pre-computed LU factors—into their respective pre-allocated tensors.

Record Keeping: Update the lru_manager and index_to_hash_map with the new item's hash and its storage index.

Part C: The update_lru_on_access() Workflow
This new method is responsible for informing the cache which items have been "used" in an iteration. It should be called immediately after find_best_k. Modify find_best_k by setting a flag that enables automatic refreshing.

Identify Used Indices: The method receives the tensor of winning indices from the find_best_k operation. It identifies the unique set of indices that were used.

Update Cache Status: Within a thread-safe lock, it iterates through these unique indices. For each index, it looks up the corresponding hash in the index_to_hash_map and "touches" that item in the lru_manager. This access marks the item as most recently used, ensuring it is not a candidate for early eviction.