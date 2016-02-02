#include <map>

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>

/*! We use thrust's sort function to reorder the particles.
 * Thrust's default behavior is to allocate and free any temporary storage
 * needed by the sort, which leads to repeated allocations and frees (one per iteration).
 * We want to avoid this, so we provide a custom allocator that caches allocations
 * and presents the same memory area every time the same amount is required.
 * This is based off the custom_temporary_allocator example in thrust
 */

#include "cuda_call.h"

#define CACHED_ALLOC_DEBUG 0 // set to 1 to see cached allocation behavior

class cached_allocator {
	/*! Map between sizes and (allocated and then released) pointers. */
	// The free blocks map is a multimap because we might have multiple
	// free blocks with the same size
	typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
	// The used block map is a simple map because it maps each block (pointer)
	// to the allocated size, so each key can only appear once.
	typedef std::map<char *, std::ptrdiff_t> used_blocks_type;

	free_blocks_type free_blocks;
	used_blocks_type used_blocks;

	void cleanup() {
#if CACHED_ALLOC_DEBUG
		std::clog << "cached_allocator cleanup" << std::endl;
#endif

		// deallocate free blocks
		for (free_blocks_type::iterator i = free_blocks.begin();
			i != free_blocks.end(); ++i) {
			cudaFree(i->second);
		}

		// deallocated nonfree blocks
		for (used_blocks_type::iterator i = used_blocks.begin();
			i != used_blocks.end(); ++i) {
			cudaFree(i->first);
		}

		free_blocks.clear();
		used_blocks.clear();
	}

public:
	// We allocate in bytes
	typedef char value_type;

	// nothing to do on construction
	cached_allocator() {}
	// clean up on destruction
	~cached_allocator() { cleanup(); }

	// allocation function: checks if we have a free block of the given size
	// before trying to allocate a new one
	char* allocate(std::ptrdiff_t num_bytes)
	{
		char *block = NULL;

		// first, check if we have a free block allocated already
		free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

		if (free_block != free_blocks.end()) {
			// Got a hit! Reuse

			block = free_block->second;
			free_blocks.erase(free_block);

#if CACHED_ALLOC_DEBUG
			std::clog << "Reusing block @" << (void*)block << " for " << num_bytes << " bytes requested" << std::endl;
#endif

		} else {
			// No block of the given size, allocate a new one
			CUDA_SAFE_CALL(cudaMalloc(&block, num_bytes));
			// TODO FIXME if alloc fails, retry after freeing some cached data

#if CACHED_ALLOC_DEBUG
			std::clog << "NEW block @" << (void*)block << " for " << num_bytes << " bytes requested" << std::endl;
#endif
		}

		used_blocks.insert(std::make_pair(block, num_bytes));

		return block;
	}

	// deallocate: move from used to free block array
	void deallocate(char *ptr, size_t n)
	{
		used_blocks_type::iterator iter = used_blocks.find(ptr);
		if (iter == used_blocks.end()) {
			throw std::invalid_argument("trying to deallocated non-cached block");
		}

		std::ptrdiff_t num_bytes = iter->second;
		used_blocks.erase(iter);
		free_blocks.insert(std::make_pair(num_bytes, ptr));

#if CACHED_ALLOC_DEBUG
		std::clog << "Block @" << (void*)ptr << " of size " << num_bytes << " cached after deallocation" << std::endl;
#endif
	}

};

