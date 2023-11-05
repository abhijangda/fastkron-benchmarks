import torch

# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    './managed-allocator.so', 'my_malloc', 'my_free')
# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)
# This will allocate memory in the device using the new allocator
b = torch.zeros(10, device='cuda')

