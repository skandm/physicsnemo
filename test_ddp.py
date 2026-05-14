import torch
import torch.distributed as dist
import os

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group('nccl')
torch.cuda.set_device(local_rank)
t = torch.ones(1000, 1000, device='cuda')
dist.all_reduce(t)
print(f'Rank {rank}: OK, mean={t.mean():.1f}')
dist.destroy_process_group()
