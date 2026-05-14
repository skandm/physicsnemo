import torch
import torch.distributed as dist
import os

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

backend = os.environ.get('DIST_BACKEND', 'nccl')
dist.init_process_group(backend)
torch.cuda.set_device(local_rank)

t = torch.ones(1000, 1000, device='cuda')
dist.all_reduce(t)
print(f'Rank {rank}: all_reduce done, mean={t.mean().item():.1f}')
dist.destroy_process_group()
