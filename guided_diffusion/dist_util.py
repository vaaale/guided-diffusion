"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    # comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "29400")
    rank = os.getenv("RANK", 0)
    world_size = os.getenv("WORLD_SIZE", 1)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {master_port}")
    print(f"RANK: {rank}")
    print(f"WORLD_SIZE: {world_size}")
    dist.init_process_group(backend=backend, init_method="env://")
    print("Process group initialized.")

# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
#
#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"
#
#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)
#
#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#     dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        rank = 0
        # return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
        return th.device(f"cuda:{rank}")
    return th.device("cpu")

# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     if th.cuda.is_available():
#         return th.device(f"cuda")
#     return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    # if MPI.COMM_WORLD.Get_rank() == 0:
    # if int(os.environ["RANK"]) == 0:
    #     print(f"Loading Blob from: {path}")
    #     with bf.BlobFile(path, "rb") as f:
    #         data = f.read()
    # else:
    #     data = None
    # data = MPI.COMM_WORLD.bcast(data)
    print(f"Loading Blob from: {path}")
    with bf.BlobFile(path, "rb") as f:
        data = f.read()

    return th.load(io.BytesIO(data), **kwargs)


# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     chunk_size = 2 ** 30  # MPI has a relatively small size limit
#     if MPI.COMM_WORLD.Get_rank() == 0:
#         with bf.BlobFile(path, "rb") as f:
#             data = f.read()
#         num_chunks = len(data) // chunk_size
#         if len(data) % chunk_size:
#             num_chunks += 1
#         MPI.COMM_WORLD.bcast(num_chunks)
#         for i in range(0, len(data), chunk_size):
#             MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
#     else:
#         num_chunks = MPI.COMM_WORLD.bcast(None)
#         data = bytes()
#         for _ in range(num_chunks):
#             data += MPI.COMM_WORLD.bcast(None)
#
#     return th.load(io.BytesIO(data), **kwargs)
#

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
