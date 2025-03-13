import os
import platform
import socket
from dataclasses import dataclass

import numpy as np

nn_eph_config = {"use_gpu": False}


class not_a_comm:
    def __init__(self):
        self.size = 1
        self.rank = 0

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def Barrier(self):
        pass

    def barrier(self):
        pass

    def Reduce(self, sendbuf, recbuf, op=None, root=0):
        np.copyto(recbuf[0], sendbuf[0])

    def reduce(self, buf, op=None, root=0):
        return buf

    def Bcast(self, buf, root=0):
        pass

    def bcast(self, buf, root=0):
        return buf

    def Gather(self, sendbuf, recbuf, root=0):
        recbuf[:] = sendbuf

    def Scatter(self, sendbuf, recbuf, root=0):
        recbuf[:] = sendbuf


@dataclass
class not_MPI:
    FLOAT = None
    DOUBLE = None
    INT = None
    SUM = None
    COMM_WORLD = not_a_comm()


def setup_jax():
    from jax import config

    # breaking change in random number generation in jax v0.5
    config.update("jax_threefry_partitionable", False)

    if nn_eph_config["use_gpu"] == True:
        config.update("jax_platform_name", "gpu")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        # TODO: add gpu performance xla flags
        hostname = socket.gethostname()
        system_type = platform.system()
        machine_type = platform.machine()
        processor = platform.processor()
        print(f"# Hostname: {hostname}")
        print(f"# System Type: {system_type}")
        print(f"# Machine Type: {machine_type}")
        print(f"# Processor: {processor}")
        uname_info = platform.uname()
        print("# Using GPU.")
        print(f"# System: {uname_info.system}")
        print(f"# Node Name: {uname_info.node}")
        print(f"# Release: {uname_info.release}")
        print(f"# Version: {uname_info.version}")
        print(f"# Machine: {uname_info.machine}")
        print(f"# Processor: {uname_info.processor}")
    else:
        nn_eph_config["use_gpu"] = False
        config.update("jax_platform_name", "cpu")
        os.environ["XLA_FLAGS"] = (
            "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        )


def setup_comm():
    if "use_mpi" not in nn_eph_config:
        if nn_eph_config["use_gpu"] == True:
            nn_eph_config["use_mpi"] = False
        else:
            nn_eph_config["use_mpi"] = True
    if nn_eph_config["use_mpi"] == True:
        from mpi4py import MPI
    else:
        MPI = not_MPI()
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0 and nn_eph_config["use_gpu"] == False:
        hostname = socket.gethostname()
        system_type = platform.system()
        machine_type = platform.machine()
        processor = platform.processor()
        print(f"# Hostname: {hostname}")
        print(f"# System Type: {system_type}")
        print(f"# Machine Type: {machine_type}")
        print(f"# Processor: {processor}")
    return MPI
