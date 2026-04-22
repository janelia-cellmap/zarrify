from dask_jobqueue import LSFCluster
from dask.distributed import Client, LocalCluster
import os
import sys
import logging
import dask
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def initialize_dask_client(cluster_type: str | None = None, log_dir: str = None, job_extra_directives: tuple[str, ...] = None) -> Client:
    """Initialize dask client.

    Args:
        cluster_type (str): type of the cluster, either local or lsf
        log_dir (str): directory for LSF worker logs
        job_extra_directives (tuple[str, ...]): additional LSF job directives

    Returns:
        (Client): instance of a dask client
    """
    if cluster_type == None:
        raise ValueError("Cluster type must be specified")
    elif cluster_type == "lsf":
        dask.config.set({"jobqueue.lsf.cancel-command": "bkill -d"})
        if job_extra_directives is None:
            job_extra_directives = []
        else:
            job_extra_directives = list(job_extra_directives)
        num_cores = 1
        cluster = LSFCluster(
            cores=num_cores,
            processes=num_cores,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="48:00",
            log_directory = log_dir,
            local_directory="/scratch/$USER/",
            job_extra_directives=job_extra_directives
        )
    elif cluster_type == "local":
        cluster = LocalCluster()
    else:
        raise ValueError(f"Unsupported cluster type: {cluster_type}")

    client = Client(cluster)

    logger.info(f"Dask dashboard link: {client.dashboard_link}")
    with open(
        os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w"
    ) as text_file:
        text_file.write(str(client.dashboard_link))

    return client


def check_shardslab_fits_in_ram(
    shard_shape: list[int],
    dtype: np.dtype,
    chunk_shape: list[int],
    client: Client,
) -> None:
    """Raise ValueError if one shard does not fit in a single Dask worker's RAM.

    Proposes a reduced shard_shape (axis-0 extent trimmed to fit) in the error message.
    Logs and skips silently if worker memory cannot be queried.
    """
    dtype_bytes = np.dtype(dtype).itemsize
    shard_bytes = int(np.prod(shard_shape)) * dtype_bytes

    try:
        client.wait_for_workers(n_workers=1, timeout=60)
        workers_info = client.scheduler_info()["workers"]
        worker_memory_bytes = next(iter(workers_info.values()))["memory_limit"]
    except Exception as e:
        logger.warning(f"Could not query worker memory limit: {e}. Skipping shard RAM check.")
        return

    logger.info(
        f"Shard size: {shard_bytes / 1e9:.2f} GB, "
        f"worker RAM: {worker_memory_bytes / 1e9:.2f} GB"
    )
    if shard_bytes <= worker_memory_bytes:
        return

    # Propose a fitting shard by reducing axis-0 (Z) to what fits in RAM
    xy_bytes = int(np.prod(shard_shape[1:])) * dtype_bytes
    max_z = int(worker_memory_bytes // xy_bytes)
    chunk_z = chunk_shape[0] if chunk_shape else 1
    proposed_z = max(chunk_z, (max_z // chunk_z) * chunk_z)
    proposed = [proposed_z] + list(shard_shape[1:])

    raise ValueError(
        f"Shard does not fit in worker RAM: "
        f"shard_shape={shard_shape}, dtype={np.dtype(dtype)}, "
        f"shard size={shard_bytes / 1e9:.2f} GB > worker limit={worker_memory_bytes / 1e9:.2f} GB. "
        f"Proposed shard_shape: {proposed}"
    )
