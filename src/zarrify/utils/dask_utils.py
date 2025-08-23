from dask_jobqueue import LSFCluster
from dask.distributed import Client, LocalCluster
import os
import sys
import logging


def initialize_dask_client(cluster_type: str | None = None, log_dir: str = None) -> Client:
    """Initialize dask client.

    Args:
        cluster_type (str): type of the cluster, either local or lsf

    Returns:
        (Client): instance of a dask client
    """
    if cluster_type == None:
        raise ValueError("Cluster type must be specified")
    elif cluster_type == "lsf":
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
        )
    elif cluster_type == "local":
        cluster = LocalCluster()
    else:
        raise ValueError(f"Unsupported cluster type: {cluster_type}")

    client = Client(cluster)
    print(str(client.dashboard_link))
    with open(
        os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w"
    ) as text_file:
        text_file.write(str(client.dashboard_link))
    logging.info(client.dashboard_link)
    return client