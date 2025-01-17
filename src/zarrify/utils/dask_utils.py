from dask_jobqueue import LSFCluster
from dask.distributed import Client, LocalCluster
import os
import sys


def initialize_dask_client(cluster_type: str) -> Client:
    """Initialize dask client.

    Args:
        cluster_type (str): type of the cluster, either local or lsf

    Returns:
        (Client): instance of a dask client
    """
    if cluster_type == "":
        print("Did not specify which instance of the dask client to use!")
        sys.exit(0)
    elif cluster_type == "lsf":
        num_cores = 1
        cluster = LSFCluster(
            cores=num_cores,
            processes=num_cores,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="48:00",
            local_directory="/scratch/$USER/",
        )
    elif cluster_type == "local":
        cluster = LocalCluster()

    client = Client(cluster)
    with open(
        os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w"
    ) as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)
    return client
