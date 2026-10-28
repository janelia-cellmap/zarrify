import numpy as np
import pytest

from zarrify.utils.dask_utils import check_shardslab_fits_in_ram

GiB = 1024 ** 3


class FakeClient:
    """Minimal stand-in for dask Client: one worker with a fixed memory limit and thread count."""
    def __init__(self, memory_limit, nthreads):
        self._info = {"workers": {"w0": {"memory_limit": memory_limit, "nthreads": nthreads}}}

    def wait_for_workers(self, n_workers, timeout):
        pass

    def scheduler_info(self):
        return self._info


def test_shard_within_per_thread_budget_passes():
    # 512^3 uint8 = 128 MiB; budget = 4 GiB / 4 threads = 1 GiB
    check_shardslab_fits_in_ram([512, 512, 512], np.uint8, [128, 128, 128], FakeClient(4 * GiB, 4))


def test_shard_over_per_thread_budget_raises():
    # 1024^3 uint8 = 1 GiB fits in the worker's 4 GiB, but not in 4 GiB / 8 threads = 512 MiB
    with pytest.raises(ValueError, match="per-thread RAM budget") as exc:
        check_shardslab_fits_in_ram([1024, 1024, 1024], np.uint8, [128, 128, 128], FakeClient(4 * GiB, 8))
    assert "Proposed shard_shape: [512, 1024, 1024]" in str(exc.value)


def test_missing_nthreads_uses_whole_worker_memory():
    # 1 GiB shard, 2 GiB worker, no nthreads reported -> treated as 1 thread
    client = FakeClient(2 * GiB, None)
    check_shardslab_fits_in_ram([1024, 1024, 1024], np.uint8, [128, 128, 128], client)
