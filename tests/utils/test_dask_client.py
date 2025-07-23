import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
from zarrify.utils.dask_utils import initialize_dask_client

@pytest.mark.parametrize("cluster_type, cluster_class_name", [
    ("local", "LocalCluster"),
    ("lsf", "LSFCluster"),
])
@patch("zarrify.utils.dask_utils.Client")
def test_initialize_dask_client_valid_types(
    mock_client, cluster_type, cluster_class_name
):
    mock_dashboard_link = "http://localhost:8787/status"
    mock_client.return_value.dashboard_link = mock_dashboard_link

    with patch(f"zarrify.utils.dask_utils.{cluster_class_name}") as mock_cluster_class:
        result = initialize_dask_client(cluster_type)

        mock_cluster_class.assert_called_once()
        mock_client.assert_called_once_with(mock_cluster_class.return_value)
        assert result == mock_client.return_value
        
@pytest.mark.parametrize("invalid_cluster_type, expected_msg", [
    (None, "Cluster type must be specified"),
    ("unknown", "Unsupported cluster type: unknown"),
])
def test_initialize_dask_client_invalid_inputs(invalid_cluster_type, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        initialize_dask_client(invalid_cluster_type)