from __future__ import annotations

from unittest.mock import patch

from common.llm import QwenClient, _parse_gpu_free, pick_free_gpu, shard_max_memory


# index, used MiB, total MiB -- three 80GB cards with very different headroom.
NVIDIA_SMI = "0, 54696, 81559\n1, 35809, 81559\n7, 25055, 81559\n"


def test_parse_reads_free_memory_per_device() -> None:
    free = _parse_gpu_free(NVIDIA_SMI)

    assert free == {0: 81559 - 54696, 1: 81559 - 35809, 7: 81559 - 25055}


def test_parse_skips_malformed_rows() -> None:
    assert _parse_gpu_free("0, 10, 100\ngarbage\n1, x, 100\n") == {0: 90}


def test_pick_free_gpu_still_takes_the_freest_by_default() -> None:
    with patch("common.llm._gpu_free_mib", return_value=_parse_gpu_free(NVIDIA_SMI)):
        assert pick_free_gpu() == "cuda:7"


def test_pick_free_gpu_refuses_a_device_without_the_required_headroom() -> None:
    free = _parse_gpu_free(NVIDIA_SMI)  # freest card has ~55 GB

    with patch("common.llm._gpu_free_mib", return_value=free):
        assert pick_free_gpu(required_gb=50) == "cuda:7"
        # The OOM case: the freest card fits the weights but not the run.
        assert pick_free_gpu(required_gb=70) == "cpu"


def test_pick_free_gpu_falls_back_when_nvidia_smi_is_unavailable() -> None:
    with patch("common.llm._gpu_free_mib", return_value={}):
        assert pick_free_gpu() == "cpu"


def test_shard_budget_reserves_headroom_for_other_users() -> None:
    with patch("common.llm._gpu_free_mib", return_value=_parse_gpu_free(NVIDIA_SMI)):
        budget = shard_max_memory(reserve_gb=10.0, min_free_gb=40.0)

    # Only 7 (~55 GB free) clears the bar; 0 (~26) and 1 (~45... below in this fixture) do not.
    assert 7 in budget
    assert budget[7].endswith("GiB")


def test_shard_budget_excludes_contended_cards() -> None:
    # The real failure: a card with 26 GB free is not full, but a neighbour growing into it
    # evicts our shard mid-generation. It must not be offered at all.
    free = {0: 26 * 1024, 3: 54 * 1024, 4: 55 * 1024}

    with patch("common.llm._gpu_free_mib", return_value=free):
        budget = shard_max_memory(reserve_gb=10.0, min_free_gb=40.0)

    assert set(budget) == {3, 4}


def test_shard_budget_falls_back_to_the_roomiest_card_when_none_qualify() -> None:
    with patch("common.llm._gpu_free_mib", return_value={0: 20 * 1024, 1: 30 * 1024}):
        budget = shard_max_memory(reserve_gb=10.0, min_free_gb=40.0)

    # Degrade rather than hand accelerate an empty budget.
    assert set(budget) == {1}


def test_shard_budget_is_empty_without_gpus() -> None:
    with patch("common.llm._gpu_free_mib", return_value={}):
        assert shard_max_memory() == {}


def test_shard_budget_is_keyed_by_int_index_for_from_pretrained() -> None:
    with patch("common.llm._gpu_free_mib", return_value={3: 60 * 1024}):
        budget = shard_max_memory(reserve_gb=10.0)

    assert all(isinstance(key, int) for key in budget)


def test_an_explicit_device_pins_the_model_as_before() -> None:
    client = QwenClient(device="cuda:3")

    # The pinned path must keep using .to(device) and never pass device_map.
    assert client.device == "cuda:3"
    assert client._input_device() == "cuda:3"


def test_an_unset_device_defers_placement_to_sharding() -> None:
    client = QwenClient()

    # No eager pick_free_gpu(): placement is accelerate's job at load time.
    assert client.device is None
    assert client.reserve_gb == 10.0
