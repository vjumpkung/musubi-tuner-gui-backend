from types import SimpleNamespace

from app import system_monitor


def test_resource_snapshot_without_nvidia_gpu(client, monkeypatch):
    monkeypatch.setattr(system_monitor.psutil, "cpu_percent", lambda interval: 37.5)
    monkeypatch.setattr(
        system_monitor.psutil,
        "cpu_count",
        lambda logical: 16 if logical else 8,
    )
    monkeypatch.setattr(
        system_monitor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            percent=62.5,
            used=10 * system_monitor.MIB,
            available=6 * system_monitor.MIB,
            total=16 * system_monitor.MIB,
        ),
    )
    monkeypatch.setattr(system_monitor.shutil, "which", lambda name: None)

    response = client.get("/api/system/resources")

    assert response.status_code == 200
    body = response.json()
    assert body["cpu"] == {
        "percent": 37.5,
        "logical_cores": 16,
        "physical_cores": 8,
    }
    assert body["ram"]["percent"] == 62.5
    assert body["ram"]["total_bytes"] == 16 * system_monitor.MIB
    assert body["gpus"] == []
    assert "nvidia-smi" in body["gpu_error"]


def test_parse_nvidia_smi_multiple_gpus():
    gpus = system_monitor._parse_nvidia_smi(
        "0, NVIDIA RTX 4090, 87, 12288, 24564, 71\n"
        "1, NVIDIA RTX 4060, N/A, 512, 8188, N/A\n"
    )

    assert gpus == [
        {
            "index": 0,
            "name": "NVIDIA RTX 4090",
            "utilization_percent": 87.0,
            "memory_used_bytes": 12288 * system_monitor.MIB,
            "memory_total_bytes": 24564 * system_monitor.MIB,
            "memory_percent": 50.0,
            "temperature_c": 71.0,
        },
        {
            "index": 1,
            "name": "NVIDIA RTX 4060",
            "utilization_percent": None,
            "memory_used_bytes": 512 * system_monitor.MIB,
            "memory_total_bytes": 8188 * system_monitor.MIB,
            "memory_percent": 6.3,
            "temperature_c": None,
        },
    ]
