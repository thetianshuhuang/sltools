"""Module interacting with Slurm via scontrol to get node info."""

import dataclasses
import json
import re
import subprocess


@dataclasses.dataclass
class Node:
    """Represents a single Slurm node."""

    name: str
    cpus: int
    memory: int  # MB
    gpus: int
    architecture: str
    state: str


def get_nodes() -> list[Node]:
    """Fetches nodes from scontrol."""
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "nodes", "--json"], text=True
        )
        data = json.loads(output)
    except Exception:
        return []

    nodes = []
    for n in data.get("nodes", []):
        # CPUs
        cpus = n.get("cpus", 0)

        # Memory
        real_memory = n.get("real_memory")
        if isinstance(real_memory, int):
            memory = real_memory
        elif isinstance(real_memory, dict):
            memory = real_memory.get("number", 0)
        else:
            memory = 0

        gres = n.get("gres", "")
        gpus = _parse_gpu_count(gres)

        node = Node(
            name=n.get("name", "unknown"),
            cpus=cpus,
            memory=memory,
            gpus=gpus,
            architecture=n.get("architecture", "unknown"),
            state=n.get("state", "UNKNOWN"),
        )
        nodes.append(node)

    nodes.sort(key=lambda x: x.name)
    return nodes


def _parse_gpu_count(gres_str: str) -> int:
    """Parses GRES string to extract total GPU count."""
    if not gres_str or "gpu" not in gres_str:
        return 0

    count = 0
    parts = gres_str.split(",")
    for part in parts:
        if part.strip().startswith("gpu"):
            # expected format: gpu[:type]:count[(...)]
            clean_part = re.sub(r"\(.*?\)", "", part)
            subparts = clean_part.split(":")
            if len(subparts) > 1:
                try:
                    count += int(subparts[-1])
                except ValueError:
                    pass
    return count
