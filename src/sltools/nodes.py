"""Module interacting with Slurm via scontrol to get node info."""

import dataclasses
import json
import re
import subprocess
from typing import List


@dataclasses.dataclass
class Node:
    """Represents a single Slurm node."""

    name: str
    cpus: int
    memory: int  # MB
    gpus: int
    architecture: str
    state: str


def get_nodes() -> List[Node]:
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
        memory = 0
        real_memory = n.get("real_memory")
        if isinstance(real_memory, int):
            memory = real_memory
        elif isinstance(real_memory, dict):
            memory = real_memory.get("number", 0)
        # Fallback to free_mem if real_memory missing? usually real_memory is the capacity.
        # Check standard fields if json structure varies.

        # GPUs from GRES
        # Format example: "gpu:6000:8" or "gpu:a100:4(S:0-1)"
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

    # Sort by name
    nodes.sort(key=lambda x: x.name)
    return nodes


def _parse_gpu_count(gres_str: str) -> int:
    """Parses GRES string to extract total GPU count.

    Examples:
        "gpu:6000:8" -> 8
        "gpu:a100:4(S:0-1)" -> 4
        "craynetwork:4,gpu:a100:2" -> 2
    """
    if not gres_str or "gpu" not in gres_str:
        return 0

    count = 0
    # Split by comma if multiple GRES types
    parts = gres_str.split(",")
    for part in parts:
        if part.strip().startswith("gpu"):
            # expected format: gpu[:type]:count[(...)]
            # remove parens first
            clean_part = re.sub(r"\(.*?\)", "", part)
            subparts = clean_part.split(":")
            # Last part should be the count if it's a number
            if len(subparts) > 1:
                try:
                    count += int(subparts[-1])
                except ValueError:
                    pass
    return count
