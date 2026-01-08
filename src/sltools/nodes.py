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

    @staticmethod
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

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        """Creates a Node instance from a dictionary (from scontrol JSON output).

        Args:
            data: Dictionary containing node information from scontrol.

        Returns:
            A Node instance with parsed and validated data.
        """
        # Parse CPUs
        cpus = data.get("cpus", 0)

        # Parse memory
        real_memory = data.get("real_memory")
        if isinstance(real_memory, int):
            memory = real_memory
        elif isinstance(real_memory, dict):
            memory = real_memory.get("number", 0)
        else:
            memory = 0

        # Parse GPUs from GRES
        gres = data.get("gres", "")
        gpus = Node._parse_gpu_count(gres)

        return cls(
            name=data.get("name", "unknown"),
            cpus=cpus,
            memory=memory,
            gpus=gpus,
            architecture=data.get("architecture", "unknown"),
            state=data.get("state", "UNKNOWN"),
        )


def get_nodes() -> list[Node]:
    """Fetches nodes from scontrol."""
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "nodes", "--json"], text=True
        )
        data = json.loads(output)
    except Exception:
        return []

    nodes = [Node.from_dict(n) for n in data.get("nodes", [])]
    nodes.sort(key=lambda x: x.name)
    return nodes


def expand_nodelist(nodelist: str) -> list[str]:
    """Expands a Slurm nodelist string into a list of node names."""
    if not nodelist:
        return []
    try:
        # Use scontrol to expand (robust standard way)
        output = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist], text=True
        )
        return output.strip().splitlines()
    except Exception:
        return []


def get_slurm_version() -> str:
    """Returns the Slurm version string."""
    try:
        # squeue --version output: "slurm-wlm 23.11.4"
        output = subprocess.check_output(["squeue", "--version"], text=True).strip()
        parts = output.split()
        if len(parts) >= 2:
            return parts[1]
        return output
    except Exception:
        return "unknown"
