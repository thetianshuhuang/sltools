"""Module interacting with Slurm via scontrol to get node info."""

import dataclasses
import re
import subprocess

from . import scontrol


@dataclasses.dataclass
class Node:
    """Represents a single Slurm node."""

    name: str
    cpus: int
    memory: int  # MB
    gpus: int
    architecture: str
    state: str
    partitions: list[str]

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
    def from_record(cls, data: dict) -> "Node":
        """Creates a Node instance from a `scontrol show node` record.

        Args:
            data: Dictionary containing node information from scontrol.

        Returns:
            A Node instance with parsed and validated data.
        """
        return cls(
            name=scontrol.get(data, "NodeName", "unknown"),
            cpus=scontrol.get_int(data, "CPUTot"),
            memory=scontrol.get_int(data, "RealMemory"),
            gpus=Node._parse_gpu_count(scontrol.get(data, "Gres")),
            architecture=scontrol.get(data, "Arch", "unknown"),
            state=scontrol.get(data, "State", "UNKNOWN"),
            partitions=scontrol.get_list(data, "Partitions"),
        )


def get_nodes(partition: str | None = None) -> list[Node]:
    """Fetches nodes from scontrol.

    Args:
        partition: If set, only include nodes in this partition.
    """
    nodes = [Node.from_record(r) for r in scontrol.show("node")]
    if partition is not None:
        nodes = [n for n in nodes if partition in n.partitions]
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
