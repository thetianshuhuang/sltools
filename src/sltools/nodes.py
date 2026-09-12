"""Module interacting with Slurm via scontrol to get node info."""

import dataclasses
import re
import subprocess

from . import scontrol

# Node capacity does not change while sltop is running, and querying it is not
# cheap on a large cluster, so it is only read once per process.
_node_cache: list["Node"] = []


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
    """Fetches nodes from scontrol, caching them for the life of the process.

    Nodes which are added, removed or resized are only picked up on a relaunch.

    Args:
        partition: If set, only include nodes in this partition.
    """
    global _node_cache

    # An empty cache is also the failure case, so it is retried.
    if not _node_cache:
        nodes = [Node.from_record(r) for r in scontrol.show("node")]
        nodes.sort(key=lambda x: x.name)
        _node_cache = nodes

    nodes = _node_cache
    if partition is not None:
        nodes = [n for n in nodes if partition in n.partitions]
    return nodes


def _split_nodelist(nodelist: str) -> list[str]:
    """Splits a nodelist on commas which are not inside a `[...]` range."""
    terms = []
    start = 0
    depth = 0
    for i, char in enumerate(nodelist):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            terms.append(nodelist[start:i])
            start = i + 1
    terms.append(nodelist[start:])
    return [t for t in terms if t]


def _expand_term(term: str) -> list[str]:
    """Expands the `[...]` ranges in a single nodelist term."""
    match = re.search(r"\[([^\]]*)\]", term)
    if not match:
        return [term]

    prefix, suffix = term[: match.start()], term[match.end() :]
    names = []
    for part in match.group(1).split(","):
        start, _, stop = part.partition("-")
        if not stop:
            names.append(f"{prefix}{start}{suffix}")
            continue
        # Node indices are zero padded to the width they are written with.
        width = len(start) if start.startswith("0") else 0
        for index in range(int(start), int(stop) + 1):
            names.append(f"{prefix}{index:0{width}d}{suffix}")

    # The suffix may contain further ranges, e.g. "b[1-2]x[3-4]".
    return [name for term in names for name in _expand_term(term)]


def expand_nodelist(nodelist: str) -> list[str]:
    """Expands a Slurm nodelist string into a list of node names.

    This is what `scontrol show hostnames` does, but that costs a subprocess
    per call, and it is called once per running job on every refresh.

    Examples:
        "b0" -> ["b0"]
        "b[0-2],c1" -> ["b0", "b1", "b2", "c1"]
        "robo-gh00[1-2]" -> ["robo-gh001", "robo-gh002"]
    """
    if not nodelist:
        return []

    try:
        return [
            name
            for term in _split_nodelist(nodelist)
            for name in _expand_term(term)
        ]
    except ValueError:
        # Fall back to scontrol for any syntax which is not understood.
        try:
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
        output = subprocess.check_output(
            ["squeue", "--version"], text=True
        ).strip()
        parts = output.split()
        if len(parts) >= 2:
            return parts[1]
        return output
    except Exception:
        return "unknown"
