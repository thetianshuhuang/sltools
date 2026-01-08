"""Module interacting with Slurm via squeue."""

import dataclasses
import datetime
import json
import subprocess
import time


@dataclasses.dataclass
class Job:
    """Represents a single Slurm job."""

    job_id: int
    partition: str
    name: str
    user_name: str
    job_state: str
    start_time: int
    node_count: int
    nodelist: str
    tres_per_node: str
    state_reason: str
    cpus: int
    memory: int  # Total memory in MB

    @property
    def time_used(self) -> str:
        """Returns the time used by the job formatted as H:MM:SS."""
        if self.job_state != "RUNNING":
            return "-"

        now = int(time.time())
        diff = now - self.start_time
        return str(datetime.timedelta(seconds=diff))

    def get_resources_per_node(self) -> dict:
        """Parses tres_per_node into a dictionary {type: count}.

        Examples:
            "gpu:4" -> {'gpu': 4}
            "cpu:8,gpu:1" -> {'cpu': 8, 'gpu': 1}
        """
        if not self.tres_per_node:
            return {}

        res = {}
        # remove "gres/" prefix if present (common in some slurm versions/configs)
        tres_str = self.tres_per_node
        if tres_str.startswith("gres/"):
            tres_str = tres_str[5:]

        parts = tres_str.split(",")
        for part in parts:
            if ":" in part:
                # key:val or key:type:val
                sub = part.split(":")
                key = sub[0]
                try:
                    val = int(sub[-1])
                    res[key] = val
                except ValueError:
                    pass
        return res


def get_jobs() -> list[Job]:
    """Fetches jobs from squeue and calls sort_jobs."""
    output = subprocess.check_output(["squeue", "--json"], text=True)
    data = json.loads(output)

    jobs = []
    for j in data.get("jobs", []):
        # Parse nested numeric fields safely
        node_count = 0
        if "node_count" in j and isinstance(j["node_count"], dict):
            node_count = j["node_count"].get("number", 0)

        # State is a list
        state = "UNKNOWN"
        if (
            "job_state" in j
            and isinstance(j["job_state"], list)
            and len(j["job_state"]) > 0
        ):
            state = j["job_state"][0]

        if "start_time" in j and isinstance(j["start_time"], dict):
            start_time = j["start_time"].get("number", 0)

        # CPUs
        cpus = 0
        if "cpus" in j and isinstance(j["cpus"], dict):
            cpus = j["cpus"].get("number", 0)

        # Memory from tres_alloc_str (e.g., mem=720000M)
        memory = 0
        tres_alloc = j.get("tres_alloc_str", "")
        if tres_alloc:
            memory = _parse_memory_from_tres(tres_alloc)

        job = Job(
            job_id=j.get("job_id", 0),
            partition=j.get("partition", ""),
            name=j.get("name", ""),
            user_name=j.get("user_name", ""),
            job_state=state,
            start_time=start_time,
            node_count=node_count,
            nodelist=j.get("nodes", ""),
            tres_per_node=j.get("tres_per_node", ""),
            state_reason=j.get("state_reason", ""),
            cpus=cpus,
            memory=memory,
        )
        jobs.append(job)

    return sort_jobs(jobs)


def sort_jobs(jobs: list[Job]) -> list[Job]:
    """Sorts jobs according to the following logic:

    1. Running jobs, in decreasing order of execution time (Longest running first).
    2. Pending jobs waiting for resources (Reason: Resources).
    3. Pending jobs with reason Priority.
    4. Pending jobs with reason Dependency.
    5. Failed/Cancelled/Other.
    """
    jobs.sort(key=lambda j: j.job_id)

    job_categories = {}
    for j in jobs:
        if j.job_state == "RUNNING":
            category = "RUNNING"
        else:
            category = j.state_reason

        if category not in jobs:
            job_categories[category] = []
        job_categories[category].append(j)

    sorted_jobs = (
        sorted(job_categories.pop("RUNNING", []), key=lambda j: j.start_time)
        + job_categories.pop("Resources", [])
        + job_categories.pop("Priority", [])
    )

    for k in job_categories:
        if "qos" in k.lower():
            sorted_jobs += job_categories.pop(k)

    sorted_jobs += job_categories.pop("Dependency", [])

    for k, v in job_categories.items():
        sorted_jobs += v

    return sorted_jobs


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


def _parse_memory_from_tres(tres_str: str) -> int:
    """Parses memory from a TRES string (e.g. cpu=96,mem=720000M,...) -> MB."""
    units = {"M": 1, "G": 1024, "T": 1024 * 1024, "K": 1 / 1024}
    # Look for mem=...
    # simple split implementation
    try:
        parts = tres_str.split(",")
        for part in parts:
            if part.startswith("mem="):
                val_str = part[4:]  # remove "mem="
                unit = val_str[-1].upper()
                if unit.isdigit():
                    return int(val_str)  # Default MB

                return int(float(val_str[:-1]) * units[unit])
    except Exception:
        pass
    return 0


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
