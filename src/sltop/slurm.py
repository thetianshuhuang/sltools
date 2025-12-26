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

    @property
    def time_used(self) -> str:
        """Returns the time used by the job formatted as H:MM:SS, or 0:00 if not running."""
        if self.job_state != "RUNNING":
            return "-"

        now = int(time.time())
        diff = now - self.start_time
        return str(datetime.timedelta(seconds=diff))


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

        # Start time
        start_time = 0
        if "start_time" in j and isinstance(j["start_time"], dict):
            start_time = j["start_time"].get("number", 0)

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
        )
        jobs.append(job)

    return sort_jobs(jobs)


def sort_jobs(jobs: list[Job]) -> list[Job]:
    """Sorts jobs according to the following logic:

    1. State: Pending before Running.
    2. Running: Increasing execution time (Youngest first).
    3. Pending: Decreasing Job ID (Newest first).
    """
    pending_jobs = [j for j in jobs if j.job_state == "PENDING"]
    running_jobs = [j for j in jobs if j.job_state == "RUNNING"]
    other_jobs = [j for j in jobs if j.job_state not in ("PENDING", "RUNNING")]

    # Sort Pending by Job ID descending
    pending_jobs.sort(key=lambda j: j.job_id, reverse=True)

    # Sort Running by Start Time descending (Youngest first)
    running_jobs.sort(key=lambda j: j.start_time, reverse=True)

    # Sort others by ID ?
    other_jobs.sort(key=lambda j: j.job_id, reverse=True)

    return pending_jobs + running_jobs + other_jobs


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
