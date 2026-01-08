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

    @staticmethod
    def _parse_memory_from_tres(tres_str: str) -> int:
        """Parses memory from a TRES string."""
        units = {"M": 1, "G": 1024, "T": 1024 * 1024, "K": 1 / 1024}
        # Look for mem=...
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

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Creates a Job instance from a dictionary (from squeue JSON output).

        Args:
            data: Dictionary containing job information from squeue.

        Returns:
            A Job instance with parsed and validated data.
        """
        # Parse node_count
        node_count = 0
        if "node_count" in data and isinstance(data["node_count"], dict):
            node_count = data["node_count"].get("number", 0)

        # Parse job_state (it's a list)
        state = "UNKNOWN"
        if (
            "job_state" in data
            and isinstance(data["job_state"], list)
            and len(data["job_state"]) > 0
        ):
            state = data["job_state"][0]

        # Parse start_time
        start_time = 0
        if "start_time" in data and isinstance(data["start_time"], dict):
            start_time = data["start_time"].get("number", 0)

        # Parse CPUs
        cpus = 0
        if "cpus" in data and isinstance(data["cpus"], dict):
            cpus = data["cpus"].get("number", 0)

        # Parse memory from tres_alloc_str (e.g., mem=720000M)
        memory = 0
        tres_alloc = data.get("tres_alloc_str", "")
        if tres_alloc:
            memory = Job._parse_memory_from_tres(tres_alloc)

        return cls(
            job_id=data.get("job_id", 0),
            partition=data.get("partition", ""),
            name=data.get("name", ""),
            user_name=data.get("user_name", ""),
            job_state=state,
            start_time=start_time,
            node_count=node_count,
            nodelist=data.get("nodes", ""),
            tres_per_node=data.get("tres_per_node", ""),
            state_reason=data.get("state_reason", ""),
            cpus=cpus,
            memory=memory,
        )


@dataclasses.dataclass
class JobGroup(Job):
    """Represents a group of similar Slurm jobs."""

    ids: list[int] = dataclasses.field(default_factory=list)
    combined_name: str = ""

    def __init__(self, job: Job):
        """Initialize from a single job."""
        for field in dataclasses.fields(Job):
            setattr(self, field.name, getattr(job, field.name))

        self.ids = [job.job_id]
        self.combined_name = job.name

    @property
    def job_id_str(self) -> str:
        """Returns the job IDs string."""
        if len(self.ids) > 1:
            return f"[{','.join(map(str, self.ids))}]"
        return str(self.ids[0])


def coalesce_jobs(jobs: list[Job]) -> list[Job]:
    """Coalesces similar jobs into JobGroups."""
    if not jobs:
        return []

    coalesced = []

    # We will iterate and maintain a 'current_group' which is either a Job or JobGroup
    # Actually, let's process linearly

    current_group = None

    for job in jobs:
        if current_group is None:
            # Start a potential new group (initially just the job itself)
            # We wrap it in JobGroup only when merging? Or always work with Job/JobGroup union?
            # To be safe, let's keep it simplest: current_group is a JobGroup candidate
            # But we don't convert until we merge.
            current_group = job
            continue

        # Check if we can merge 'job' into 'current_group'
        merged = False

        # 1. Basic Eligibility Conditions
        if (
            current_group.job_state != "RUNNING"
            and job.job_state != "RUNNING"
            and current_group.partition == job.partition
            and current_group.user_name == job.user_name
            and current_group.job_state == job.job_state
            and current_group.state_reason == job.state_reason
        ):
            # 2. Similarity Check
            name1 = (
                current_group.combined_name
                if isinstance(current_group, JobGroup)
                else current_group.name
            )
            name2 = job.name

            res = _get_smart_diff(name1, name2)
            if res:
                prefix, diff1, diff2, suffix = res
                L = len(name2)
                D = len(diff2)
                if (D < (L // 4)) or ((D < 5) and (D < (L // 2))):
                    if not isinstance(current_group, JobGroup):
                        current_group = JobGroup(current_group)

                    current_group.ids.append(job.job_id)

                    if diff1.startswith("[") and diff1.endswith("]"):
                        existing_diffs = diff1[1:-1]  # "a,b"
                        new_diffs = f"{existing_diffs},{diff2}"
                        current_group.combined_name = f"{prefix}[{new_diffs}]{suffix}"
                    else:
                        current_group.combined_name = (
                            f"{prefix}[{diff1},{diff2}]{suffix}"
                        )

                    merged = True

        if not merged:
            coalesced.append(current_group)
            current_group = job

    if current_group:
        coalesced.append(current_group)

    return coalesced


def _get_smart_diff(s1: str, s2: str) -> tuple[str, str, str, str] | None:
    """Returns the differing substring respecting delimiters.

    Returns: (prefix, diff1, diff2, suffix) or None if not compatible.
    """
    if s1 == s2:
        return None

    # 1. Find Longest Common Prefix
    prefix_len = 0
    min_len = min(len(s1), len(s2))
    while prefix_len < min_len and s1[prefix_len] == s2[prefix_len]:
        prefix_len += 1

    # 2. Find Longest Common Suffix
    suffix_len = 0
    # Must stop before overlapping with prefix
    # Adjust for remaining length
    rem1 = len(s1) - prefix_len
    rem2 = len(s2) - prefix_len

    while (
        suffix_len < min(rem1, rem2) and s1[-(suffix_len + 1)] == s2[-(suffix_len + 1)]
    ):
        suffix_len += 1

    # Initial diff not strictly needed if we recalculate later
    # diff1 = s1[prefix_len : len(s1) - suffix_len]
    # diff2 = s2[prefix_len : len(s2) - suffix_len]

    # 3. Expansion heuristics (Backtrack prefix and suffix to delimiters)
    delimiters = {".", "-", "_", "/", ":"}  # Added colon just in case

    # Backtrack prefix
    current_prefix_len = prefix_len
    while current_prefix_len > 0:
        char = s1[current_prefix_len - 1]
        if char in delimiters:
            break
        current_prefix_len -= 1

    # Shrink suffix (which effectively moves the boundary leftwards from the end)
    current_suffix_len = suffix_len
    while current_suffix_len > 0:
        char = s1[len(s1) - current_suffix_len]  # First char of suffix
        if char in delimiters:
            break
        current_suffix_len -= 1

    final_prefix = s1[:current_prefix_len]
    final_suffix = s1[len(s1) - current_suffix_len :] if current_suffix_len > 0 else ""

    final_diff1 = s1[current_prefix_len : len(s1) - current_suffix_len]
    final_diff2 = s2[current_prefix_len : len(s2) - current_suffix_len]

    # 4. Check that nothing is lost or malformed
    # (The simple expansion logic should be safe but let's be sure we have a "clean" difference).
    # We want exactly ONE difference block. Our logic forces that structure: P + D + S.
    # But we should ensure we didn't eat too much or create overlaps?
    # With the logic above, we strictly reduced prefix_len and suffix_len, so gaps only got bigger (good).

    return final_prefix, final_diff1, final_diff2, final_suffix


def get_jobs() -> list[Job]:
    """Fetches jobs from squeue and calls sort_jobs."""
    output = subprocess.check_output(["squeue", "--json"], text=True)
    data = json.loads(output)

    jobs = [Job.from_dict(j) for j in data.get("jobs", [])]
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

        if category not in job_categories:
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
