"""Module interacting with Slurm via scontrol to get job info."""

import dataclasses
import datetime
import time

from . import scontrol

# Base states which Slurm considers finished; scontrol keeps reporting these
# jobs for a few minutes after they complete, but squeue hides them by default.
_FINISHED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


@dataclasses.dataclass
class Job:
    """Represents a single Slurm job."""

    job_id: int
    partition: str
    name: str
    user_name: str
    job_state: str
    start_time: int
    nice: int
    node_count: int
    nodelist: str
    tres_per_node: str
    state_reason: str
    cpus: int
    tres_alloc: str  # Total allocated TRES, e.g. "cpu=64,mem=375G,gres/gpu=4"

    @property
    def partitions(self) -> list[str]:
        """Returns the partitions the job may run in."""
        return self.partition.split(",")

    @property
    def gres(self) -> str:
        """Returns the generic resources per node, e.g. "gpu:h100:2".

        Slurm prefixes these with "gres/" or "gres:" depending on the version.
        """
        return self.tres_per_node.removeprefix("gres/").removeprefix("gres:")

    @property
    def time_used(self) -> str:
        """Returns the time used by the job formatted as H:MM:SS."""
        if self.job_state != "RUNNING":
            return "-"

        now = int(time.time())
        diff = now - self.start_time
        return str(datetime.timedelta(seconds=diff))

    @staticmethod
    def _parse_memory(mem_str: str) -> int:
        """Parses a memory size such as "375G" into MB (no suffix: already MB)."""
        units = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
        try:
            unit = mem_str[-1].upper()
            if unit.isdigit():
                return int(mem_str)  # Default MB

            return int(float(mem_str[:-1]) * units[unit])
        except (IndexError, KeyError, ValueError):
            return 0

    def get_resources_per_node(self) -> dict:
        """Parses tres_per_node into a dictionary {type: count}.

        Examples:
            "gpu:4" -> {'gpu': 4}
            "cpu:8,gpu:1" -> {'cpu': 8, 'gpu': 1}
        """
        res = {}
        for part in self.gres.split(","):
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

    def get_resources_total(self) -> dict:
        """Parses tres_alloc into a dictionary {type: count} for the whole job.

        Memory is converted to MB.

        Examples:
            "cpu=64,mem=375G,node=1,gres/gpu=4"
                -> {'cpu': 64, 'mem': 384000, 'node': 1, 'gpu': 4}
        """
        res = {}
        for part in self.tres_alloc.split(","):
            key, _, val = part.partition("=")
            key = key.rsplit("/", 1)[-1]  # "gres/gpu" -> "gpu"
            if key == "mem":
                res[key] = Job._parse_memory(val)
            else:
                try:
                    res[key] = int(val)
                except ValueError:
                    pass
        return res

    @staticmethod
    def _parse_job_id(job_id: str) -> int:
        """Parses a job ID, e.g. "12345", or "12345_7" for job array tasks."""
        try:
            return int(job_id.split("_")[0])
        except ValueError:
            return 0

    @classmethod
    def from_record(cls, data: dict) -> "Job":
        """Creates a Job instance from a `scontrol show job` record.

        Args:
            data: Dictionary containing job information from scontrol.

        Returns:
            A Job instance with parsed and validated data.
        """
        return cls(
            job_id=Job._parse_job_id(scontrol.get(data, "JobId")),
            partition=scontrol.get(data, "Partition"),
            name=scontrol.get(data, "JobName"),
            # "UserId=alice(1000)" -> "alice"
            user_name=scontrol.get(data, "UserId").split("(")[0],
            job_state=scontrol.get(data, "JobState", "UNKNOWN"),
            start_time=scontrol.get_time(data, "StartTime"),
            nice=scontrol.get_int(data, "Nice"),
            node_count=scontrol.get_int(data, "NumNodes"),
            nodelist=scontrol.get(data, "NodeList"),
            tres_per_node=scontrol.get(data, "TresPerNode"),
            state_reason=scontrol.get(data, "Reason", "None"),
            cpus=scontrol.get_int(data, "NumCPUs"),
            # Older Slurm versions report the allocation as "TRES" instead.
            tres_alloc=scontrol.get(data, "AllocTRES") or scontrol.get(data, "TRES"),
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
        """Returns the job IDs string, factoring out shared leading digits.

        Collapses consecutive runs into dash ranges.

        Examples:
            [12345] -> "12345"
            [12345, 12346, 12347] -> "1234[5-7]"
            [12345, 12347, 12348] -> "1234[5,7-8]"
        """
        if len(self.ids) == 1:
            return str(self.ids[0])

        ids = sorted(self.ids)
        id_strs = [str(i) for i in ids]

        # Longest common prefix across all ids.
        prefix_len = 0
        min_len = min(len(s) for s in id_strs)
        while prefix_len < min_len and all(
            s[prefix_len] == id_strs[0][prefix_len] for s in id_strs
        ):
            prefix_len += 1

        prefix = id_strs[0][:prefix_len]
        diffs = [s[prefix_len:] for s in id_strs]

        # Collapse consecutive runs of ids into dash ranges.
        parts = []
        i = 0
        while i < len(ids):
            j = i
            while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
                j += 1
            parts.append(diffs[i] if j == i else f"{diffs[i]}-{diffs[j]}")
            i = j + 1

        return f"{prefix}[{','.join(parts)}]"


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


def get_jobs(include_invalid: bool = False, partition: str | None = None) -> list[Job]:
    """Fetches jobs from scontrol and calls sort_jobs.

    Args:
        include_invalid: Whether to include jobs which can never run, i.e. jobs
            whose dependencies can never be satisfied.
        partition: If set, only include jobs in this partition.
    """
    jobs = [Job.from_record(r) for r in scontrol.show("job")]
    jobs = [j for j in jobs if j.job_state not in _FINISHED_STATES]
    if partition is not None:
        jobs = [j for j in jobs if partition in j.partitions]
    if not include_invalid:
        jobs = [j for j in jobs if j.state_reason != "DependencyNeverSatisfied"]
    return sort_jobs(jobs)


def sort_jobs(jobs: list[Job]) -> list[Job]:
    """Sorts jobs according to the following logic:

    1. Running jobs, in decreasing order of execution time (Longest running first).
    2. Pending jobs waiting for resources (Reason: Resources).
    3. Pending jobs with reason Priority, sorted by nice.
    4. Pending jobs with reason Dependency, sorted by nice.
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

    for category, category_jobs in job_categories.items():
        if category != "RUNNING":
            category_jobs.sort(key=lambda j: j.nice)

    sorted_jobs = (
        sorted(job_categories.pop("RUNNING", []), key=lambda j: j.start_time)
        + job_categories.pop("Resources", [])
        + job_categories.pop("Priority", [])
    )

    for k in job_categories:
        if "qos" in k.lower():
            sorted_jobs += job_categories[k]
    job_categories = {k: v for k, v in job_categories.items() if "qos" not in k.lower()}

    sorted_jobs += job_categories.pop("Dependency", [])

    for k, v in job_categories.items():
        sorted_jobs += v

    return sorted_jobs
