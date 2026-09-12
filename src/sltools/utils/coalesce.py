"""Merging of similar jobs into groups, for a more compact job list."""

import dataclasses

from .jobs import Job


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

    # Iterate linearly, maintaining a 'current_group' which is either a Job or
    # a JobGroup.

    current_group = None

    for job in jobs:
        if current_group is None:
            # Start a potential new group (initially just the job itself);
            # it is only converted to a JobGroup once something merges into it.
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
                length = len(name2)
                diff_len = len(diff2)
                if (diff_len < (length // 4)) or (
                    (diff_len < 5) and (diff_len < (length // 2))
                ):
                    if not isinstance(current_group, JobGroup):
                        current_group = JobGroup(current_group)

                    current_group.ids.append(job.job_id)

                    if diff1.startswith("[") and diff1.endswith("]"):
                        existing_diffs = diff1[1:-1]  # "a,b"
                        new_diffs = f"{existing_diffs},{diff2}"
                        current_group.combined_name = (
                            f"{prefix}[{new_diffs}]{suffix}"
                        )
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

    There is always exactly one difference block, since the common prefix and
    suffix are only ever shortened to reach a delimiter, which can only widen
    the gap between them; nothing can be lost or overlap.

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
        suffix_len < min(rem1, rem2)
        and s1[-(suffix_len + 1)] == s2[-(suffix_len + 1)]
    ):
        suffix_len += 1

    # 3. Expansion heuristics (Backtrack prefix and suffix to delimiters)
    delimiters = {".", "-", "_", "/", ":"}  # Added colon just in case

    # Backtrack prefix
    current_prefix_len = prefix_len
    while current_prefix_len > 0:
        char = s1[current_prefix_len - 1]
        if char in delimiters:
            break
        current_prefix_len -= 1

    # Shrink suffix, moving the boundary leftwards from the end
    current_suffix_len = suffix_len
    while current_suffix_len > 0:
        char = s1[len(s1) - current_suffix_len]  # First char of suffix
        if char in delimiters:
            break
        current_suffix_len -= 1

    final_prefix = s1[:current_prefix_len]
    final_suffix = (
        s1[len(s1) - current_suffix_len :] if current_suffix_len > 0 else ""
    )

    final_diff1 = s1[current_prefix_len : len(s1) - current_suffix_len]
    final_diff2 = s2[current_prefix_len : len(s2) - current_suffix_len]

    return final_prefix, final_diff1, final_diff2, final_suffix
