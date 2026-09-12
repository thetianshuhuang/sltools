"""Module for reading Slurm state via `scontrol`.

Slurm's `--json` output is only available on recent versions, so everything is
read from the plain `key=value` output of `scontrol --oneliner`, which is
stable across versions.
"""

import datetime
import re
import subprocess

# Keys are alphanumeric, but may also contain "/", ":" or "." (e.g. "CPUs/Task",
# "ReqB:S:C:T"). Values run up to the next " Key=" pair, since some values do
# contain spaces (e.g. "JobName=my job", "OS=Linux 6.8.0-138-generic #138 ...").
_KEY_VALUE = re.compile(
    r"([A-Za-z_][\w/:.-]*)=(.*?)(?=\s+[A-Za-z_][\w/:.-]*=|$)",
)

# Values Slurm prints in place of an unset field.
_UNSET = {"", "(null)", "N/A"}


def parse_record(line: str) -> dict[str, str]:
    """Parses a single `key=value ...` line of scontrol output.

    Args:
        line: One record of `scontrol --oneliner` output.

    Returns:
        The record as a dictionary; keys which appear more than once keep their
        first value.
    """
    record: dict[str, str] = {}
    for key, value in _KEY_VALUE.findall(line):
        record.setdefault(key, value)
    return record


def show(entity: str, contains: str | None = None) -> list[dict[str, str]]:
    """Runs `scontrol show <entity>`, returning one record per entity.

    Args:
        entity: The entity to show, e.g. "job" or "node".
        contains: If set, records which do not contain this substring are
            dropped without being parsed. Parsing dominates the cost of this
            function on a large cluster, so this is worth doing even though it
            is only an approximation of the real filter.

    Returns:
        A list of parsed records; empty if scontrol is unavailable or failed.
    """
    try:
        output = subprocess.check_output(
            ["scontrol", "--oneliner", "show", entity],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    # Filter out empty records, e.g. the "No jobs in the system" message.
    records = [
        parse_record(line)
        for line in output.splitlines()
        if contains is None or contains in line
    ]
    return [r for r in records if r]


def get(record: dict[str, str], key: str, default: str = "") -> str:
    """Returns a string field, substituting `default` for unset values."""
    value = record.get(key, "").strip()
    return default if value in _UNSET else value


def get_list(record: dict[str, str], key: str) -> list[str]:
    """Returns a comma separated field, e.g. `Partitions=dev,batch`."""
    value = get(record, key)
    return value.split(",") if value else []


def get_int(record: dict[str, str], key: str, default: int = 0) -> int:
    """Returns an integer field, e.g. `NumCPUs=64`.

    Ranges such as `NumNodes=1-4` are reported as their lower bound.
    """
    match = re.match(r"-?\d+", get(record, key))
    return int(match.group()) if match else default


def get_time(record: dict[str, str], key: str, default: int = 0) -> int:
    """Returns a timestamp field, e.g. `StartTime=2026-09-12T04:45:31`.

    Placeholders such as `Unknown` are reported as `default`.
    """
    try:
        return int(
            datetime.datetime.fromisoformat(get(record, key)).timestamp()
        )
    except ValueError:
        return default
