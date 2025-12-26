"""Main entrypoint and UI rendering for sltop."""

import datetime
import time
from typing import List

import tyro
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from sltop.slurm import Job, get_jobs, get_slurm_version


def format_resources(job: Job) -> str:
    """Formats the resources string for a job (e.g., 'b0 [gpu:4]' or '(Dependency)')."""
    # If pending, show reason
    if job.job_state == "PENDING":
        reason = job.state_reason
        if reason == "None":
            return ""
        return f"({reason})"

    # Otherwise show nodes and TRES
    # Format: "b0 [gpu:4]" or just "b0" if no TRES

    # TRES format from squeue: "gres/gpu:4"
    # We want to extract "gpu:4"
    tres = job.tres_per_node
    if tres.startswith("gres/"):
        tres = tres[5:]  # remove "gres/" prefix

    if tres:
        return f"{job.nodelist} [{tres}]"

    return job.nodelist


def render(jobs: List[Job], slurm_version: str) -> Panel:
    """Renders the list of jobs into a Rich Panel."""
    table = Table(box=None, padding=(0, 1), show_lines=False, expand=True)

    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("PART", style="dim", no_wrap=True)
    table.add_column("USER", style="yellow", no_wrap=True)
    table.add_column("NAME", no_wrap=True)
    table.add_column("ST", style="bold", no_wrap=True)
    table.add_column("TIME", justify="right", no_wrap=True)
    table.add_column("RESOURCES", no_wrap=True)

    for job in jobs:
        # Status color
        st_style = "green" if job.job_state == "RUNNING" else "yellow"
        if job.job_state == "PENDING":
            st_style = "yellow"
        elif job.job_state == "CANCELLED":
            st_style = "red"

        # State short code
        st_code = job.job_state[:2]  # RU, PE...
        if job.job_state == "RUNNING":
            st_code = "R"
        elif job.job_state == "PENDING":
            st_code = "PD"

        table.add_row(
            str(job.job_id),
            job.partition,
            job.user_name,
            job.name,
            Text(st_code, style=st_style),
            job.time_used,
            escape(format_resources(job)),
        )

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header_grid = Table.grid(expand=True)
    header_grid.add_column(justify="left")
    header_grid.add_column(justify="right")
    header_grid.add_row(
        Text(f"sltop/slurm v{slurm_version}", style="bold white"),
        Text(now_str, style="bold white"),
    )

    content = Group(Padding(header_grid, (0, 1)), Rule(style="dim"), table)

    return Panel(content, box=box.ROUNDED, padding=0)


def entrypoint(refresh: float = 1.0) -> None:
    """sltop: A top-like queue viewer for Slurm.

    Args:
        refresh: Refresh rate in seconds.
    """
    console = Console()
    slurm_version = get_slurm_version()

    with Live(console=console, screen=True, auto_refresh=False) as live:
        while True:
            jobs = get_jobs()
            panel = render(jobs, slurm_version)
            live.update(panel, refresh=True)
            time.sleep(refresh)


if __name__ == "__main__":
    tyro.cli(entrypoint)
