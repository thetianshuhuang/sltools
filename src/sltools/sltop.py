"""Main entrypoint and UI rendering for sltop."""

import datetime
import select
import sys
import termios
import time
import tty
from typing import List

import tyro
from rich import box
from rich.bar import Bar
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .jobs import Job, JobGroup, coalesce_jobs, get_jobs
from .nodes import Node, expand_nodelist, get_nodes, get_slurm_version


def format_resources(job: Job) -> str:
    """Formats the resources string for a job (e.g., 'b0 [gpu:4]' or '(Dependency)')."""
    if job.job_state == "PENDING":
        reason = job.state_reason
        if reason == "None":
            return ""
        return f"({reason})"

    tres = job.tres_per_node
    if tres.startswith("gres/"):
        tres = tres[5:]  # remove "gres/" prefix

    if tres:
        return f"{job.nodelist} [{tres}]"

    return job.nodelist


def calculate_node_usage(nodes: List[Node], jobs: List[Job]) -> dict:
    """Calculates used resources per node, broken down by partition.

    Structure:
    {
        "node_name": {
            "cpus": { "partition1": count, "partition2": count },
            "gpus": { "partition1": count, ... },
            "memory": { "partition1": count, ... }
        }
    }
    """
    usage = {n.name: {"cpus": {}, "gpus": {}, "memory": {}} for n in nodes}

    for job in jobs:
        if job.job_state != "RUNNING":
            continue

        affected_nodes = expand_nodelist(job.nodelist)
        num_nodes = len(affected_nodes)
        if num_nodes == 0:
            continue

        # Determine resources used per node for this job
        res_per_node = job.get_resources_per_node()

        # 1. GPU: Rely on tres_per_node (GRES)
        gpus_alloc = res_per_node.get("gpu", 0)

        # 2. CPU: explicit tres or distribute total
        cpus_alloc = res_per_node.get("cpu", 0)
        if cpus_alloc == 0 and job.cpus > 0:
            cpus_alloc = job.cpus // num_nodes

        # 3. Mem: explicit tres or distribute total
        mem_alloc = res_per_node.get("mem", 0)
        if mem_alloc == 0 and job.memory > 0:
            mem_alloc = job.memory // num_nodes

        for node_name in affected_nodes:
            if node_name not in usage:
                continue

            # Accumulate
            p = job.partition

            if cpus_alloc > 0:
                usage[node_name]["cpus"][p] = (
                    usage[node_name]["cpus"].get(p, 0) + cpus_alloc
                )

            if gpus_alloc > 0:
                usage[node_name]["gpus"][p] = (
                    usage[node_name]["gpus"].get(p, 0) + gpus_alloc
                )

            if mem_alloc > 0:
                usage[node_name]["memory"][p] = (
                    usage[node_name]["memory"].get(p, 0) + mem_alloc
                )

    return usage


def render_node_section(nodes: List[Node], usage_data: dict) -> Table:
    """Renders the reserved resources section."""
    table = Table(box=None, padding=(0, 1), show_lines=False, expand=True)
    table.add_column("", style="bold white", no_wrap=True)
    table.add_column("GPU", ratio=1)
    table.add_column("", style="white dim", no_wrap=True)
    table.add_column("CPU", ratio=1)
    table.add_column("", style="white dim", no_wrap=True)
    table.add_column("MEM", ratio=1)
    table.add_column("", style="white dim", no_wrap=True)

    for node in nodes:
        u = usage_data.get(node.name, {"cpus": {}, "gpus": {}, "memory": {}})

        gpu_used = sum(u["gpus"].values())
        cpu_used = sum(u["cpus"].values())
        mem_used = sum(u["memory"].values())

        # Convert memory to GB
        mem_total_gb = node.memory // 1000
        mem_used_gb = mem_used // 1000

        def make_cell(total: int, used: int, color: str, units: str = ""):
            bar = Bar(
                size=total,
                begin=0,
                end=used,
                width=None,
                color=color,
                bgcolor="bright_black",
            )
            stats = Text(f"{used}/{total}{units}", style="white dim")
            return bar, stats

        table.add_row(
            node.name,
            *make_cell(node.gpus, gpu_used, "cyan"),
            *make_cell(node.cpus, cpu_used, "magenta"),
            *make_cell(mem_total_gb, mem_used_gb, "green", units="G"),
        )

    return table


def render(jobs: List[Job], nodes: List[Node], slurm_version: str) -> Panel:
    """Renders the list of jobs into a Rich Panel."""
    # 1. Top Section Header
    now_str = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header_grid = Table.grid(expand=True)
    header_grid.add_column(justify="left")
    header_grid.add_column(justify="right")
    header_grid.add_row(
        Text(f"sltop/slurm v{slurm_version}", style="bold white"),
        Text(now_str, style="bold white"),
    )

    # 2. Middle Section: Node Usage
    node_usage = calculate_node_usage(nodes, jobs)
    node_table = render_node_section(nodes, node_usage)

    # 3. Bottom Section: Job List
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

        # Prepare display values
        if isinstance(job, JobGroup):
            job_id_display = job.job_id_str
            job_name_display = job.combined_name
        else:
            job_id_display = str(job.job_id)
            job_name_display = job.name

        table.add_row(
            escape(job_id_display),
            job.partition,
            job.user_name,
            escape(job_name_display),
            Text(st_code, style=st_style),
            job.time_used,
            escape(format_resources(job)),
        )

    # Combine sections: Header -> Rule -> Nodes -> Rule -> Jobs
    content = Group(
        Padding(header_grid, (0, 1)),
        Rule(style="dim"),
        node_table,
        Rule(style="dim"),
        table,
    )

    return Panel(content, box=box.ROUNDED, padding=0)


def main(refresh: float = 1.0, merge: bool = True) -> int:
    """sltop: A top-like queue viewer for Slurm.

    Args:
        refresh: Refresh rate in seconds.
        merge: Whether to merge similar jobs.
    """
    console = Console()
    slurm_version = get_slurm_version()

    old_settings = None
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                jobs = get_jobs()
                if merge:
                    jobs = coalesce_jobs(jobs)
                nodes = get_nodes()
                panel = render(jobs, nodes, slurm_version)
                live.update(panel, refresh=True)

                if sys.stdin.isatty():
                    rlist, _, _ = select.select([sys.stdin], [], [], refresh)
                    if rlist:
                        if sys.stdin.read(1).lower() == "q":
                            break
                else:
                    time.sleep(refresh)
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C
    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    return 0


def _cli() -> int:
    return tyro.cli(main)
