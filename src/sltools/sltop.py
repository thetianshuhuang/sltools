"""Main entrypoint and UI rendering for sltop."""

import datetime
import os
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

from . import __version__
from .utils.coalesce import JobGroup, coalesce_jobs
from .utils.jobs import Job, get_jobs
from .utils.nodes import Node, get_nodes, get_slurm_version
from .utils.usage import calculate_node_usage

# Keys which scroll the job list, including the escape sequences sent by the
# arrow, page, home and end keys.
_KEY_UP = ("k", "\x1b[A")
_KEY_DOWN = ("j", "\x1b[B")
_KEY_PAGE_UP = ("\x1b[5~",)
_KEY_PAGE_DOWN = (" ", "\x1b[6~")
_KEY_HOME = ("g", "\x1b[H", "\x1b[1~")
_KEY_END = ("G", "\x1b[F", "\x1b[4~")
_KEY_QUIT = ("q", "Q")

# Rows the panel uses for everything except the job list itself: the two panel
# borders, the header, the two rules, the node table's header, and the job
# table's header. The node rows are counted separately, since they vary.
_CHROME_ROWS = 7

# The interactive view also has a rule and the controls bar below the jobs.
_CONTROLS_ROWS = 2

# Controls listed by the bar at the bottom of the interactive view.
_CONTROLS = (
    ("\u2191/\u2193 j/k", "scroll"),
    ("PgUp/PgDn", "page"),
    ("g/G", "top/bottom"),
    ("q", "quit"),
)


def format_resources(job: Job) -> str:
    """Formats the resources for a job, e.g. "b0 [gpu:4]" or "(Dependency)"."""
    if job.job_state == "PENDING":
        reason = job.state_reason
        if reason == "None":
            return ""
        return f"({reason})"

    if job.gres:
        return f"{job.nodelist} [{job.gres}]"

    return job.nodelist


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


def render_controls() -> Text:
    """Renders the bar listing the controls of the interactive view."""
    bar = Text(justify="center")
    for index, (keys, action) in enumerate(_CONTROLS):
        if index:
            bar.append("    ")
        bar.append(keys, style="bold cyan")
        bar.append(f" {action}", style="dim")
    return bar


def render(
    jobs: List[Job],
    nodes: List[Node],
    slurm_version: str,
    offset: int = 0,
    max_rows: int | None = None,
    controls: bool = False,
) -> Panel:
    """Renders the list of jobs into a Rich Panel.

    Args:
        jobs: All of the jobs; node usage is always computed from all of them,
            no matter which of them are currently on screen.
        nodes: Nodes to show usage for.
        slurm_version: Slurm version, shown in the header.
        offset: Index of the first job to show.
        max_rows: Maximum number of jobs to show, or None to show all of them.
        controls: Whether to show the controls bar, i.e. whether the view is
            interactive.
    """
    visible = jobs[offset : offset + max_rows] if max_rows else jobs

    # 1. Top Section Header
    now_str = (
        datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    title = Text(
        f"sltop v{__version__} / slurm v{slurm_version}", style="bold white"
    )

    # The scroll position sits left of the clock, so that the clock stays put
    # as the position appears and disappears.
    status = Text()
    if len(visible) < len(jobs):
        shown = f"{offset + 1}-{offset + len(visible)} of {len(jobs)}"
        status.append(f"[{shown}]  ", style="bold orange1")
    status.append(now_str, style="bold white")

    header_grid = Table.grid(expand=True)
    header_grid.add_column(justify="left")
    header_grid.add_column(justify="right")
    header_grid.add_row(title, status)

    # 2. Middle Section: Node Usage
    node_usage = calculate_node_usage(nodes, jobs)
    node_table = render_node_section(nodes, node_usage)

    # 3. Bottom Section: Job List
    table = Table(box=None, padding=(0, 1), show_lines=False, expand=True)

    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("PART/NICE", no_wrap=True)
    table.add_column("USER", style="yellow", no_wrap=True)
    table.add_column("NAME", no_wrap=True, ratio=2)
    table.add_column("ST", style="bold", no_wrap=True)
    table.add_column("TIME", justify="right", no_wrap=True)
    table.add_column("RESOURCES", no_wrap=True, ratio=1)

    for job in visible:
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

        if job.nice > 0:
            nice_style = "bright_green"
        elif job.nice < 0:
            nice_style = "bright_red"
        else:
            nice_style = "dim"

        part_nice = Text(f"{job.partition}/")
        part_nice.append(str(job.nice), style=nice_style)

        table.add_row(
            escape(job_id_display),
            part_nice,
            job.user_name,
            escape(job_name_display),
            Text(st_code, style=st_style),
            job.time_used,
            escape(format_resources(job)),
        )

    # Combine sections: Header -> Rule -> Nodes -> Rule -> Jobs [-> Controls]
    sections = [
        Padding(header_grid, (0, 1)),
        Rule(style="dim"),
        node_table,
        Rule(style="dim"),
        table,
    ]
    if controls:
        sections += [Rule(style="dim"), Padding(render_controls(), (0, 1))]

    return Panel(Group(*sections), box=box.ROUNDED, padding=0)


def read_key(timeout: float) -> str | None:
    """Waits up to `timeout` seconds for a keypress.

    Returns:
        The key pressed, which is an escape sequence for e.g. the arrow keys,
        or None if nothing was pressed before the timeout.
    """
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return None

    # An escape sequence arrives as one burst, so a single read gets the key.
    return os.read(sys.stdin.fileno(), 8).decode(errors="ignore")


def scroll(key: str, offset: int, page: int, max_offset: int) -> int:
    """Returns the job list offset after a keypress, clamped to the list.

    Args:
        key: The key pressed, as returned by `read_key`.
        offset: The current offset.
        page: How far the page up/down keys move, i.e. the rows on screen.
        max_offset: The largest offset which still fills the screen.
    """
    if key in _KEY_UP:
        offset -= 1
    elif key in _KEY_DOWN:
        offset += 1
    elif key in _KEY_PAGE_UP:
        offset -= page
    elif key in _KEY_PAGE_DOWN:
        offset += page
    elif key in _KEY_HOME:
        offset = 0
    elif key in _KEY_END:
        offset = max_offset

    return max(0, min(offset, max_offset))


def main(
    refresh: float = 1.0,
    merge: bool = False,
    include_invalid: bool = False,
    partition: str | None = None,
    static: bool = False,
) -> int:
    """sltop: A top-like queue viewer for Slurm.

    Args:
        refresh: Refresh rate in seconds.
        merge: Whether to merge similar jobs.
        include_invalid: Also show jobs which can never run, i.e. jobs whose
            dependencies can never be satisfied.
        partition: Only show jobs and nodes in this partition.
        static: Print the full job list once and exit, instead of showing a
            scrollable view which refreshes.
    """
    console = Console()
    slurm_version = get_slurm_version()
    nodes = get_nodes(partition=partition)

    def fetch_jobs() -> List[Job]:
        """Fetches the job list, merging similar jobs if asked to."""
        jobs = get_jobs(include_invalid=include_invalid, partition=partition)
        return coalesce_jobs(jobs) if merge else jobs

    if static:
        # Printed directly, so that the list is left in the terminal.
        console.print(render(fetch_jobs(), nodes, slurm_version))
        return 0

    old_settings = None
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    jobs: List[Job] = []
    offset = 0
    last_fetch = float("-inf")

    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                if (time.monotonic() - last_fetch) >= refresh:
                    jobs = fetch_jobs()
                    last_fetch = time.monotonic()

                # The job list is re-windowed every frame, since both the
                # number of jobs and the size of the terminal can change.
                max_rows = max(
                    console.size.height
                    - _CHROME_ROWS
                    - _CONTROLS_ROWS
                    - len(nodes),
                    1,
                )
                max_offset = max(len(jobs) - max_rows, 0)
                offset = min(offset, max_offset)

                panel = render(
                    jobs, nodes, slurm_version, offset, max_rows, controls=True
                )
                live.update(panel, refresh=True)

                # Redraw on a keypress without refetching, so that scrolling
                # stays responsive no matter how slow the queue is to read.
                timeout = max(refresh - (time.monotonic() - last_fetch), 0.0)
                if not sys.stdin.isatty():
                    time.sleep(timeout)
                    continue

                key = read_key(timeout)
                if key is None:
                    continue
                if key in _KEY_QUIT:
                    break
                offset = scroll(key, offset, max_rows, max_offset)
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C
    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    return 0


def _cli() -> int:
    return tyro.cli(main)
