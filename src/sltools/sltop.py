"""Main entrypoint and UI rendering for sltop."""

import datetime
import os
import re
import select
import sys
import termios
import time
import tty
from typing import Annotated, List

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
_KEY_INFO = ("i", "I")

# Escape sequence sent by a single special key, e.g. "\x1b[A" for the up arrow.
_ESCAPE_SEQUENCE = re.compile(r"\x1b(\[[0-9;]*[~A-Za-z]|O[A-Za-z]|.)")

# Rows the panel uses for everything except the job list itself: the two panel
# borders, the header, the two rules, the node table's header, and the job
# table's header. The node rows are counted separately, since they vary.
_CHROME_ROWS = 7

# The interactive view also has a rule and the controls bar below the jobs.
_CONTROLS_ROWS = 2

# Lines of detail shown for the selected job, plus the rule above them.
_INFO_LINES = 4
_INFO_ROWS = _INFO_LINES + 1

# Style of the row the cursor is on, which inverts the row into a solid bar.
# The bold is switched off explicitly, since a row style overrides the colour
# of a column but merges with its attributes.
_CURSOR_STYLE = "not bold black on white"

# Controls listed by the bar at the bottom of the interactive view.
_CONTROLS = (
    ("i", "info"),
    ("\u2191/\u2193 j/k", "select"),
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


def render_job_info(job: Job | None) -> Text:
    """Renders the detail pane for the selected job.

    Args:
        job: The selected job, or None if the list is empty.

    Returns:
        Text of exactly `_INFO_LINES` lines, so that the pane does not change
        height as the selection moves.
    """
    lines = []
    if job is not None:
        name = job.combined_name if isinstance(job, JobGroup) else job.name
        job_id = (
            job.job_id_str if isinstance(job, JobGroup) else str(job.job_id)
        )

        # A job which is waiting reports how long it has been waiting, and what
        # it asked for rather than what it holds.
        if job.job_state == "PENDING":
            waited = f"queued {job.time_queued}"
        else:
            waited = f"elapsed {job.time_used}"

        if job.tres_alloc:
            label = "alloc"
            resources = job.get_resources_total()
            nodes = job.nodelist
        else:
            label = "req"
            resources = job.get_resources_requested()
            nodes = job.req_nodes

        held = (
            f"{resources.get('cpu', 0)} cpus"
            f"   {resources.get('gpu', 0)} gpus"
            f"   {resources.get('mem', 0) // 1024} GB"
        )
        if nodes:
            held += f"   on {nodes}"

        lines = [
            ("job", f"{job_id}  {job.partition}  {name}"),
            ("cmd", f"{job.user_name}  {job.command or '-'}"),
            (
                "state",
                f"{job.job_state} ({job.state_reason})"
                f"   nice {job.nice}   {waited}",
            ),
            (label, held),
        ]

    info = Text(no_wrap=True, overflow="ellipsis")
    for index in range(_INFO_LINES):
        label, value = lines[index] if index < len(lines) else ("", "")
        if index:
            info.append("\n")
        info.append(f"{label:>6}  ", style="bold cyan")
        info.append(value, style="white")
    return info


def render_bottom_bar(position: str) -> Table:
    """Renders the bar of controls, with the scroll position on the left.

    Args:
        position: Which jobs are on screen, e.g. "21-30 of 57", or an empty
            string when the whole list fits.
    """
    controls = Text()
    for index, (keys, action) in enumerate(_CONTROLS):
        if index:
            controls.append("    ")
        controls.append(keys, style="bold cyan")
        controls.append(f" {action}", style="dim")

    # The controls are centred in the bar itself, so the side columns share
    # the space left over equally. The right one is reserved for later.
    bar = Table.grid(expand=True)
    bar.add_column(justify="left", ratio=1, no_wrap=True)
    bar.add_column(justify="center", no_wrap=True)
    bar.add_column(justify="right", ratio=1, no_wrap=True)
    bar.add_row(Text(position, style="bold orange1"), controls, Text(""))
    return bar


def render(
    jobs: List[Job],
    nodes: List[Node],
    slurm_version: str,
    offset: int = 0,
    max_rows: int | None = None,
    controls: bool = False,
    cursor: int | None = None,
    info: bool = False,
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
        cursor: Index of the selected job, which is highlighted, or None for
            no selection.
        info: Whether to show the detail pane for the selected job.
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
    header_grid = Table.grid(expand=True)
    header_grid.add_column(justify="left")
    header_grid.add_column(justify="right")
    header_grid.add_row(title, Text(now_str, style="bold white"))

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

    for index, job in enumerate(visible, start=offset):
        # The cursor row is drawn as one solid bar, so its cells drop the
        # colours they would otherwise be given: a row style overrides the
        # style of a column, but not the style of a cell inside it.
        selected = index == cursor

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
        part_nice.append(str(job.nice), style="" if selected else nice_style)

        table.add_row(
            escape(job_id_display),
            part_nice,
            job.user_name,
            escape(job_name_display),
            Text(st_code, style="" if selected else st_style),
            job.time_used,
            escape(format_resources(job)),
            style=_CURSOR_STYLE if selected else None,
        )

    # Combine sections: Header -> Rule -> Nodes -> Rule -> Jobs [-> Controls]
    sections = [
        Padding(header_grid, (0, 1)),
        Rule(style="dim"),
        node_table,
        Rule(style="dim"),
        table,
    ]
    if info:
        selected = jobs[cursor] if cursor is not None and jobs else None
        sections += [
            Rule(style="dim"),
            Padding(render_job_info(selected), (0, 1)),
        ]
    if controls:
        position = ""
        if len(visible) < len(jobs):
            position = f"{offset + 1}-{offset + len(visible)} of {len(jobs)}"
        sections += [
            Rule(style="dim"),
            Padding(render_bottom_bar(position), (0, 1)),
        ]

    return Panel(Group(*sections), box=box.ROUNDED, padding=0)


def split_keys(data: str) -> list[str]:
    """Splits a burst of terminal input into individual keys.

    Args:
        data: Raw bytes read from the terminal, decoded.

    Returns:
        One entry per key, with escape sequences kept whole.
    """
    keys = []
    while data:
        match = _ESCAPE_SEQUENCE.match(data) if data[0] == "\x1b" else None
        length = match.end() if match else 1
        keys.append(data[:length])
        data = data[length:]
    return keys


def read_keys(timeout: float) -> list[str]:
    """Waits up to `timeout` seconds for keypresses.

    Returns:
        The keys pressed, which is more than one if they were typed faster than
        the screen refreshes, e.g. when a key repeats. Empty if nothing was
        pressed before the timeout.
    """
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return []

    return split_keys(os.read(sys.stdin.fileno(), 64).decode(errors="ignore"))


def move_cursor(
    key: str, cursor: int, offset: int, page: int, total: int
) -> tuple[int, int]:
    """Returns the cursor and scroll offset after a keypress.

    The cursor moves within the visible rows, and the list only scrolls once
    the cursor would leave them.

    Args:
        key: The key pressed, as returned by `read_key`.
        cursor: Index of the currently selected job.
        offset: Index of the first visible job.
        page: Number of jobs on screen.
        total: Number of jobs in the list.

    Returns:
        The new (cursor, offset), both clamped to the list.
    """
    if key in _KEY_UP:
        cursor -= 1
    elif key in _KEY_DOWN:
        cursor += 1
    elif key in _KEY_PAGE_UP:
        cursor -= page
    elif key in _KEY_PAGE_DOWN:
        cursor += page
    elif key in _KEY_HOME:
        cursor = 0
    elif key in _KEY_END:
        cursor = total - 1

    cursor = max(0, min(cursor, total - 1))

    # Scroll only as far as is needed to keep the cursor on screen.
    offset = min(offset, cursor)
    offset = max(offset, cursor - page + 1)
    return cursor, max(0, min(offset, max(total - page, 0)))


def main(
    refresh: Annotated[float, tyro.conf.arg(aliases=["-r"])] = 1.0,
    merge: Annotated[bool, tyro.conf.arg(aliases=["-m"])] = False,
    include_invalid: Annotated[bool, tyro.conf.arg(aliases=["-i"])] = False,
    partition: Annotated[str | None, tyro.conf.arg(aliases=["-p"])] = None,
    static: Annotated[bool, tyro.conf.arg(aliases=["-s"])] = False,
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
    cursor = 0
    offset = 0
    show_info = False
    last_fetch = float("-inf")

    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                if (time.monotonic() - last_fetch) >= refresh:
                    jobs = fetch_jobs()
                    last_fetch = time.monotonic()

                # The job list is re-windowed every frame, since both the
                # number of jobs and the size of the terminal can change.
                chrome = _CHROME_ROWS + _CONTROLS_ROWS + len(nodes)
                if show_info:
                    chrome += _INFO_ROWS
                max_rows = max(console.size.height - chrome, 1)

                # Re-clamp, since the list and the terminal both change size.
                cursor = max(0, min(cursor, len(jobs) - 1))
                offset = max(0, min(offset, max(len(jobs) - max_rows, 0)))
                cursor, offset = move_cursor(
                    "", cursor, offset, max_rows, len(jobs)
                )

                panel = render(
                    jobs,
                    nodes,
                    slurm_version,
                    offset,
                    max_rows,
                    controls=True,
                    cursor=cursor,
                    info=show_info,
                )
                live.update(panel, refresh=True)

                # Redraw on a keypress without refetching, so that scrolling
                # stays responsive no matter how slow the queue is to read.
                timeout = max(refresh - (time.monotonic() - last_fetch), 0.0)
                if not sys.stdin.isatty():
                    time.sleep(timeout)
                    continue

                keys = read_keys(timeout)
                if any(key in _KEY_QUIT for key in keys):
                    break

                for key in keys:
                    if key in _KEY_INFO:
                        show_info = not show_info
                    else:
                        cursor, offset = move_cursor(
                            key, cursor, offset, max_rows, len(jobs)
                        )
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C
    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    return 0


def _cli() -> int:
    # Without FlagCreatePairsOff, every boolean alias would also generate an
    # inverse flag, and those all collide with each other.
    return tyro.cli(main, config=(tyro.conf.FlagCreatePairsOff,))
