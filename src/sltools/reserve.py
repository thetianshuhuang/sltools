"""Reserve resources using a "blocker" job."""

import getpass
import os
import subprocess
import time
from pathlib import Path

import tyro
import yaml
from rich.console import Console

SLTOOLS_DIR = Path(os.environ.get("SLTOOLS_DIR", Path.home() / ".sltools"))


def check_existing_job(user: str) -> tuple[str, str] | None:
    """Checks if a reservation job already exists for the user.

    Returns:
        A tuple of (job ID, state) if found, otherwise None.
    """
    try:
        job_name = "reserved"
        output = subprocess.check_output(
            [
                "squeue",
                f"--name={job_name}",
                f"--users={user}",
                "--states=R,PD",
                "--noheader",
                "--format=%A %T",
            ],
            text=True,
        ).strip()
        if output:
            parts = output.split()
            if len(parts) >= 2:
                return parts[0], parts[1]
            return parts[0], "UNKNOWN"
    except subprocess.CalledProcessError:
        pass
    return None


def create_sbatch_script(
    partition: str = "dev",
    gres: str = "gpu:1",
    cpus: int = 32,
    mem: str = "128G",
    time_limit: str = "08:00:00",
) -> Path:
    """Creates the sbatch script."""
    job_name = "reserved"
    output_path = SLTOOLS_DIR / "reserved_%j.yaml"
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_path}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}

echo "node: $(hostname)"
echo "gpus: $SLURM_JOB_GPUS"
echo "job_id: $SLURM_JOB_ID"

echo "# To use these GPUs, run:"
echo "# ssh $(hostname)"
echo "# export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS"

sleep infinity
"""
    script_path = SLTOOLS_DIR / "_reserve.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_path


def submit_job(script_path: Path) -> str:
    """Submits the job and returns the job ID."""
    output = subprocess.check_output(
        ["sbatch", "--parsable", str(script_path)], text=True
    ).strip()
    return output.split(";")[0]


def wait_for_output_file(job_id: str, console: Console) -> Path:
    """Waits for the output file reserved_<job_id>.yaml to appear."""
    filename = SLTOOLS_DIR / f"reserved_{job_id}.yaml"
    with console.status(f"[bold green]Waiting for job {job_id} to start..."):
        while not filename.exists():
            time.sleep(1)

    time.sleep(1)
    return filename


def read_job_info(yaml_file: Path) -> dict:
    """Reads the node and GPU info from the YAML file."""
    for _ in range(5):
        try:
            content = yaml_file.read_text()
            if not content.strip():
                time.sleep(1)
                continue
            data = yaml.safe_load(content)
            if data and "node" in data:
                return data
        except Exception:
            pass
        time.sleep(1)

    raise RuntimeError(f"Could not read valid YAML from {yaml_file}")


def handle_clean(console: Console) -> int:
    """Handles the --clean flag."""
    # Ensure SLTOOLS_DIR exists
    SLTOOLS_DIR.mkdir(parents=True, exist_ok=True)

    console.print(f"Cleaning up {SLTOOLS_DIR}...")
    for p in SLTOOLS_DIR.glob("reserved_*.yaml"):
        p.unlink()
        console.print(f"Removed {p}")
    script = SLTOOLS_DIR / "_reserve.sh"
    if script.exists():
        script.unlink()
        console.print(f"Removed {script}")
    console.print("[bold green]Clean complete.[/bold green]")
    return 0


def handle_cancel(job_info: tuple[str, str] | None, console: Console) -> int:
    """Handles the --cancel flag.

    Args:
        job_info: Existing job info (job_id, state) or None.
        console: The rich console.

    Returns:
        0 on success, non-zero on error.
    """
    if job_info is not None:
        job_id, _ = job_info
        try:
            subprocess.check_call(["scancel", job_id])
            console.print(f"[bold green]Cancelled job {job_id}[/bold green]")
            return 0
        except subprocess.CalledProcessError:
            console.print(f"[bold red]Error: Failed to cancel job {job_id}[/bold red]")
            return 1
    else:
        console.print("[bold red]Error: No reservation job found to cancel.[/bold red]")
        return -1


def setup_node(job_id: str, console: Console) -> None:
    """Waits for job start and sets up the environment on the node.

    This function does NOT return on success (it execs a new shell).

    Args:
        job_id: The Slurm job ID.
        console: The rich console.
    """
    try:
        yaml_file = wait_for_output_file(job_id, console)
    except KeyboardInterrupt:
        console.print(
            "[bold red]Wait interrupted; the job will remain queued.[/bold red]"
        )
        return

    info = read_job_info(yaml_file)
    node = info["node"]
    gpus = str(info.get("gpus", ""))

    current_node = subprocess.check_output(["hostname"], text=True).strip()

    if current_node == node:
        console.print(f"[bold green]Already on reserved node {node}.[/bold green]")
        console.print(f"Setting CUDA_VISIBLE_DEVICES={gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        shell = os.environ.get("SHELL", "/bin/bash")
        os.execlp(shell, shell, "--login")
    else:
        console.print(f"[bold green]Allocated Node:[/bold green] {node}")
        console.print(f"[bold green]Allocated GPUs:[/bold green] {gpus}")

        shell = os.environ.get("SHELL", "/bin/bash")
        remote_cmd = f"export CUDA_VISIBLE_DEVICES={gpus}; exec {shell} --login"

        console.print(f"[dim]Connecting to {node}...[/dim]")

        ssh_args = ["ssh", "-t", node, remote_cmd]
        os.execlp("ssh", *ssh_args)


def main(
    partition: str = "dev",
    gres: str = "gpu:1",
    cpus: int = 32,
    mem: str = "128G",
    time_limit: str = "08:00:00",
    cancel: bool = False,
    clean: bool = False,
) -> int:
    """Reserve resources via blocking job.

    Args:
        partition: Slurm partition to use.
        gres: Generic resources, e.g. GPUs.
        cpus: CPUs per task.
        mem: Memory limit.
        time_limit: Time limit for the reservation.
        cancel: Cancel the existing reservation job.
        clean: Clean up all reservation files from ~/.sltools.
    """
    console = Console()

    if clean:
        return handle_clean(console)

    user = getpass.getuser()
    job_info = check_existing_job(user)

    if cancel:
        return handle_cancel(job_info, console)

    if job_info is not None:
        job_id, state = job_info
        if state == "PENDING":
            console.print(
                "[bold red]Error: A job is already queued. Please cancel this job or wait for it to run and try again.[/bold red]"
            )
            return 1

        console.print(f"[yellow]Found existing reservation job {job_id}[/yellow]")
        expected_file = SLTOOLS_DIR / f"reserved_{job_id}.yaml"
        if not expected_file.exists():
            console.print(
                f"[bold red]Error: Job {job_id} exists but output file {expected_file} is missing.[/bold red]"
            )
            return 1
    else:
        # Ensure SLTOOLS_DIR exists before creating script
        SLTOOLS_DIR.mkdir(parents=True, exist_ok=True)
        console.print("[bold blue]Creating new reservation...[/bold blue]")
        script_path = create_sbatch_script(partition, gres, cpus, mem, time_limit)
        job_id = submit_job(script_path)
        console.print(f"Submitted job {job_id}")
        console.print(
            f"The reservation is saved to: {SLTOOLS_DIR / f'reserved_{job_id}.yaml'}"
        )

    if job_id:
        setup_node(job_id, console)
        # If setup_node returns, it means something like KeyboardInterrupt happened
        return 1

    return 0


def _cli() -> int:
    return tyro.cli(main)
