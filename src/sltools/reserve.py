"""Reserve resources using a "blocker" job."""

import getpass
import os
import subprocess
import time
from pathlib import Path

import tyro
import yaml
from rich.console import Console


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
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=reserved_%j.yaml
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
    script_path = Path("_reserve.sh")
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
    filename = Path(f"reserved_{job_id}.yaml")
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


def connect_to_node(node: str, gpus: str, console: Console):
    """SSHs into the node and sets up the environment."""
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
) -> int:
    """Reserve resources via blocking job.

    Args:
        partition: Slurm partition to use.
        gres: generic resources request (e.g. gpu:1).
        cpus: CPUs per task.
        mem: Memory limit.
        time_limit: Time limit for the reservation.
    """
    console = Console()
    user = getpass.getuser()

    job_info = check_existing_job(user)
    if job_info is not None:
        job_id, state = job_info
        if state == "PENDING":
            console.print(
                "[bold red]Error: A job is already queued. Please cancel this job or wait for it to run and try again.[/bold red]"
            )
            return 1

        console.print(f"[yellow]Found existing reservation job {job_id}[/yellow]")
        expected_file = Path(f"reserved_{job_id}.yaml")
        if not expected_file.exists():
            console.print(
                f"[bold red]Error: Job {job_id} exists but output file {expected_file} is missing.[/bold red]"
            )
            return 1
    else:
        console.print("[bold blue]Creating new reservation...[/bold blue]")
        script_path = create_sbatch_script(partition, gres, cpus, mem, time_limit)
        job_id = submit_job(script_path)
        console.print(f"Submitted job {job_id}")

    if job_id:
        try:
            yaml_file = wait_for_output_file(job_id, console)
        except KeyboardInterrupt:
            console.print(
                "[bold red]Wait interrupted; the job will remain queued.[/bold red]"
            )
            return 1

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
            connect_to_node(node, gpus, console)

    return 0


def _cli() -> int:
    return tyro.cli(main)
