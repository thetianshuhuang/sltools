# sltop

## Overview

`sltop` is a top-like queue viewer for slurm.

It should show information like this command:

```sh
$ alias squeue2
alias squeue2='squeue -o "%.6i %.5P %.36j %.8u %.2t %.9M %.2D %13R %b"'
$ squeue2
 JOBID PARTI                                 NAME     USER ST      TIME NO NODELIST(REAS TRES_PER_NODE
   115 batch test.conv_basic_4x_lpw0.5.p100_lr10x  tianshu PD      0:00  1 (Dependency)  gres/gpu:1
   111 batch        test.conv_basic_4x_lpw0.5.p20  tianshu PD      0:00  1 (Dependency)  gres/gpu:1
   109 batch        test.conv_basic_4x_lpw0.5.p50  tianshu PD      0:00  1 (Dependency)  gres/gpu:1
   107 batch       test.conv_basic_4x_lpw0.5.p100  tianshu PD      0:00  1 (Dependency)  gres/gpu:1
   114 batch train.conv_basic_4x_lpw0.5.p100_lr10  tianshu  R      6:41  1 b1            gres/gpu:4
   106 batch      train.conv_basic_4x_lpw0.5.p100  tianshu  R   9:53:04  1 b0            gres/gpu:4
   108 batch       train.conv_basic_4x_lpw0.5.p50  tianshu  R   9:53:04  1 b0            gres/gpu:4
   110 batch       train.conv_basic_4x_lpw0.5.p20  tianshu  R   9:53:04  1 b1            gres/gpu:4
```

## Environment

`sltop` should be managed using `uv` with `pyproject.toml`. It should have a command line interface written via tyro, and should be installable using `uv tool install`.

## Architecture

`sltop` should be structured as follows:

- A `Job` dataclass
- `get_jobs`: call `squeue --json`, and parse the output into a `list[Job]`.
- A `render` function that takes a `list[Job]` and renders it into rich output.
- A main loop that calls `get_jobs` and `render`.

## Interface

`sltop` should use `rich` to display a beautiful and colorful output with the job ID, name, user, partition, number of nodes and GPUs, nodelist, status, and execution time.
