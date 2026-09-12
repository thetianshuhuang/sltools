"""Accounting of which node resources are used by which jobs."""

from .jobs import Job
from .nodes import Node, expand_nodelist


def calculate_node_usage(nodes: list[Node], jobs: list[Job]) -> dict:
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

        # Determine resources used per node for this job: prefer what the job
        # explicitly requested per node, and otherwise spread its allocation
        # (which is a total across all of its nodes) evenly over them.
        res_per_node = job.get_resources_per_node()
        total = job.get_resources_total()

        def per_node(key: str, fallback: int = 0) -> int:
            if res_per_node.get(key, 0) > 0:
                return res_per_node[key]
            return max(total.get(key, 0), fallback) // num_nodes

        # 1. GPU: tres_per_node (GRES), or the "gres/gpu" part of the allocation
        gpus_alloc = per_node("gpu")

        # 2. CPU: explicit tres or distribute total
        cpus_alloc = per_node("cpu", job.cpus)

        # 3. Mem: explicit tres or distribute total
        mem_alloc = per_node("mem")

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
