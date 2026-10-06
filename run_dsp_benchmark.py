"""
Benchmark: MSRP, MrsP, DSP-S, DSP-F.
Builds tasks/resources, runs each method, collects schedulability and busy-wait stats,
and plots comparison. Run repeatedly (e.g. num_runs) to get aggregate results.
"""
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scheduling_base
from scheduling_base import (
    Core,
    Task,
    Resource,
    Job,
    plt,
    compute_hyperperiods,
    reset,
    unifast,
    assign_resources_with_local_constraints,
    simulate_msrp,
    simulate_mrsp,
    simulate_msrp_with_secondary_queue,
    simulate_dsp_s,
)

# ---------------------------------------------------------------------------
# Parameters (tune for batch runs)
# ---------------------------------------------------------------------------

num_tasks_per_core = 15
periods = [100, 200, 500, 1000, 2000, 5000]
RSP = 0.5
MAX_ACCESS = 4

cores_list = [4, 8, 16]
utilizations = [0.1, 0.25, 0.5, 0.75]
num_runs = 200


def build_cores_and_resources(num_cores, total_utilization, seed=None):
    """Build fresh cores and resources for one experiment. Optional seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
    num_resources = int(1.5 * num_cores)
    resources = [Resource(i) for i in range(num_resources)]
    cores = []
    for core_id in range(num_cores):
        core = Core(core_id)
        task_utilizations = unifast(num_tasks_per_core, total_utilization)
        for task_id, utilization in enumerate(task_utilizations):
            period = random.choice(periods)
            task = Task(task_id, period, utilization)
            core.add_task(task)
        core.assign_preemption_level()
        cores.append(core)
    assign_resources_with_local_constraints(cores, resources, RSP, MAX_ACCESS)
    for core in cores:
        core.build_resource_ceiling_table(resources)
    return cores, resources


def _total_busy(cores):
    return sum(c.busy_waiting_time for c in cores)


def run_one_simulation(cores, resources):
    """
    Run MSRP, MrsP, DSP-S, DSP-F on the same (cores, resources).
    Returns schedulable (bool) and total busy_waiting_time for each.
    """
    # MSRP
    miss_msrp = simulate_msrp(cores, resources)
    busy_msrp = _total_busy(cores)
    reset(cores)

    # MrsP: need MSRP-style ceiling first for initial state; then rebuild for MrsP
    for core in cores:
        core.resource_ceiling_table = {}
        core.build_resource_ceiling_table_mrsp(resources)
    miss_mrsp = simulate_mrsp(cores, resources)
    busy_mrsp = _total_busy(cores)
    reset(cores)

    # Restore MSRP ceiling for DSP protocols
    for core in cores:
        core.resource_ceiling_table = {}
        core.build_resource_ceiling_table(resources)

    # DSP-S
    miss_dsp_s = simulate_dsp_s(cores, resources)
    busy_dsp_s = _total_busy(cores)
    reset(cores)

    # DSP-F (msrp_with_secondary_queue)
    miss_dsp_f = simulate_msrp_with_secondary_queue(cores, resources)
    busy_dsp_f = _total_busy(cores)

    return (
        not miss_msrp,
        not miss_mrsp,
        not miss_dsp_s,
        not miss_dsp_f,
        busy_msrp,
        busy_mrsp,
        busy_dsp_s,
        busy_dsp_f,
    )


def main():
    scheduling_base.SIMULATE_QUIET = True
    results = {
        "msrp": {},
        "mrsp": {},
        "dsp_s": {},
        "dsp_f": {},
    }
    busy_avg = {
        "msrp": {},
        "mrsp": {},
        "dsp_s": {},
        "dsp_f": {},
    }

    for ncores in cores_list:
        for util in utilizations:
            key = (ncores, util)
            sched_msrp = sched_mrsp = sched_dsp_s = sched_dsp_f = 0
            busy_msrp_sum = busy_mrsp_sum = busy_dsp_s_sum = busy_dsp_f_sum = 0

            for run in range(num_runs):
                cores, resources = build_cores_and_resources(ncores, util)
                (
                    ok_msrp,
                    ok_mrsp,
                    ok_dsp_s,
                    ok_dsp_f,
                    b_msrp,
                    b_mrsp,
                    b_dsp_s,
                    b_dsp_f,
                ) = run_one_simulation(cores, resources)
                if ok_msrp:
                    sched_msrp += 1
                if ok_mrsp:
                    sched_mrsp += 1
                if ok_dsp_s:
                    sched_dsp_s += 1
                if ok_dsp_f:
                    sched_dsp_f += 1
                busy_msrp_sum += b_msrp
                busy_mrsp_sum += b_mrsp
                busy_dsp_s_sum += b_dsp_s
                busy_dsp_f_sum += b_dsp_f

            results["msrp"][key] = sched_msrp / num_runs
            results["mrsp"][key] = sched_mrsp / num_runs
            results["dsp_s"][key] = sched_dsp_s / num_runs
            results["dsp_f"][key] = sched_dsp_f / num_runs
            busy_avg["msrp"][key] = busy_msrp_sum / num_runs
            busy_avg["mrsp"][key] = busy_mrsp_sum / num_runs
            busy_avg["dsp_s"][key] = busy_dsp_s_sum / num_runs
            busy_avg["dsp_f"][key] = busy_dsp_f_sum / num_runs

            print(
                f"Cores: {ncores}, Util: {util:.2f} | "
                f"MSRP: {results['msrp'][key]:.2f} (busy_avg={busy_avg['msrp'][key]:.0f}) | "
                f"MrsP: {results['mrsp'][key]:.2f} (busy_avg={busy_avg['mrsp'][key]:.0f}) | "
                f"DSP-S: {results['dsp_s'][key]:.2f} (busy_avg={busy_avg['dsp_s'][key]:.0f}) | "
                f"DSP-F: {results['dsp_f'][key]:.2f} (busy_avg={busy_avg['dsp_f'][key]:.0f})"
            )

    if plt is not None:
        for ncores in cores_list:
            plt.figure(figsize=(12, 6))
            xs = [util for util in utilizations]
            plt.plot(xs, [results["msrp"][(ncores, u)] for u in utilizations], marker="o", label="MSRP")
            plt.plot(xs, [results["mrsp"][(ncores, u)] for u in utilizations], marker="s", label="MrsP")
            plt.plot(xs, [results["dsp_s"][(ncores, u)] for u in utilizations], marker="^", label="DSP-S")
            plt.plot(xs, [results["dsp_f"][(ncores, u)] for u in utilizations], marker="d", label="DSP-F")
            plt.xlabel("Total Utilization per Core")
            plt.ylabel("Fraction Schedulable")
            plt.title(f"Schedulability (cores={ncores})")
            plt.legend()
            plt.grid(True)
            plt.show()

        for ncores in cores_list:
            plt.figure(figsize=(12, 6))
            xs = [util for util in utilizations]
            plt.plot(xs, [busy_avg["msrp"][(ncores, u)] for u in utilizations], marker="o", label="MSRP")
            plt.plot(xs, [busy_avg["mrsp"][(ncores, u)] for u in utilizations], marker="s", label="MrsP")
            plt.plot(xs, [busy_avg["dsp_s"][(ncores, u)] for u in utilizations], marker="^", label="DSP-S")
            plt.plot(xs, [busy_avg["dsp_f"][(ncores, u)] for u in utilizations], marker="d", label="DSP-F")
            plt.xlabel("Total Utilization per Core")
            plt.ylabel("Average Total Busy Waiting (all cores)")
            plt.title(f"Busy Waiting (cores={ncores})")
            plt.legend()
            plt.grid(True)
            plt.show()
    else:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
