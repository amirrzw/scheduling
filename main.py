import random
import math
from functools import reduce
import heapq
import matplotlib.pyplot as plt



class Task:
    def __init__(self, id, period, utilization):
        self.id = id
        self.period = period
        self.utilization = utilization
        self.execution_time = max(1, int(utilization * period))
        self.deadline = period
        self.preemption_level = None
        self.resource_intervals = {}

    def __repr__(self):
        total_resource_access_time = sum(
            sum(end - start for start, end in intervals) for intervals in self.resource_intervals.values()
        )
        return (f"Task(id={self.id}, period={self.period}, utilization={self.utilization:.4f}, "
                f"execution_time={self.execution_time}, deadline={self.deadline}, preemption_level={self.preemption_level}, "
                f"resource_intervals={self.resource_intervals}), x:{total_resource_access_time / self.execution_time:.2f}")


class Resource:
    def __init__(self, id):
        self.id = id
        self.accessed_by = set()  # Keep track of which cores/tasks access this resource
        self.type = None  # Local or Global

    def determine_type(self):
        self.type = "Local" if len(self.accessed_by) == 1 else "Global"

    def __repr__(self):
        return f"Resource(id={self.id}, type={self.type}), {self.accessed_by}"


class Job:
    def __init__(self, task, release_time):
        self.task_id = task.id
        self.remaining_time = task.execution_time
        self.execution_time = task.execution_time
        self.deadline = release_time + task.deadline
        self.preemption_level = task.preemption_level
        self.resource_intervals = task.resource_intervals.copy()
        self.current_interval_index = {res_id: 0 for res_id in self.resource_intervals}

    def __lt__(self, other):
        return self.deadline < other.deadline


class Core:
    def __init__(self, id):
        self.id = id
        self.tasks = []
        self.job_queue = []  # Min-heap based on job deadlines
        self.current_job = None
        self.system_ceiling = 0
        self.resource_ceiling_table = {}
        self.locked_resources = set()
        self.hyperperiod = None
        self.busy_waiting_time = 0  # Total busy waiting time

    def add_task(self, task):
        self.tasks.append(task)

    def assign_preemption_level(self):
        self.tasks.sort(key=lambda t: t.deadline, reverse=True)
        preemption_level = 1
        for i, task in enumerate(self.tasks):
            if i > 0 and task.deadline == self.tasks[i - 1].deadline:
                task.preemption_level = self.tasks[i - 1].preemption_level
            else:
                task.preemption_level = preemption_level
                preemption_level += 1

    def build_resource_ceiling_table(self, resources):
        max_preemption_level = max([task.preemption_level for task in self.tasks])
        for task in self.tasks:
            for res_id in task.resource_intervals:
                if resources[res_id].type == "Global":
                    self.resource_ceiling_table[res_id] = max_preemption_level
                else:
                    ceiling = self.resource_ceiling_table.get(res_id, 0)
                    self.resource_ceiling_table[res_id] = max(ceiling, task.preemption_level)

    def build_resource_ceiling_table_mrsp(self, resources):
        for task in self.tasks:
            for res_id in task.resource_intervals:
                ceiling = self.resource_ceiling_table.get(res_id, 0)
                self.resource_ceiling_table[res_id] = max(ceiling, task.preemption_level)

    def calculate_system_ceiling(self):
        return max(
            [self.resource_ceiling_table.get(res_id, 0) for res_id in self.locked_resources], default=0
        )

    def __repr__(self):
        return f"Core(id={self.id}, tasks={self.tasks})"


def reset_cores_and_resources(cores, resources):
    for core in cores:
        core.job_queue = []
        core.secondary_queue = []
        core.current_job = None
        core.busy_waiting_time = 0
        core.locked_resources = set()
        core.system_ceiling = 0


def unifast(num_tasks, total_utilization):
    """Distribute the total utilization across num_tasks using the Unifast algorithm."""
    utilizations = []
    sum_u = total_utilization
    for i in range(1, num_tasks):
        next_sum_u = sum_u * (random.uniform(0, 1) ** (1 / (num_tasks - i)))
        utilizations.append(sum_u - next_sum_u)
        sum_u = next_sum_u
    utilizations.append(sum_u)
    return utilizations


def slice_execution_time(execution_time, method="method_a"):
    if method == "method_a":
        num_chunks = random.randint(2, 4)
    elif method == "method_b":
        if execution_time <= 5:
            min_slices, max_slices = 1, 3
        elif execution_time <= 10:
            min_slices, max_slices = 3, 5
        elif execution_time <= 99:
            min_slices, max_slices = 5, 15
        else:
            min_slices, max_slices = 15, 30
        num_chunks = random.randint(min_slices, max_slices)
    else:
        raise ValueError("Invalid slicing method. Use 'method_a' or 'method_b'.")

    # Ensure we don't request more chunks than the available time units
    if execution_time < num_chunks:
        num_chunks = execution_time

    if num_chunks <= 1:
        return [(0, execution_time)]

    # Generate num_chunks - 1 unique cut points between 1 and execution_time - 1
    cut_points = sorted(random.sample(range(1, execution_time), num_chunks - 1))

    # Determine the lengths of each piece
    pieces = []
    prev = 0
    for cp in cut_points:
        pieces.append(cp - prev)
        prev = cp
    pieces.append(execution_time - prev)

    # Build intervals from the pieces
    intervals = []
    start = 0
    for piece in pieces:
        intervals.append((start, start + piece))
        start += piece
    return intervals


def assign_resources_with_local_constraints(cores, resources, rsp, max_access):
    num_local_resources = len(cores)

    global_resources = resources[num_local_resources:]
    for i in range(num_local_resources):
        resources[i].type = "Local"
        resources[i].accessed_by.add(i % len(cores))

    for core_id, core in enumerate(cores):
        for task in core.tasks:
            if random.random() < 0.1:
                continue
            remaining_rsp_time = int(rsp * task.execution_time)
            total_execution_time = task.execution_time

            intervals = slice_execution_time(total_execution_time, method="method_a")

            random.shuffle(intervals)  # Shuffle intervals to spread access across runtime

            resource_usage_count = {resource.id: 0 for resource in resources}
            task.resource_intervals = {}

            for start, end in intervals:
                if remaining_rsp_time <= 0:
                    break
                if random.random() < 0.7:
                    local_resources = [res for res in resources[:num_local_resources] if core_id in res.accessed_by]
                    resource = random.choice(local_resources) if local_resources else random.choice(global_resources)
                else:
                    resource = random.choice(global_resources)

                interval_length = end - start
                if resource_usage_count[resource.id] < max_access and interval_length <= remaining_rsp_time:
                    if resource.id not in task.resource_intervals:
                        task.resource_intervals[resource.id] = []
                    task.resource_intervals[resource.id].append((start, end))
                    resource_usage_count[resource.id] += 1
                    remaining_rsp_time -= interval_length
                    resource.accessed_by.add(core_id)

            for resource_id in task.resource_intervals:
                task.resource_intervals[resource_id].sort(key=lambda x: x[0])

    for resource in resources[num_local_resources:]:
        resource.determine_type()


def lcm(numbers):
    return reduce(lambda a, b: a * b // math.gcd(a, b), numbers)


def compute_hyperperiods(cores):
    for core in cores:
        periods_list = [task.period for task in core.tasks]
        core.hyperperiod = lcm(periods_list)



def simulate_msrp(cores, resources):
    global_resource_locks = {}
    miss_occurred = False
    compute_hyperperiods(cores)
    time = 0
    max_hyperperiod = max(core.hyperperiod for core in cores)
    while time < max_hyperperiod:
        for core in cores:

            if core.current_job:
                for res_id, intervals in core.current_job.resource_intervals.items():
                    idx = core.current_job.current_interval_index[res_id]
                    if idx < len(intervals):
                        start, end = intervals[idx]
                        if end == (core.current_job.execution_time - core.current_job.remaining_time):
                            core.locked_resources.discard(res_id)
                            if resources[res_id].type == "Global":
                                global_resource_locks.pop(res_id, None)
                            core.system_ceiling = core.calculate_system_ceiling()
                            core.current_job.current_interval_index[res_id] += 1

            # Release new jobs
            for task in core.tasks:
                if time % task.period == 0:
                    job = Job(task, release_time=time)
                    heapq.heappush(core.job_queue, job)

            # Remove completed jobs
            if core.current_job and core.current_job.remaining_time <= 0:
                core.current_job = None

            # Check if we can schedule a new job or preempt the current one
            if core.job_queue:
                candidate_job = heapq.heappop(core.job_queue)

                if core.current_job:
                    if candidate_job.preemption_level > core.system_ceiling and candidate_job.preemption_level > core.current_job.preemption_level:
                        heapq.heappush(core.job_queue, core.current_job)
                        core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)
                else:
                    if candidate_job.preemption_level > core.system_ceiling:
                        core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)

            # Execute current job if any
            if core.current_job:
                has_busy_waiting = False
                for res_id, intervals in core.current_job.resource_intervals.items():
                    idx = core.current_job.current_interval_index[res_id]
                    if idx < len(intervals):
                        start, end = intervals[idx]
                        if start == (core.current_job.execution_time - core.current_job.remaining_time):
                            can_lock = True
                            if resources[res_id].type == "Global":
                                if res_id in global_resource_locks:
                                    can_lock = False
                            if can_lock:
                                core.locked_resources.add(res_id)
                                if resources[res_id].type == "Global":
                                    global_resource_locks[res_id] = core.id
                                core.system_ceiling = core.calculate_system_ceiling()
                            else:
                                core.busy_waiting_time += 1
                                has_busy_waiting = True
                                break

                if not has_busy_waiting:
                    core.current_job.remaining_time -= 1

                if time > core.current_job.deadline:
                    miss_occurred = True
                    break
        if miss_occurred:
            break
        time += 1

    for core in cores:
        print(f"msrp, Core {core.id} total busy waiting time: {core.busy_waiting_time}")
        for job in core.job_queue:
            if job.deadline < time:
                miss_occurred = True
                break
    return miss_occurred


def simulate_custom(cores, resources):
    miss_occurred = False
    global_resource_locks_custom = {}
    compute_hyperperiods(cores)
    time = 0
    max_hyperperiod = max(core.hyperperiod for core in cores)

    for core in cores:
        core.secondary_queue = []

    while time < max_hyperperiod:
        for core in cores:

            if core.current_job:
                for res_id, intervals in core.current_job.resource_intervals.items():
                    idx = core.current_job.current_interval_index[res_id]
                    if idx < len(intervals):
                        start, end = intervals[idx]
                        if end == (core.current_job.execution_time - core.current_job.remaining_time):
                            core.locked_resources.discard(res_id)
                            if resources[res_id].type == "Global":
                                global_resource_locks_custom.pop(res_id, None)
                                for other_core in cores:
                                    for sec_job in list(other_core.secondary_queue):
                                        task = other_core.tasks[sec_job.task_id]
                                        global_res = [
                                            r_id for r_id in task.resource_intervals
                                            if resources[r_id].type == "Global"
                                        ]
                                        if res_id in global_res:
                                            other_core.secondary_queue.remove(sec_job)
                                            heapq.heappush(other_core.job_queue, sec_job)
                            core.system_ceiling = core.calculate_system_ceiling()
                            core.current_job.current_interval_index[res_id] += 1

            for task in core.tasks:
                if time % task.period == 0:
                    job = Job(task, release_time=time)
                    heapq.heappush(core.job_queue, job)


            if core.current_job and core.current_job.remaining_time <= 0:
                core.current_job = None

            while core.job_queue:
                candidate_job = heapq.heappop(core.job_queue)
                task = core.tasks[candidate_job.task_id]
                global_res = [
                    res_id
                    for res_id in task.resource_intervals
                    if resources[res_id].type == "Global"
                ]
                # our algorithm
                if any(res_id in global_resource_locks_custom for res_id in global_res):
                    core.secondary_queue.append(candidate_job)
                    continue

                # todo: remove this -> check it
                if core.current_job:
                    if candidate_job.preemption_level > core.system_ceiling and candidate_job.preemption_level > core.current_job.preemption_level:
                        heapq.heappush(core.job_queue, core.current_job)
                        core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)
                else:
                    if candidate_job.preemption_level > core.system_ceiling:
                        core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)
                break

            if core.current_job:
                has_busy_waiting = False
                for res_id, intervals in core.current_job.resource_intervals.items():
                    idx = core.current_job.current_interval_index[res_id]
                    if idx < len(intervals):
                        start, end = intervals[idx]
                        if start == (core.current_job.execution_time - core.current_job.remaining_time):
                            can_lock = True
                            if resources[res_id].type == "Global":
                                if res_id in global_resource_locks_custom:
                                    can_lock = False
                            if can_lock:
                                core.locked_resources.add(res_id)
                                if resources[res_id].type == "Global":
                                    global_resource_locks_custom[res_id] = core.id
                                core.system_ceiling = core.calculate_system_ceiling()
                            else:
                                core.busy_waiting_time += 1
                                has_busy_waiting = True
                                break

                if not has_busy_waiting:
                    core.current_job.remaining_time -= 1

                if time > core.current_job.deadline:
                    miss_occurred = True
                    break

        if miss_occurred:
            break
        time += 1

    for core in cores:
        print(f"custom, Core {core.id} total busy waiting time: {core.busy_waiting_time}")
        for job in core.job_queue + core.secondary_queue:
            if time > job.deadline:
                miss_occurred = True
                break
    return miss_occurred


def get_resource(core, resources, global_resource_locks_custom):
    for res_id, intervals in core.current_job.resource_intervals.items():
        idx = core.current_job.current_interval_index[res_id]
        if idx < len(intervals):
            start, end = intervals[idx]
            if start == (core.current_job.execution_time - core.current_job.remaining_time):
                can_lock = True
                if resources[res_id].type == "Global":
                    if res_id in global_resource_locks_custom:
                        can_lock = False
                if can_lock:
                    core.locked_resources.add(res_id)
                    if resources[res_id].type == "Global":
                        global_resource_locks_custom[res_id] = core.id
                    core.system_ceiling = core.calculate_system_ceiling()
                else:
                    core.secondary_queue.append(core.current_job)
                    core.current_job = None
                    if core.job_queue:
                        candidate_job = heapq.heappop(core.job_queue)
                        core.current_job = candidate_job
                        get_resource(core, resources, global_resource_locks_custom)
                    return



def simulate_custom2(cores, resources):
    miss_occurred = False
    global_resource_locks_custom = {}
    compute_hyperperiods(cores)
    time = 0
    max_hyperperiod = max(core.hyperperiod for core in cores)

    for core in cores:
        core.secondary_queue = []

    while time < max_hyperperiod:
        for core in cores:

            if core.current_job:
                for res_id, intervals in core.current_job.resource_intervals.items():
                    idx = core.current_job.current_interval_index[res_id]
                    if idx < len(intervals):
                        start, end = intervals[idx]
                        if end == (core.current_job.execution_time - core.current_job.remaining_time):
                            core.locked_resources.discard(res_id)
                            if resources[res_id].type == "Global":
                                global_resource_locks_custom.pop(res_id, None)
                                for other_core in cores:
                                    for sec_job in list(other_core.secondary_queue):
                                        task = other_core.tasks[sec_job.task_id]
                                        global_res = [
                                            r_id for r_id in task.resource_intervals
                                            if resources[r_id].type == "Global"
                                        ]
                                        if res_id in global_res:
                                            other_core.secondary_queue.remove(sec_job)
                                            heapq.heappush(other_core.job_queue, sec_job)
                            core.system_ceiling = core.calculate_system_ceiling()
                            core.current_job.current_interval_index[res_id] += 1

            for task in core.tasks:
                if time % task.period == 0:
                    job = Job(task, release_time=time)
                    heapq.heappush(core.job_queue, job)


            if core.current_job and core.current_job.remaining_time <= 0:
                core.current_job = None

            while core.job_queue:
                candidate_job = heapq.heappop(core.job_queue)

                # todo: remove this -> check it
                if core.current_job:
                    if candidate_job.preemption_level > core.system_ceiling and candidate_job.preemption_level > core.current_job.preemption_level:
                        heapq.heappush(core.job_queue, core.current_job)
                        core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)
                else:
                    if candidate_job.preemption_level > core.system_ceiling:
                        core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)
                break


            if core.current_job:
                get_resource(core, resources, global_resource_locks_custom)


                if core.current_job:
                    core.current_job.remaining_time -= 1

                    if time > core.current_job.deadline:
                        miss_occurred = True
                        break

        if miss_occurred:
            break
        time += 1

    for core in cores:
        print(f"custom, Core {core.id} total busy waiting time: {core.busy_waiting_time}")
        for job in core.job_queue + core.secondary_queue:
            if time > job.deadline:
                miss_occurred = True
                break
    return miss_occurred


def simulate_mrsp(cores, resources):
    miss_occurred = False

    global_resource_locks_mrsp = {}

    resource_wait_queue = {res.id: [] for res in resources}

    compute_hyperperiods(cores)
    time = 0
    max_hyperperiod = max(core.hyperperiod for core in cores)

    while time < max_hyperperiod:
        for core in cores:
            # ------------------------------
            # Release resource if needed (end of an interval) + MrsP hand-off
            # ------------------------------
            if core.current_job:
                for res_id, intervals in core.current_job.resource_intervals.items():
                    idx = core.current_job.current_interval_index.get(res_id, 0)
                    if idx < len(intervals):
                        start, end = intervals[idx]
                        if end == (core.current_job.execution_time - core.current_job.remaining_time):
                            core.locked_resources.discard(res_id)
                            if resources[res_id].type == "Global":
                                global_resource_locks_mrsp.pop(res_id, None)

                                if resource_wait_queue[res_id]:
                                    waiter_core_id, waiter_job = resource_wait_queue[res_id].pop(0)
                                    dest = cores[waiter_core_id]

                                    if waiter_job in dest.job_queue:
                                        dest.job_queue.remove(waiter_job)
                                        heapq.heapify(dest.job_queue)

                                    global_resource_locks_mrsp[res_id] = waiter_core_id
                                    dest.locked_resources.add(res_id)

                                    if dest.current_job is not None and dest.current_job is not waiter_job:
                                        heapq.heappush(dest.job_queue, dest.current_job)
                                    dest.current_job = waiter_job

                                    core.system_ceiling = core.calculate_system_ceiling()
                                    dest.system_ceiling = dest.calculate_system_ceiling()

                            core.system_ceiling = core.calculate_system_ceiling()
                            core.current_job.current_interval_index[res_id] = idx + 1

            # ------------------------------
            # Release new jobs (periodic)
            # ------------------------------
            for task in core.tasks:
                if time % task.period == 0:
                    job = Job(task, release_time=time)
                    heapq.heappush(core.job_queue, job)

            # ------------------------------
            # Remove completed jobs (and MrsP hand-off any held globals)
            # ------------------------------
            if core.current_job and core.current_job.remaining_time <= 0:
                core.current_job = None

            # ------------------------------
            # Pick next job / preempt (MrsP helping on preemption)
            # ------------------------------
            if core.job_queue:
                candidate_job = heapq.heappop(core.job_queue)

                if core.current_job:
                    if (candidate_job.preemption_level > core.system_ceiling and
                        candidate_job.preemption_level > core.current_job.preemption_level):

                        holder = core.current_job
                        migrated = False

                        # MrsP: if holder owns a global with waiters, migrate holder to waiter core
                        for res_id in list(core.locked_resources):
                            if resources[res_id].type == "Global" and resource_wait_queue[res_id]:
                                waiter_core_id, waiter_job = resource_wait_queue[res_id].pop(0)
                                dest = cores[waiter_core_id]

                                # Remove waiter from its heap if present
                                if waiter_job in dest.job_queue:
                                    dest.job_queue.remove(waiter_job)
                                    heapq.heapify(dest.job_queue)

                                # Preempt dest's running job (if any)
                                if dest.current_job is not None:
                                    heapq.heappush(dest.job_queue, dest.current_job)

                                holder.preemption_level = waiter_job.preemption_level + 1
                                dest.current_job = holder

                                # Transfer lock ownership and update ceilings
                                global_resource_locks_mrsp[res_id] = waiter_core_id
                                dest.locked_resources.add(res_id)
                                core.locked_resources.discard(res_id)
                                core.system_ceiling = core.calculate_system_ceiling()
                                dest.system_ceiling = dest.calculate_system_ceiling()

                                migrated = True
                                break

                        if migrated:
                            core.current_job = candidate_job
                        else:
                            # Normal preemption
                            heapq.heappush(core.job_queue, core.current_job)
                            core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)
                else:
                    if candidate_job.preemption_level > core.system_ceiling:
                        core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)

            # ------------------------------
            # Execute current job (or spin) with MrsP self-ownership check
            # ------------------------------
            if core.current_job:
                has_busy_waiting = False
                for res_id, intervals in core.current_job.resource_intervals.items():
                    idx = core.current_job.current_interval_index.get(res_id, 0)
                    if idx < len(intervals):
                        start, end = intervals[idx]
                        if start == (core.current_job.execution_time - core.current_job.remaining_time):
                            can_lock = True
                            if resources[res_id].type == "Global":
                                holder = global_resource_locks_mrsp.get(res_id)
                                if holder is not None and holder != core.id:
                                    can_lock = False

                            if can_lock:
                                if res_id not in core.locked_resources:
                                    core.locked_resources.add(res_id)
                                if resources[res_id].type == "Global":
                                    global_resource_locks_mrsp[res_id] = core.id
                                core.system_ceiling = core.calculate_system_ceiling()

                                # If we were queued for this resource, remove ourselves
                                if (core.id, core.current_job) in resource_wait_queue[res_id]:
                                    resource_wait_queue[res_id].remove((core.id, core.current_job))
                            else:
                                # Add once to FIFO wait-queue and spin
                                if (core.id, core.current_job) not in resource_wait_queue[res_id]:
                                    resource_wait_queue[res_id].append((core.id, core.current_job))
                                core.busy_waiting_time += 1
                                has_busy_waiting = True
                                break

                if not has_busy_waiting:
                    core.current_job.remaining_time -= 1

                if time > core.current_job.deadline:
                    miss_occurred = True
                    break

        if miss_occurred:
            break
        time += 1

    for core in cores:
        print(f"MRSP, Core {core.id} total busy waiting time: {core.busy_waiting_time}")
        for job in core.job_queue:
            if job.deadline < time:
                miss_occurred = True
                break
    return miss_occurred


def reset(cores):
    for core in cores:
        core.job_queue = []
        core.current_job = None
        core.system_ceiling = 0
        core.locked_resources = set()
        core.hyperperiod = None
        core.busy_waiting_time = 0




num_tasks_per_core = 20
periods = [100, 200, 500, 1000, 2000, 5000]
RSP = 0.6
MAX_ACCESS = 5


def run_simulation(num_cores, total_utilization):
    """
    Generates a set of cores and resources, assigns tasks/resources, and runs a simulation.
    The simulation function returns True if a deadline miss occurred.
    Therefore, 'scheduleable' is taken as (not result).
    """
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

    msrp = not simulate_msrp(cores, resources)
    reset(cores)
    custom1 = not simulate_custom(cores, resources)
    reset(cores)
    custom2 = not simulate_custom2(cores, resources)
    reset(cores)
    for core in cores:
        core.resource_ceiling_table = {}
        core.build_resource_ceiling_table_mrsp(resources)

    mrsp = not simulate_mrsp(cores, resources)

    return msrp, custom1, custom2, mrsp


# cores_list = [2, 4, 8, 16]
# utilizations = [0.1, 0.25, 0.75]
num_runs = 30

cores_list = [8]
utilizations = [0.1, 0.25, 0.5, 0.75]

# Dictionaries to store the fraction of schedulable runs for each combination
results_msrp = {}  # key: (num_cores, total_utilization)
results_custom1 = {}
results_custom2 = {}
results_mrsp = {}

for ncores in cores_list:
    for util in utilizations:
        msrp_success = 0
        custom1_success = 0
        custom2_success = 0
        mrsp_success = 0
        for i in range(num_runs):
            msrp, custom1, custom2, mrsp = run_simulation(ncores, util)
            if msrp:
                msrp_success += 1
            if custom1:
                custom1_success += 1
            if custom2:
                custom2_success += 1
            if mrsp:
                mrsp_success += 1

        results_msrp[(ncores, util)] = msrp_success / num_runs
        results_custom1[(ncores, util)] = custom1_success / num_runs
        results_custom2[(ncores, util)] = custom2_success / num_runs
        results_mrsp[(ncores, util)] = mrsp_success / num_runs
        print(f"Num Cores: {ncores}, Utilization: {util:.2f} --> "
              f"MSRP: {results_msrp[(ncores, util)]:.2f}, "
              f"Custom1: {results_custom1[(ncores, util)]:.2f}, "
              f"Custom2: {results_custom2[(ncores, util)]:.2f}, "
              f"MRSP: {results_mrsp[(ncores, util)]:.2f}")


plt.figure(figsize=(12, 6))
for ncores in cores_list:
    xs = []
    ys = []
    for util in utilizations:
        xs.append(util)
        ys.append(results_msrp[(ncores, util)])
    plt.plot(xs, ys, marker='o', label=f"Num Cores = {ncores}")
plt.xlabel("Total Utilization per Core")
plt.ylabel("Fraction Schedulable (MSRP)")
plt.title("MSRP: Scheduleability vs. Total Utilization per Core")
plt.legend()
plt.grid(True)
plt.show()

# Plot for Custom1 method
plt.figure(figsize=(12, 6))
for ncores in cores_list:
    xs = []
    ys = []
    for util in utilizations:
        xs.append(util)
        ys.append(results_custom1[(ncores, util)])
    plt.plot(xs, ys, marker='o', label=f"Num Cores = {ncores}")
plt.xlabel("Total Utilization per Core")
plt.ylabel("Fraction Schedulable (Custom1)")
plt.title("Custom1 Method: Scheduleability vs. Total Utilization per Core")
plt.legend()
plt.grid(True)
plt.show()

# Plot for Custom2 method
plt.figure(figsize=(12, 6))
for ncores in cores_list:
    xs = []
    ys = []
    for util in utilizations:
        xs.append(util)
        ys.append(results_custom2[(ncores, util)])
    plt.plot(xs, ys, marker='o', label=f"Num Cores = {ncores}")
plt.xlabel("Total Utilization per Core")
plt.ylabel("Fraction Schedulable (Custom2)")
plt.title("Custom2 Method: Scheduleability vs. Total Utilization per Core")
plt.legend()
plt.grid(True)
plt.show()


# Plot MRSP
plt.figure(figsize=(12, 6))
for ncores in cores_list:
    xs = []
    ys = []
    for util in utilizations:
        xs.append(util)
        ys.append(results_mrsp[(ncores, util)])
    plt.plot(xs, ys, marker='o', label=f"MRSP: Cores = {ncores}")
plt.xlabel("Total Utilization per Core")
plt.ylabel("Fraction Schedulable")
plt.title("MRSP: Schedulability vs. Utilization")
plt.legend()
plt.grid(True)
plt.show()
