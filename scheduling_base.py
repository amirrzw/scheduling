import random
import math
from functools import reduce
import heapq

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

MISS_INFO = None
# اختیاری: اگر set شود، هر بار که busy_waiting_time += 1 می‌شود با (core, time, reason) صدا زده می‌شود
BUSY_LOG_CALLBACK = None
# اگر True باشد، توابع شبیه‌سازی در پایان برای هر core چاپ نمی‌کنند (برای batch اجرا)
SIMULATE_QUIET = False
# اختیاری: در simulate_dsp_s هر تیک با (time, cores) صدا زده می‌شود (برای دیباگ)
DSP_TRACE_CALLBACK = None


# ---------------------------------------------------------------------------
# مدل‌ها
# ---------------------------------------------------------------------------

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
        self.accessed_by = set()
        self.type = None

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
        self.is_busy_waiting = False
        self.blocked_on_resource = None

    def __lt__(self, other):
        return self.deadline < other.deadline


class Core:
    def __init__(self, id):
        self.id = id
        self.tasks = []
        self.job_queue = []
        self.current_job = None
        self.secondary_queue = []
        self.system_ceiling = 0
        self.resource_ceiling_table = {}
        self.locked_resources = set()
        self.hyperperiod = None
        self.busy_waiting_time = 0

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


# ---------------------------------------------------------------------------
# توابع کمکی
# ---------------------------------------------------------------------------

def reset_cores_and_resources(cores, resources):
    for core in cores:
        core.job_queue = []
        core.secondary_queue = []
        core.current_job = None
        core.busy_waiting_time = 0
        core.locked_resources = set()
        core.system_ceiling = 0


def unifast(num_tasks, total_utilization):
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
    if execution_time < num_chunks:
        num_chunks = execution_time
    if num_chunks <= 1:
        return [(0, execution_time)]
    cut_points = sorted(random.sample(range(1, execution_time), num_chunks - 1))
    pieces = []
    prev = 0
    for cp in cut_points:
        pieces.append(cp - prev)
        prev = cp
    pieces.append(execution_time - prev)
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
            random.shuffle(intervals)
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


def _job_allowed_under_ceiling_restriction(core, job, threshold_pl):
    if job.preemption_level >= threshold_pl:
        return False
    for res_id in job.resource_intervals:
        if core.resource_ceiling_table.get(res_id, 0) >= threshold_pl:
            return False
    return True


def would_job_lock_resource_this_tick(job, res_id):
    if res_id not in job.resource_intervals:
        return False
    exec_point = job.execution_time - job.remaining_time
    for start, end in job.resource_intervals[res_id]:
        if start == exec_point:
            return True
    return False


def would_job_lock_any_resource_this_tick(job):
    """آیا اگر این جاب همین تیک اجرا شود ریسورسی را قفل می‌کند؟ (بخش critical این تیک)"""
    return any(would_job_lock_resource_this_tick(job, res_id) for res_id in job.resource_intervals)


def would_job_lock_global_this_tick(job, resources):
    """آیا اگر این جاب همین تیک اجرا شود یک ریسورس گلوبال را قفل می‌کند؟ (برای DSP-S: فقط از این جلوگیری می‌کنیم)"""
    for res_id in job.resource_intervals:
        if resources[res_id].type == "Global" and would_job_lock_resource_this_tick(job, res_id):
            return True
    return False


def has_upcoming_critical_section(job, time_units=1):
    current_exec_point = job.execution_time - job.remaining_time
    future_exec_point = current_exec_point + time_units
    for res_id, intervals in job.resource_intervals.items():
        for start, end in intervals:
            if start < future_exec_point and end > current_exec_point:
                if start <= current_exec_point:
                    return (True, 0)
                return (True, start - current_exec_point)
    return (False, 0)


# ---------------------------------------------------------------------------
# شبیه‌سازی MSRP
# ---------------------------------------------------------------------------

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
                    idx = core.current_job.current_interval_index.get(res_id, 0)
                    if idx < len(intervals):
                        start, end = intervals[idx]
                        if end == (core.current_job.execution_time - core.current_job.remaining_time):
                            core.locked_resources.discard(res_id)
                            if resources[res_id].type == "Global":
                                global_resource_locks.pop(res_id, None)
                            core.system_ceiling = core.calculate_system_ceiling()
                            core.current_job.current_interval_index[res_id] = idx + 1

        for core in cores:
            for task in core.tasks:
                if time % task.period == 0:
                    job = Job(task, release_time=time)
                    heapq.heappush(core.job_queue, job)

        for core in cores:
            if core.current_job and core.current_job.remaining_time <= 0:
                core.current_job = None

        for core in cores:
            if core.job_queue:
                candidate_job = heapq.heappop(core.job_queue)
                if core.current_job:
                    if not core.current_job.is_busy_waiting:
                        if candidate_job.preemption_level > core.system_ceiling and candidate_job.preemption_level > core.current_job.preemption_level and candidate_job.deadline <= core.current_job.deadline:
                            heapq.heappush(core.job_queue, core.current_job)
                            core.current_job = candidate_job
                        else:
                            heapq.heappush(core.job_queue, candidate_job)
                    else:
                        heapq.heappush(core.job_queue, candidate_job)
                else:
                    if candidate_job.preemption_level > core.system_ceiling:
                        core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)

        for core in cores:
            if core.current_job:
                has_busy_waiting = False
                for res_id, intervals in core.current_job.resource_intervals.items():
                    idx = core.current_job.current_interval_index.get(res_id, 0)
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
                                core.current_job.is_busy_waiting = False
                            else:
                                core.busy_waiting_time += 1
                                core.current_job.is_busy_waiting = True
                                has_busy_waiting = True
                                break
                if not has_busy_waiting:
                    core.current_job.remaining_time -= 1
                if time > core.current_job.deadline:
                    miss_occurred = True
                    if MISS_INFO is not None:
                        MISS_INFO.append(("msrp", time, core.id, core.current_job))
                    break
        if miss_occurred:
            break
        time += 1

    for core in cores:
        if not SIMULATE_QUIET:
            print(f"msrp, Core {core.id} total busy waiting time: {core.busy_waiting_time}")
        for job in core.job_queue:
            if job.deadline < time:
                miss_occurred = True
                break
    return miss_occurred


# ---------------------------------------------------------------------------
# MSRP با صف ثانویه: جاب منتظر ریسورس گلوبال به secondary_queue می‌رود؛
# در زمان انتظار فقط بخش‌های non-critical (بدون لاک ریسورس) اجرا می‌شوند؛
# پس از آزاد شدن ریسورس، جاب فوراً برمی‌گردد و لاک می‌کند.
# ---------------------------------------------------------------------------

def _pick_noncritical_job(core):
    """از job_queue جابی را برمی‌گرداند که این تیک ریسورس لاک نمی‌کند. جاب‌های دیگر را برمی‌گرداند به صف."""
    put_back = []
    while core.job_queue:
        j = heapq.heappop(core.job_queue)
        if not would_job_lock_any_resource_this_tick(j):
            for b in put_back:
                heapq.heappush(core.job_queue, b)
            return j
        put_back.append(j)
    for b in put_back:
        heapq.heappush(core.job_queue, b)
    return None


def _job_has_any_global_resource(job, core, resources):
    """True if the task of this job has any global resource in its resource_intervals."""
    task = next((t for t in core.tasks if t.id == job.task_id), None)
    if task is None:
        return True
    return any(resources[res_id].type == "Global" for res_id in task.resource_intervals)


def _pick_dsp_s_job(core, resources):
    """DSP-S: only jobs whose task has no global resources, and that do not lock a global resource this tick.
    Allowing jobs that lock only local this tick preserves dominance over MSRP (core is not left idle)."""
    put_back = []
    while core.job_queue:
        j = heapq.heappop(core.job_queue)
        if not _job_has_any_global_resource(j, core, resources) and not would_job_lock_global_this_tick(j, resources):
            for b in put_back:
                heapq.heappush(core.job_queue, b)
            return j
        put_back.append(j)
    for b in put_back:
        heapq.heappush(core.job_queue, b)
    return None


def _resource_at_exec_point(job, resources):
    """در نقطه اجرای فعلی جاب به کدام ریسورس نیاز دارد؟ (شروع بازه) برمی‌گرداند (res_id, is_global) یا (None, False)."""
    exec_point = job.execution_time - job.remaining_time
    for res_id, intervals in job.resource_intervals.items():
        idx = job.current_interval_index.get(res_id, 0)
        if idx >= len(intervals):
            continue
        start, end = intervals[idx]
        if start != exec_point:
            continue
        return (res_id, resources[res_id].type == "Global")
    return (None, False)


def simulate_msrp_with_secondary_queue(cores, resources):
    """
    مثل MSRP با این تفاوت:
    - جابی که به ریسورس گلوبال نیاز دارد و قفل است → به secondary_queue می‌رود.
    - وقتی کسی در secondary_queue است فقط بخش non-critical اجرا می‌شود (این تیک ریسورس لاک نمی‌شود).
    - با آزاد شدن ریسورس گلوبال، جاب منتظر فوراً برمی‌گردد (pending_returned_job) و لاک می‌کند و اجرا می‌شود.
    """
    global_locks = {}
    miss_occurred = False
    compute_hyperperiods(cores)
    time = 0
    max_hyperperiod = max(core.hyperperiod for core in cores)
    for core in cores:
        core.secondary_queue = []
        setattr(core, "pending_returned_job", None)

    while time < max_hyperperiod:
        # ─── ۱) رها کردن ریسورس در پایان بازه + برگرداندن جاب منتظر از secondary_queue ───
        for core in cores:
            if not core.current_job:
                continue
            for res_id, intervals in core.current_job.resource_intervals.items():
                idx = core.current_job.current_interval_index.get(res_id, 0)
                if idx >= len(intervals):
                    continue
                start, end = intervals[idx]
                if end != (core.current_job.execution_time - core.current_job.remaining_time):
                    continue
                core.locked_resources.discard(res_id)
                if resources[res_id].type == "Global":
                    global_locks.pop(res_id, None)
                    for other in cores:
                        for sec in list(getattr(other, "secondary_queue", [])):
                            if getattr(sec, "blocked_on_resource", None) == res_id:
                                other.secondary_queue.remove(sec)
                                sec.blocked_on_resource = None
                                other.pending_returned_job = sec
                                break
                        if getattr(other, "pending_returned_job", None) is not None:
                            break
                core.system_ceiling = core.calculate_system_ceiling()
                core.current_job.current_interval_index[res_id] = idx + 1

        # ─── ۲) رهاسازی جاب‌های جدید ───
        for core in cores:
            for task in core.tasks:
                if time % task.period == 0:
                    heapq.heappush(core.job_queue, Job(task, release_time=time))

        # ─── ۳) حذف جاب‌های تمام‌شده ───
        for core in cores:
            if core.current_job and core.current_job.remaining_time <= 0:
                core.current_job = None

        # ─── ۴) انتخاب جاب برای اجرا ───
        for core in cores:
            core.current_job_just_returned = False
            pending = getattr(core, "pending_returned_job", None)
            if pending is not None:
                core.pending_returned_job = None
                core.current_job_just_returned = True
                if core.current_job:
                    heapq.heappush(core.job_queue, core.current_job)
                core.current_job = pending
            elif core.job_queue:
                if core.secondary_queue:
                    job = _pick_noncritical_job(core)
                    if job is not None:
                        if core.current_job:
                            heapq.heappush(core.job_queue, core.current_job)
                        core.current_job = job
                    else:
                        if core.job_queue:
                            cand = heapq.heappop(core.job_queue)
                            if core.current_job:
                                heapq.heappush(core.job_queue, core.current_job)
                            core.current_job = cand
                else:
                    cand = heapq.heappop(core.job_queue)
                    if core.current_job:
                        if cand.preemption_level > core.system_ceiling and cand.preemption_level > core.current_job.preemption_level and cand.deadline <= core.current_job.deadline:
                            heapq.heappush(core.job_queue, core.current_job)
                            core.current_job = cand
                        else:
                            heapq.heappush(core.job_queue, cand)
                    else:
                        core.current_job = cand

        # ─── ۵) اجرا: اگر صف ثانویه پر است و جاب فعلی این تیک لاک می‌کند (و برگشتی نیست) → جاب non-critical ───
        for core in cores:
            if core.current_job and core.secondary_queue:
                if would_job_lock_any_resource_this_tick(core.current_job) and not getattr(core, "current_job_just_returned", False):
                    heapq.heappush(core.job_queue, core.current_job)
                    core.current_job = _pick_noncritical_job(core)
                    if core.current_job is None and core.job_queue:
                        core.current_job = heapq.heappop(core.job_queue)
                    if core.current_job is None:
                        core.busy_waiting_time += 1
                        if BUSY_LOG_CALLBACK is not None:
                            BUSY_LOG_CALLBACK(core, time, "replace_no_noncritical")

            while core.current_job:
                job = core.current_job
                res_id, is_global = _resource_at_exec_point(job, resources)
                if res_id is not None and is_global and res_id in global_locks:
                    job.blocked_on_resource = res_id
                    # Release resources this job was holding so system_ceiling drops (same fix as DSP-S)
                    for rid in list(core.locked_resources):
                        core.locked_resources.discard(rid)
                        if resources[rid].type == "Global":
                            global_locks.pop(rid, None)
                    core.system_ceiling = core.calculate_system_ceiling()
                    core.secondary_queue.append(job)
                    core.current_job = None
                    core.current_job = _pick_noncritical_job(core)
                    if core.current_job is None and core.job_queue:
                        core.current_job = heapq.heappop(core.job_queue)
                    if core.current_job is None:
                        core.busy_waiting_time += 1
                        if BUSY_LOG_CALLBACK is not None:
                            BUSY_LOG_CALLBACK(core, time, "secondary_no_noncritical")
                    break
                if res_id is not None:
                    core.locked_resources.add(res_id)
                    if is_global:
                        global_locks[res_id] = core.id
                    core.system_ceiling = core.calculate_system_ceiling()
                break

            if core.current_job:
                core.current_job.remaining_time -= 1
                if time > core.current_job.deadline:
                    miss_occurred = True
                    if MISS_INFO is not None:
                        MISS_INFO.append(("msrp_secondary", time, core.id, core.current_job))
                    break
        if miss_occurred:
            break
        time += 1

    for core in cores:
        if not SIMULATE_QUIET:
            print(f"msrp_secondary, Core {core.id} busy_wait={core.busy_waiting_time}")
        for job in core.job_queue + core.secondary_queue:
            if time > job.deadline:
                miss_occurred = True
                if MISS_INFO is not None:
                    MISS_INFO.append(("msrp_secondary", time, core.id, job))
                break
    return miss_occurred


# ---------------------------------------------------------------------------
# DSP-S: صف ثانویه؛ در زمان spin فقط جاب‌های بدون هیچ ریسورس گلوبال
# ---------------------------------------------------------------------------

def simulate_dsp_s(cores, resources):
    """
    Like simulate_msrp_with_secondary_queue (DSP-F) but during spin only jobs
    with no global resources are run (DSP-S: safe, no cascade).
    """
    global_locks = {}
    miss_occurred = False
    compute_hyperperiods(cores)
    time = 0
    max_hyperperiod = max(core.hyperperiod for core in cores)
    for core in cores:
        core.secondary_queue = []
        setattr(core, "pending_returned_job", None)

    while time < max_hyperperiod:
        if DSP_TRACE_CALLBACK is not None:
            DSP_TRACE_CALLBACK(time, cores)
        for core in cores:
            if not core.current_job:
                continue
            for res_id, intervals in core.current_job.resource_intervals.items():
                idx = core.current_job.current_interval_index.get(res_id, 0)
                if idx >= len(intervals):
                    continue
                start, end = intervals[idx]
                if end != (core.current_job.execution_time - core.current_job.remaining_time):
                    continue
                core.locked_resources.discard(res_id)
                if resources[res_id].type == "Global":
                    global_locks.pop(res_id, None)
                    for other in cores:
                        for sec in list(getattr(other, "secondary_queue", [])):
                            if getattr(sec, "blocked_on_resource", None) == res_id:
                                other.secondary_queue.remove(sec)
                                sec.blocked_on_resource = None
                                other.pending_returned_job = sec
                                break
                        if getattr(other, "pending_returned_job", None) is not None:
                            break
                core.system_ceiling = core.calculate_system_ceiling()
                core.current_job.current_interval_index[res_id] = idx + 1

        for core in cores:
            for task in core.tasks:
                if time % task.period == 0:
                    heapq.heappush(core.job_queue, Job(task, release_time=time))

        for core in cores:
            if core.current_job and core.current_job.remaining_time <= 0:
                core.current_job = None

        for core in cores:
            core.current_job_just_returned = False
            pending = getattr(core, "pending_returned_job", None)
            if pending is not None:
                core.pending_returned_job = None
                core.current_job_just_returned = True
                if core.current_job:
                    heapq.heappush(core.job_queue, core.current_job)
                core.current_job = pending
            elif core.job_queue:
                if core.secondary_queue:
                    job = _pick_dsp_s_job(core, resources)
                    if job is not None:
                        if core.current_job:
                            heapq.heappush(core.job_queue, core.current_job)
                        core.current_job = job
                    else:
                        # Fallback: no eligible no-global job; run earliest-deadline to avoid idle core (preserves dominance)
                        cand = heapq.heappop(core.job_queue)
                        if core.current_job:
                            heapq.heappush(core.job_queue, core.current_job)
                        core.current_job = cand
                else:
                    cand = heapq.heappop(core.job_queue)
                    if core.current_job:
                        if cand.preemption_level > core.system_ceiling and cand.preemption_level > core.current_job.preemption_level and cand.deadline <= core.current_job.deadline:
                            heapq.heappush(core.job_queue, core.current_job)
                            core.current_job = cand
                        else:
                            heapq.heappush(core.job_queue, cand)
                    else:
                        # No current job: core is free (released or no holder), so run earliest-deadline to avoid idle core
                        core.current_job = cand

        for core in cores:
            if core.current_job and core.secondary_queue:
                if would_job_lock_any_resource_this_tick(core.current_job) and not getattr(core, "current_job_just_returned", False):
                    heapq.heappush(core.job_queue, core.current_job)
                    core.current_job = _pick_dsp_s_job(core, resources)
                    if core.current_job is None:
                        core.busy_waiting_time += 1
                        if BUSY_LOG_CALLBACK is not None:
                            BUSY_LOG_CALLBACK(core, time, "dsp_s_replace_no_job")

            while core.current_job:
                job = core.current_job
                res_id, is_global = _resource_at_exec_point(job, resources)
                if res_id is not None and is_global and res_id in global_locks:
                    job.blocked_on_resource = res_id
                    # Release all resources this job was holding (e.g. local) so system_ceiling drops and next schedule can pick a job
                    for rid in list(core.locked_resources):
                        core.locked_resources.discard(rid)
                        if resources[rid].type == "Global":
                            global_locks.pop(rid, None)
                    core.system_ceiling = core.calculate_system_ceiling()
                    core.secondary_queue.append(job)
                    core.current_job = None
                    core.current_job = _pick_dsp_s_job(core, resources)
                    if core.current_job is None and core.job_queue:
                        core.current_job = heapq.heappop(core.job_queue)
                    if core.current_job is None:
                        core.busy_waiting_time += 1
                        if BUSY_LOG_CALLBACK is not None:
                            BUSY_LOG_CALLBACK(core, time, "dsp_s_secondary_no_job")
                    break
                if res_id is not None:
                    core.locked_resources.add(res_id)
                    if is_global:
                        global_locks[res_id] = core.id
                    core.system_ceiling = core.calculate_system_ceiling()
                break

            if core.current_job:
                core.current_job.remaining_time -= 1
                if time > core.current_job.deadline:
                    miss_occurred = True
                    if MISS_INFO is not None:
                        MISS_INFO.append(("dsp_s", time, core.id, core.current_job))
                    break
        if miss_occurred:
            break
        time += 1

    for core in cores:
        if not SIMULATE_QUIET:
            print(f"DSP-S, Core {core.id} busy_wait={core.busy_waiting_time}")
        for job in core.job_queue + core.secondary_queue:
            if time > job.deadline:
                miss_occurred = True
                if MISS_INFO is not None:
                    MISS_INFO.append(("dsp_s", time, core.id, job))
                break
    return miss_occurred


# ---------------------------------------------------------------------------
# شبیه‌سازی MRSP
# ---------------------------------------------------------------------------

def simulate_mrsp(cores, resources):
    miss_occurred = False
    global_resource_locks_mrsp = {}
    resource_wait_queue = {res.id: [] for res in resources}
    compute_hyperperiods(cores)
    time = 0
    max_hyperperiod = max(core.hyperperiod for core in cores)
    while time < max_hyperperiod:
        for core in cores:
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

        for core in cores:
            for task in core.tasks:
                if time % task.period == 0:
                    job = Job(task, release_time=time)
                    heapq.heappush(core.job_queue, job)
        for core in cores:
            if core.current_job and core.current_job.remaining_time <= 0:
                core.current_job = None
        for core in cores:
            if core.job_queue:
                candidate_job = heapq.heappop(core.job_queue)
                if core.current_job:
                    if (candidate_job.preemption_level > core.system_ceiling and
                        candidate_job.preemption_level > core.current_job.preemption_level and candidate_job.deadline <= core.current_job.deadline):
                        holder = core.current_job
                        migrated = False
                        for res_id in list(core.locked_resources):
                            if resources[res_id].type == "Global" and resource_wait_queue[res_id]:
                                waiter_core_id, waiter_job = resource_wait_queue[res_id].pop(0)
                                dest = cores[waiter_core_id]
                                if waiter_job in dest.job_queue:
                                    dest.job_queue.remove(waiter_job)
                                    heapq.heapify(dest.job_queue)
                                if dest.current_job is not None:
                                    heapq.heappush(dest.job_queue, dest.current_job)
                                holder.preemption_level = waiter_job.preemption_level + 1
                                dest.current_job = holder
                                global_resource_locks_mrsp[res_id] = waiter_core_id
                                dest.locked_resources.add(res_id)
                                core.locked_resources.discard(res_id)
                                core.system_ceiling = core.calculate_system_ceiling()
                                dest.system_ceiling = dest.calculate_system_ceiling()
                                migrated = True
                                break
                            elif resources[res_id].type == "Global":
                                global_resource_locks_mrsp.pop(res_id, None)
                        core.locked_resources = set()
                        if migrated:
                            core.current_job = candidate_job
                        else:
                            heapq.heappush(core.job_queue, core.current_job)
                            core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)
                else:
                    if candidate_job.preemption_level > core.system_ceiling:
                        core.current_job = candidate_job
                    else:
                        heapq.heappush(core.job_queue, candidate_job)
        for core in cores:
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
                                if (core.id, core.current_job) in resource_wait_queue[res_id]:
                                    resource_wait_queue[res_id].remove((core.id, core.current_job))
                            else:
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
        if not SIMULATE_QUIET:
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
        core.secondary_queue = []
        setattr(core, "pending_returned_job", None)
