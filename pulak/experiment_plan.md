# Experiment Plan: Scheduling-Driven Multitenancy on MIG-Enabled GPUs

## 1. Overview

This project investigates whether *smarter scheduling* can effectively solve the multitenancy problem on NVIDIA GPUs with Multi-Instance GPU (MIG) support.  
Scheduling improves allocation efficiency and handles fairness under bursty, heterogeneous workloads.
Instead of focusing solely on hardware partitioning, we treat MIG partitions and GPU time as scheduling resources that can be allocated, reallocated, or overcommitted to improve fairness, utilization, and quality of service (QoS).

**Core Question:**  
*Can a scheduling-centric approach outperform static MIG partitioning in real multi-tenant GPU workloads?*

---

## 2. Goals

1. **Quantify multitenancy bottlenecks** on MIG GPUs under naive or static allocation.
2. **Design and implement a GPU scheduling framework** that:
   - Dynamically assigns MIG slices,
   - Schedules processes within each slice,
   - Orchestrates elastic or composable MIG partition changes.
3. **Measure fairness, utilization, and slowdown** under real workloads.
4. **Compare against baseline systems**:
   - Static MIG configs,
   - First-come-first-served (FCFS),
   - NVIDIA MPS,
   - Fully shared (non-MIG) GPU environments.
5. **Show that scheduling can substantially improve multitenant fairness and throughput.**

---

## 3. Experimental Questions

1. **Does scheduling + MIG outperform static MIG?**  
   Measure improvements in fairness, utilization, and tail latency.

2. **Does a dynamic scheduler reduce interference between tenants?**  
   Compare execution-time slowdown under mixed workloads.

3. **Can time-sharing across MIG slices improve utilization?**  
   Are some workloads naturally bursty or queue-limited?

4. **How often should MIG partitions be recomposed?**  
   What is the overhead and performance tradeoff?

5. **Do different scheduling algorithms behave differently?**
   - Dominant Resource Fairness (DRF)
   - Weighted fair queuing
   - Shortest-remaining-processing-time (SRPT)
   - Reinforcement-learning-based scheduling

---

## 4. Experiments

This plan compares GPU sharing across two hardware targets:

- NVIDIA A100: Evaluate scheduling + MIG spatial partitioning and hybrid MIG+MPS modes.
- NVIDIA T4: Perform large scheduling sweeps using MPS-only time sharing.

**Workloads:** Heterogeneous batch of 6 jobs:

- Inference-Heavy (4 vLLM, 2 Training)
- Training-Heavy (4 Training, 2 vLLM)
- Balanced (3 Training, 3 vLLM)

### 1. NVIDIA A100 Experiments — Scheduling + MIG
**Objective:** Show that scheduling decisions (placement, ordering, hybrid sharing) improve fairness, utilization, and P99 latency even when hardware isolation (MIG) is available.

- **Phase A0 — Isolated baseline**
  - Sharing: MIG — each job gets an appropriate dedicated slice.
  - Goal: Establish isolated JRT and P99 latency (interference factor = 1.0).

- **Phase A1 — Naïve MIG placement (random)**
  - Sharing: Fixed MIG layout (e.g., two 2g.10gb + one 3g.20gb).
  - Goal: Show fragmentation/packing inefficiency from non-scheduled placement.

- **Phase A2 — Profile-aware MIG placement**
  - Sharing: Same MIG layout.
  - Scheduler: Assign by job profile (runtime, memory, SM demand).
  - Focus: Improve utilization and makespan vs naïve placement.

- **Phase A3 — Latency-prioritized placement**
  - Strategy: Put inference jobs on small isolated slices; throughput jobs on larger slices.
  - Focus: Minimize P99 latency while keeping utilization reasonable.

### 2. NVIDIA T4 Experiments — Scheduling Policy Sweeps (MPS)
**Objective:** Quantify the impact of scheduling policy and resource capping on makespan, fairness, queueing behavior, and throughput under MPS-only sharing.

- **Phase T0 — MPS isolated baseline**
  - Sharing: Jobs run one at a time.
  - Focus: Establish isolated JCT (interference factor reference).

- **Phase T1 — Default (random arrival)**
  - Sharing: MPS (`mps_level=100`).
  - Ordering: Typical/random job order.
  - Focus: Baseline for scheduling sensitivity.

- **Phase T2 — Scheduling policy sweep**
  - Policies under MPS: FIFO; SJF (profiled runtime); SRTF (preemptive approx); size-/memory-aware packing; fairness-aware (e.g., weighted fair queuing).
  - Focus: Compare fairness, makespan, JCT, queueing behavior.

- **Phase T3 — Adversarial ordering stress test**
  - Ordering: Largest first → smallest last.
  - Sharing: MPS (`mps_level=100`).
  - Focus: See how T2 policies mitigate head-of-line blocking.

- **Phase T4 — Resource cap sweep (policy × `mps_level`)**
  - Ordering: Same as T3.
  - Sweep: `mps_level` ∈ {25, 33, 50, 100}.
  - Focus: Fairness vs throughput trade-off under the best policies.
---

## 5. Conclusion

This project reframes GPU multitenancy as a **scheduling problem**, not just a hardware partitioning problem.  
If successful, it will demonstrate that:

> **Smart scheduling + MIG → better fairness, better QoS, higher utilization, lower interference.**

This can inform the design of future GPU cluster managers and datacenter schedulers.
