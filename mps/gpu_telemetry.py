#!/usr/bin/env python3
"""
dcgm_nvml_collector.py

Combined DCGM (CLI) + NVML telemetry collector.

Features:
- DCGM CLI (dcgmi dmon --csv) -> GPU-wide timeseries (ndjson per GPU)
- NVML (pynvml) -> per-process memory & process list samples (ndjson per GPU)
- Rotating NDJSON writers (size-based), optional gzip compression, file retention
- Bounded in-memory summaries via Welford's algorithm (per-field)
- CLI: python dcgm_nvml_collector.py 0,1 logs/telemetry 1.0
"""

from __future__ import annotations
import time
import json
import os
import sys
import threading
import subprocess
import shutil
import gzip
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict

# Attempt to import pynvml
try:
    import pynvml
except Exception as e:
    pynvml = None

# -----------------------
# Configurable defaults
# -----------------------
DEFAULT_SAMPLE_INTERVAL = 1.0
DEFAULT_OUTPUT_DIR = "logs/telemetry"
DEFAULT_MAX_BYTES_PER_FILE = 50 * 1024 * 1024  # 50 MB
DEFAULT_MAX_ROTATED_FILES = 10
DEFAULT_COMPRESS_ROTATED = True
DEFAULT_SUMMARY_WRITE_INTERVAL = 60.0  # seconds

# -----------------------
# Utilities
# -----------------------
class RotatingNDJSONWriter:
    """Write one JSON object per line; rotate on size threshold; optionally compress rotated files."""
    def __init__(self, path: Path, max_bytes: int, max_files: int, compress: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.max_files = int(max_files)
        self.compress = bool(compress)
        self._open()

    def _open(self):
        # append to existing current file to avoid losing partial runs
        self._f = open(self.path, "a", buffering=1)  # line buffered

    def _size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def _rotate(self):
        try:
            self._f.close()
        except Exception:
            pass
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        rotated = self.path.with_name(f"{self.path.stem}.{timestamp}.ndjson")
        shutil.move(str(self.path), str(rotated))
        if self.compress:
            gz = str(rotated) + ".gz"
            with open(rotated, "rb") as src, gzip.open(gz, "wb") as dst:
                shutil.copyfileobj(src, dst)
            try:
                os.remove(rotated)
            except Exception:
                pass
            rotated = Path(gz)
        # enforce retention
        self._enforce_retention()
        self._open()

    def _enforce_retention(self):
        parent = self.path.parent
        stem = self.path.stem
        # captured rotated names like stem.YYYYMMDDTHHMMSS.ndjson or .ndjson.gz
        candidates = sorted(parent.glob(f"{stem}.*.ndjson*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in candidates[self.max_files:]:
            try:
                os.remove(old)
            except Exception:
                pass

    def write(self, obj: Dict[str, Any]):
        try:
            line = json.dumps(obj, default=str)
            self._f.write(line + "\n")
            if self._size() >= self.max_bytes:
                self._rotate()
        except Exception as e:
            # do not crash collector on write errors
            print(f"[RotatingNDJSONWriter] write error: {e}", file=sys.stderr)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

class OnlineStats:
    """Welford's algorithm for streaming mean/std/min/max/count."""
    __slots__ = ("n", "mean", "M2", "min", "max")
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = None
        self.max = None

    def add(self, x: float):
        if x is None:
            return
        x = float(x)
        if self.n == 0:
            self.n = 1
            self.mean = x
            self.M2 = 0.0
            self.min = x
            self.max = x
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x

    def summary(self) -> Dict[str, Any]:
        if self.n == 0:
            return {}
        variance = (self.M2 / self.n) if self.n > 1 else 0.0
        return {"count": int(self.n), "mean": float(self.mean), "min": float(self.min), "max": float(self.max), "std": float(variance ** 0.5)}

# -----------------------
# DCGM CLI backend
# -----------------------
class DCGMCLIReader:
    """
    Runs `dcgmi dmon -e <ms> --csv` and parses CSV rows.
    Calls back to on_sample(dict).
    """
    def __init__(self, gpu_ids: List[int], sample_interval: float, on_sample: Callable[[Dict[str, Any]], None]):
        self.gpu_ids = list(gpu_ids)
        self.sample_interval = sample_interval
        self.on_sample = on_sample
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        # build command
        ms = max(1, int(self.sample_interval * 1000))
        cmd = ["dcgmi", "dmon", "-e", str(ms), "--csv"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        except FileNotFoundError as e:
            raise RuntimeError("dcgmi not found in PATH. Install DCGM CLI or ensure dcgmi is on PATH.") from e
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        assert self._proc is not None and self._proc.stdout is not None
        import csv
        reader = csv.reader(self._proc.stdout)
        header = None
        # Some dcgmi versions print occasional banner lines; we detect header when a row contains 'timestamp' or 'gpu'
        while self._running:
            try:
                row = next(reader)
            except StopIteration:
                break
            except Exception:
                continue
            if not row:
                continue
            # header detection
            low = [c.strip().lower() for c in row]
            if any("timestamp" in c or "gpu" == c or c.startswith("gpu ") for c in low) and any("util" in c or "mem" in c or "power" in c for c in low):
                header = [c.strip() for c in row]
                continue
            if header is None:
                # skip until header found
                continue
            # map row to dict
            d = {header[i]: (row[i].strip() if i < len(row) else "") for i in range(len(header))}
            # attempt to parse GPU id field
            gid = None
            for candidate in header:
                cl = candidate.lower()
                if cl in ("gpu", "gpu id", "gpu_id", "device"):
                    try:
                        gid = int(d[candidate])
                        break
                    except Exception:
                        gid = None
            # if gpu couldn't be determined and only one GPU monitored, assign
            if gid is None and len(self.gpu_ids) == 1:
                gid = self.gpu_ids[0]
            # drop rows for gpus not configured (if we found gid)
            if gid is not None and gid not in self.gpu_ids:
                continue
            # convert numeric-looking values
            metrics = {"timestamp": time.time()}
            if gid is not None:
                metrics["gpu_id"] = gid
            for k, v in d.items():
                val = v
                if val == "":
                    metrics[k] = None
                    continue
                s = val.replace("%", "").replace(",", "")
                try:
                    if "." in s:
                        metrics[k] = float(s)
                    else:
                        metrics[k] = int(s)
                except Exception:
                    metrics[k] = val
            # push callback
            try:
                self.on_sample(metrics)
            except Exception:
                pass
        # end loop
        self._running = False

    def stop(self):
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._running

# -----------------------
# NVML process sampler
# -----------------------
class NVMLProcessSampler:
    """
    Samples per-GPU process lists and process GPU memory usage using pynvml.
    Calls on_sample(metrics) with a per-gpu dict: {'timestamp', 'gpu_id', 'processes':[{'pid', 'used_memory_mb', ...}], 'num_processes'}
    """
    def __init__(self, gpu_ids: List[int], sample_interval: float, on_sample: Callable[[Dict[str, Any]], None]):
        if pynvml is None:
            raise RuntimeError("pynvml is not installed. Install with: pip install nvidia-ml-py3")
        self.gpu_ids = list(gpu_ids)
        self.sample_interval = sample_interval
        self.on_sample = on_sample
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # init nvml once
        try:
            pynvml.nvmlInit()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}")

        # map gpu_ids to handles
        self.handles = {}
        for gid in self.gpu_ids:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(gid)
                self.handles[gid] = h
            except Exception as e:
                print(f"[NVML] warning: failed to get handle for GPU {gid}: {e}", file=sys.stderr)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ts = time.time()
            for gid, h in list(self.handles.items()):
                try:
                    procs = []
                    # nvmlDeviceGetComputeRunningProcesses (may differ per NVML version)
                    try:
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                    except Exception:
                        # older/newer NVMLs might require nvmlDeviceGetComputeRunningProcesses_v2
                        try:
                            processes = pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
                        except Exception:
                            processes = []
                    for p in processes:
                        # p.pid and p.usedGpuMemory usually present
                        pid = getattr(p, "pid", None)
                        mem = getattr(p, "usedGpuMemory", None)
                        # convert bytes -> MB
                        mem_mb = (mem / (1024 * 1024)) if mem is not None else None
                        procs.append({"pid": int(pid) if pid is not None else None, "used_memory_mb": float(mem_mb) if mem_mb is not None else None})
                    metrics = {"timestamp": ts, "gpu_id": gid, "processes": procs, "num_processes": len(procs)}
                    # callback
                    try:
                        self.on_sample(metrics)
                    except Exception:
                        pass
                except Exception as e:
                    # continue sampling other GPUs
                    print(f"[NVML] sampling error for gpu {gid}: {e}", file=sys.stderr)
            # sleep until next iteration
            time.sleep(self.sample_interval)
        # exiting: don't call nvmlShutdown() here (upper layer will handle if desired)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

# -----------------------
# Combined collector
# -----------------------
class CombinedCollector:
    """
    Orchestrates DCGM CLI reader + NVML process sampler; writes NDJSON per GPU and per-process files,
    and maintains OnlineStats summaries.
    """
    def __init__(self,
                 gpu_ids: List[int],
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
                 max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
                 max_rotated_files: int = DEFAULT_MAX_ROTATED_FILES,
                 compress_rotated: bool = DEFAULT_COMPRESS_ROTATED,
                 summary_write_interval: float = DEFAULT_SUMMARY_WRITE_INTERVAL):
        self.gpu_ids = list(gpu_ids)
        self.output_dir = Path(output_dir)
        self.sample_interval = sample_interval
        self.max_bytes_per_file = max_bytes_per_file
        self.max_rotated_files = max_rotated_files
        self.compress_rotated = compress_rotated
        self.summary_write_interval = summary_write_interval

        # Writers:
        # - gpu writers: one per gpu for DCGM GPU-wide metrics
        # - proc writers: one per gpu for NVML per-process samples
        self.gpu_writers: Dict[int, RotatingNDJSONWriter] = {}
        self.proc_writers: Dict[int, RotatingNDJSONWriter] = {}
        for gid in self.gpu_ids:
            self.gpu_writers[gid] = RotatingNDJSONWriter(self.output_dir / f"gpu_{gid}_gpuwide.ndjson", self.max_bytes_per_file, self.max_rotated_files, self.compress_rotated)
            self.proc_writers[gid] = RotatingNDJSONWriter(self.output_dir / f"gpu_{gid}_processes.ndjson", self.max_bytes_per_file, self.max_rotated_files, self.compress_rotated)

        # running stats: per gpu -> field -> OnlineStats
        self.stats: Dict[int, Dict[str, OnlineStats]] = defaultdict(dict)
        self.proc_stats: Dict[int, Dict[str, OnlineStats]] = defaultdict(dict)  # e.g., per-process mem stats aggregated by gpu

        # DCGM backend
        self.dcgmi_reader = DCGMCLIReader(self.gpu_ids, self.sample_interval, on_sample=self._on_dcgm_sample)

        # NVML sampler
        if pynvml is None:
            print("[CombinedCollector] warning: pynvml not installed; per-process sampling disabled. Install nvidia-ml-py3 to enable.", file=sys.stderr)
            self.nvml_sampler = None
        else:
            self.nvml_sampler = NVMLProcessSampler(self.gpu_ids, self.sample_interval, on_sample=self._on_nvml_sample)

        self._running = False
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._last_summary_write = time.time()

    def _on_dcgm_sample(self, metrics: Dict[str, Any]):
        """Called for each DCGM CSV row parsed. Normalize and write to GPU-wide writer and update stats."""
        with self._lock:
            now = metrics.get("timestamp", time.time())
            if self._start_time is None:
                self._start_time = now
            metrics["time_from_start"] = now - self._start_time
            gid = metrics.get("gpu_id")
            if gid is None:
                if len(self.gpu_ids) == 1:
                    gid = self.gpu_ids[0]
                    metrics["gpu_id"] = gid
                else:
                    # unknown gpu -> drop
                    return
            try:
                self.gpu_writers[int(gid)].write(metrics)
            except Exception:
                pass
            # update numeric fields stats
            for k, v in metrics.items():
                if k in ("timestamp", "time_from_start", "gpu_id"):
                    continue
                if isinstance(v, (int, float)):
                    sdict = self.stats[int(gid)]
                    st = sdict.get(k)
                    if st is None:
                        st = OnlineStats()
                        sdict[k] = st
                    st.add(float(v))
            # periodic summary write
            if time.time() - self._last_summary_write >= self.summary_write_interval:
                try:
                    self._write_summaries()
                except Exception:
                    pass
                self._last_summary_write = time.time()

    def _on_nvml_sample(self, metrics: Dict[str, Any]):
        """Called for each NVML per-gpu process sample."""
        with self._lock:
            now = metrics.get("timestamp", time.time())
            if self._start_time is None:
                self._start_time = now
            metrics["time_from_start"] = now - self._start_time
            gid = metrics.get("gpu_id")
            if gid is None:
                return
            try:
                self.proc_writers[int(gid)].write(metrics)
            except Exception:
                pass
            # Update aggregated per-gpu process memory stats: e.g., sum memory, max per-sample
            # compute per-sample totals
            procs = metrics.get("processes", [])
            total_mem = 0.0
            max_mem = 0.0
            for p in procs:
                mem = p.get("used_memory_mb")
                if mem is None:
                    continue
                total_mem += float(mem)
                if float(mem) > max_mem:
                    max_mem = float(mem)
                # optionally track per-pid stats (not stored here to avoid unbounded memory)
            # record totals
            sdict = self.proc_stats[int(gid)]
            t_st = sdict.get("process_total_memory_mb")
            if t_st is None:
                t_st = OnlineStats(); sdict["process_total_memory_mb"] = t_st
            t_st.add(total_mem)
            m_st = sdict.get("process_max_memory_mb")
            if m_st is None:
                m_st = OnlineStats(); sdict["process_max_memory_mb"] = m_st
            m_st.add(max_mem)

    def start(self):
        if self._running:
            print("[CombinedCollector] already running")
            return
        # sanity checks
        if shutil.which("dcgmi") is None:
            raise RuntimeError("dcgmi not found in PATH. Install NVIDIA DCGM (dcgmi) or ensure it's on PATH.")
        # start DCGM reader
        self.dcgmi_reader.start()
        # start NVML sampler if available
        if self.nvml_sampler is not None:
            self.nvml_sampler.start()
        self._running = True
        print("[CombinedCollector] started")

    def stop(self):
        if not self._running:
            return
        # stop readers
        try:
            self.dcgmi_reader.stop()
        except Exception:
            pass
        if self.nvml_sampler is not None:
            try:
                self.nvml_sampler.stop()
            except Exception:
                pass
        # close writers
        for w in list(self.gpu_writers.values()) + list(self.proc_writers.values()):
            try:
                w.close()
            except Exception:
                pass
        # write final summary
        try:
            self._write_summaries(final=True)
        except Exception:
            pass
        # shutdown nvml explicitly (if we initialized it)
        try:
            if pynvml is not None:
                pynvml.nvmlShutdown()
        except Exception:
            pass
        self._running = False
        print("[CombinedCollector] stopped")

    def _write_summaries(self, final: bool = False):
        """Write compact JSON summaries for each GPU (gpu-wide and proc aggregates)."""
        summary_dir = self.output_dir
        summary_dir.mkdir(parents=True, exist_ok=True)
        for gid in self.gpu_ids:
            out = {"gpu_id": int(gid), "final": bool(final), "last_updated": time.time(), "fields": {}}
            # gpu-wide
            gstats = self.stats.get(gid, {})
            for fname, st in gstats.items():
                out["fields"][fname] = st.summary()
            # proc aggregated
            pstats = self.proc_stats.get(gid, {})
            for fname, st in pstats.items():
                out["fields"][fname] = st.summary()
            out["sample_interval"] = self.sample_interval
            out["sample_count_estimate"] = sum(st.summary().get("count", 0) for st in list(gstats.values()))
            # write atomically
            p = summary_dir / f"summary_gpu_{gid}.json"
            tmp = p.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(out, f, indent=2)
            tmp.replace(p)

    # context manager
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

# -----------------------
# CLI entrypoint
# -----------------------
def parse_args(argv: List[str]):
    gpu_ids = [0]
    output_dir = DEFAULT_OUTPUT_DIR
    sample_interval = DEFAULT_SAMPLE_INTERVAL
    if len(argv) > 1:
        gpu_ids = [int(x.strip()) for x in argv[1].split(",") if x.strip() != ""]
    if len(argv) > 2:
        output_dir = argv[2]
    if len(argv) > 3:
        sample_interval = float(argv[3])
    return gpu_ids, output_dir, sample_interval

def main(argv: List[str]):
    gpu_ids, output_dir, sample_interval = parse_args(argv)
    print(f"Starting combined collector: gpus={gpu_ids} out={output_dir} interval={sample_interval}s")
    collector = CombinedCollector(gpu_ids, output_dir=output_dir, sample_interval=sample_interval)
    try:
        collector.start()
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted -> stopping collector...")
    finally:
        collector.stop()

if __name__ == "__main__":
    main(sys.argv)
