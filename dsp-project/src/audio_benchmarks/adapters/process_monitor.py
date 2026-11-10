"""Optional process-tree resource instrumentation; never inspect unrelated processes."""

from pathlib import Path
import subprocess
from time import perf_counter,sleep
from typing import Any

from audio_benchmarks.exceptions import BenchmarkProviderError


def run_monitored(command: list[str], cwd: Path, log: Path, timeout: float,
                  poll_seconds: float) -> dict[str,Any]:
    """Sample owned process/descendant RSS and CPU; include interpreter startup in time."""
    try:
        import psutil
    except ImportError as error:
        raise BenchmarkProviderError("Resource profiling requires optional psutil") from error
    if poll_seconds<=0 or timeout<=0:
        raise BenchmarkProviderError("Resource sampling bounds must be positive")
    log.parent.mkdir(parents=True,exist_ok=True)
    peak=0;cpus:dict[tuple[int,float],float]={};count=0;timed_out=False
    started=perf_counter()
    with log.open("w",encoding="utf-8") as stream:
        process=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,shell=False)
        parent=psutil.Process(process.pid)
        while process.poll() is None:
            processes=[parent]
            try:processes+=parent.children(recursive=True)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            rss=0
            for child in processes:
                try:
                    rss+=child.memory_info().rss
                    times=child.cpu_times()
                    key=(child.pid,child.create_time())
                    cpus[key]=max(cpus.get(key,0),times.user+times.system)
                except (psutil.NoSuchProcess,psutil.AccessDenied):continue
            peak=max(peak,rss);count+=1
            if perf_counter()-started>timeout:
                timed_out=True
                for child in reversed(processes):
                    try:child.terminate()
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                break
            sleep(poll_seconds)
        try:exit_code=process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill();exit_code=process.wait(timeout=10)
    elapsed=perf_counter()-started
    return {"exit_code":exit_code,"timed_out":timed_out,"elapsed_process_seconds":elapsed,
        "peak_process_tree_rss_mib":peak/1024**2 if count else None,
        "sampled_cpu_seconds":sum(cpus.values()),"samples":count,"poll_interval_seconds":poll_seconds,
        "scope":"sum of sampled parent/descendant RSS; shared pages may be counted more than once; brief process peaks may be missed"}
