"""Separate repeated startup/RAM/CPU profiles; not mixed with pipeline-only timings."""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from audio_benchmarks.config import SuiteConfig
from audio_benchmarks.adapters.process_monitor import run_monitored
from audio_benchmarks.adapters.file_inputs import fingerprint,read_audio
from audio_benchmarks.exceptions import BenchmarkConfigurationError


def main() -> None:
    """Run the explicitly declared repeat policy, preserving raw reports and logs."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/benchmark_resources.json")
    args=parser.parse_args();policy=json.loads(args.config.read_text(encoding="utf-8"))
    suite=SuiteConfig.load(ROOT/policy["suite_config"],ROOT)
    if (not isinstance(policy["repeats"],int) or isinstance(policy["repeats"],bool) or not 1<=policy["repeats"]<=10
            or policy["poll_interval_seconds"]<=0 or policy["timeout_seconds"]<=0):
        raise BenchmarkConfigurationError("Invalid bounded resource profile")
    output=suite.output/policy["output_subdirectory"]
    if not output.resolve().is_relative_to(suite.output.resolve()):
        raise BenchmarkConfigurationError("Resource profiles must stay inside the selected run")
    output.mkdir(parents=True,exist_ok=True)
    frozen=output/"policy.json"
    if frozen.exists() and json.loads(frozen.read_text())!=policy:
        raise BenchmarkConfigurationError("Resource policy changed; use a new output folder")
    frozen.write_text(json.dumps(policy,indent=2),encoding="utf-8")
    dataset=next(d for d in suite.document["datasets"] if d["id"]==policy["dataset_id"])
    audio=ROOT/dataset["audio"];values,rate=read_audio(audio);duration=len(values)/rate
    records=[]
    for model in suite.document["models"]:
        for repeat in range(policy["repeats"]):
            tag=f"resource_{model['id']}_{repeat}";path=output/f"{tag}.json"
            if path.exists():record=json.loads(path.read_text(encoding="utf-8"))
            else:
                command=[sys.executable,str(ROOT/"scripts/benchmark_diarization.py"),str(audio),
                    "--reference",str(ROOT/dataset["reference"]),"--config",str(ROOT/model["config"]),
                    "--provider",model["provider"],"--tag",tag,"--output",str(output/"raw")]
                record=run_monitored(command,ROOT,output/f"{tag}.log",policy["timeout_seconds"],policy["poll_interval_seconds"])
                record.update(model=model["id"],repeat=repeat,duration_seconds=duration,input_sha256=fingerprint(audio))
                record["real_time_factor_including_process_start"]=record["elapsed_process_seconds"]/duration
                path.write_text(json.dumps(record,indent=2,allow_nan=False),encoding="utf-8")
            records.append(record)
            print(f"RESOURCE {tag}: {record['elapsed_process_seconds']:.2f}s; peak {record['peak_process_tree_rss_mib']:.1f} MiB; exit {record['exit_code']}",flush=True)
    summary=[]
    for model in suite.document["models"]:
        rows=[r for r in records if r["model"]==model["id"] and r["exit_code"]==0 and not r["timed_out"]]
        if rows:
            summary.append({"model":model["id"],"completed_repeats":len(rows),
                "median_elapsed_seconds":statistics.median(r["elapsed_process_seconds"] for r in rows),
                "min_elapsed_seconds":min(r["elapsed_process_seconds"] for r in rows),
                "max_elapsed_seconds":max(r["elapsed_process_seconds"] for r in rows),
                "median_peak_tree_rss_mib":statistics.median(r["peak_process_tree_rss_mib"] for r in rows),
                "median_sampled_cpu_seconds":statistics.median(r["sampled_cpu_seconds"] for r in rows)})
    (suite.output/"resources.json").write_text(json.dumps({"policy":policy,"records":records,"summary":summary},indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":main()
