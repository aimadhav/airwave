"""Allowlisted report export. Public audio/metrics only; raw logs and paths stay local."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from audio_benchmarks.config import SuiteConfig
from audio_benchmarks.adapters.file_inputs import fingerprint,read_audio
from audio_benchmarks.exceptions import BenchmarkInputError
from audio_benchmarks.services.separation import intelligibility_scores


def collect(config: SuiteConfig, settings: dict[str,Any]) -> dict[str,Any]:
    """Fail explicitly when a stage is missing; never turn absent results into zero errors."""
    stages={name:json.loads((config.output/f"{name}.json").read_text(encoding="utf-8"))
            for name in ("diarization","recognition","recovery","separation","resources")}
    results=[]
    for raw in stages["diarization"]["cases"]:
        row={k:raw[k] for k in ("id","dataset_id","condition","model","status")}
        if raw["status"]=="completed":
            row.update({k:raw[k] for k in ("duration_seconds","elapsed_seconds","real_time_factor","segments",
                "clip_validation_errors","scores","fingerprints","hypothesis")})
        else:row["error_type"]=raw["error_type"]
        results.append(row)
    recognition=[]
    for raw in stages["recognition"]["cases"]:
        row={k:raw[k] for k in ("id","status")}
        if raw["status"]=="completed":
            row.update({k:raw[k] for k in ("dataset_id","condition","elapsed_seconds","duration_seconds",
                "real_time_factor","chronological_serialization_scores","speaker_attributed_scores","hypothesis_text")})
            row["recognized_word_tokens"]=len(raw["words"])
            row["model_sha256"]=raw["signature"]["model"]
            row["human_reference_sha256"]=raw["signature"]["reference"]
        else:row["error_type"]=raw["error_type"]
        recognition.append(row)
    separation=[]
    for raw in stages["separation"]["cases"]:
        row=dict(raw)
        if raw["status"]=="completed":
            folder=config.output/"separation"/raw["id"]
            a,rate=read_audio(folder/"source_0.wav");b,_=read_audio(folder/"source_1.wav")
            mix,_=read_audio(folder/"mixture.wav")
            import numpy as np
            row["actual_source_0_to_1_snr_db"]=float(10*np.log10(np.mean(a.astype(float)**2)/np.mean(b.astype(float)**2)))
            row["gain_label_note"]="relative_level_db is gain applied to source 1 relative to its original clip, not target mixture SNR"
            row["mixture_intelligibility"]=intelligibility_scores((a[:,0],b[:,0]),(mix[:,0],mix[:,0]),[0,1],rate)
            row["audio"]={name:f"benchmark-audio/{raw['id']}/{name}.wav" for name in ("mixture","track_0","track_1")}
        separation.append(row)
    evidence={}
    for name in ("frozen-config.json","environment.json","diarization.json","recognition.json","recovery.json","separation.json","resources.json"):
        evidence[name]=fingerprint(config.output/name)
    source_files=["scripts/run_audio_benchmarks.py","scripts/benchmark_diarization.py","src/dsp/pipeline.py",
        "src/dsp/preprocessing/normalizer.py","src/dsp/diarization/adapters/nemo_native_backend.py",
        "src/dsp/diarization/adapters/sherpa_backend.py","src/recording_viewer/adapters/faster_whisper.py",
        "src/recording_viewer/services/overlap_recovery.py","src/recording_viewer/services/recovery_quality.py",
        "src/audio_benchmarks/services/diarization.py","src/audio_benchmarks/services/transcription.py"]
    document={"schema_version":1,"report_date":settings["report_date"],"run_id":config.document["run_id"],
        "datasets":[{k:v for k,v in d.items() if k in {"id","source_url","license","note"}} for d in config.document["datasets"]],
        "diarization":results,"recognition":recognition,"recovery":stages["recovery"],"separation":separation,
        "resources":stages["resources"],"environment":json.loads((config.output/"environment.json").read_text()),
        "evidence_sha256":evidence,"code_sha256":{p:fingerprint(ROOT/p) for p in source_files},
        "sample_audio":settings["sample_asset"],"primary_sources":settings["primary_sources"],
        "limits":["Small development/test-fixture suite, not unseen held-out data or an official leaderboard.",
            "Four real excerpts have different RTTM protocols. Do not pool noise/rate variants as independent meetings.",
            "DER/JER and cpWER are local implementations, analytically tested and DER cross-checked against the existing local scorer; not certified dscore/MeetEval outputs.",
            "ASR WER/CER uses supplied human text, but chronological serialization is overlap-order-dependent. cpWER handles speaker permutation; it is not time-constrained.",
            "The first human utterance starts -0.22 s; boundary time is clamped to zero and supplied text retained.",
            "Two controlled remixes from one eligible source pair are not a real-overlap generalization benchmark; stems can contain background noise.",
            "Mixed baseline STOI uses the saved PCM16 mixture/stems; separated STOI uses floating-point model outputs. Quantization can cause tiny numerical differences.",
            "Process-tree RSS is sampled and can double-count shared pages. Three startup-inclusive repeats are not p95 latency or cold-cache measurements.",
            "No isolated sources for real meeting overlap: real-overlap SI-SDR/PESQ/MOS are unavailable. No per-word human timings: overlap-only WER/tcpWER are unavailable.",
            "No models were retrained and no product thresholds were tuned after viewing these results."]}
    serialized=json.dumps(document,ensure_ascii=False,allow_nan=False)
    if any(marker in serialized for marker in ("C:/Users/","C:\\\\Users\\\\","secret-token")):
        raise BenchmarkInputError("Public report contains a private local path")
    return document


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/benchmark_report.json")
    parser.add_argument("--prepare-site",action="store_true",help="Copy finalized report/assets into the already-open selected Site")
    args=parser.parse_args();settings=json.loads(args.config.read_text(encoding="utf-8"))
    config=SuiteConfig.load(ROOT/settings["suite_config"],ROOT)
    document=collect(config,settings)
    output=(ROOT/settings["output_json"]).resolve()
    if not output.is_relative_to((ROOT/"web/recording-viewer").resolve()):raise BenchmarkInputError("Report must stay in the website source folder")
    output.write_text(json.dumps(document,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8",newline="\n")
    csv_path=config.output/"diarization-summary.csv"
    with csv_path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["dataset","condition","model","status","DER_no_collar","DER_250ms","JER_no_collar","overlap_F1","elapsed_seconds","RTF"])
        for row in document["diarization"]:
            if row["status"]!="completed":writer.writerow([row["dataset_id"],row["condition"],row["model"],row["status"]]);continue
            s=row["scores"]["collar_0_overlap_True"]
            writer.writerow([row["dataset_id"],row["condition"],row["model"],row["status"],s["der"],row["scores"]["collar_0.25_overlap_True"]["der"],s["jer"],s["overlap_detection"]["f1"],row["elapsed_seconds"],row["real_time_factor"]])
    lines=["# Comprehensive local audio benchmark","",f"Report date: {settings['report_date']}; run: `{document['run_id']}`.","",
        "Four real annotated excerpts; 24 diarization conditions/systems. No official leaderboard or unseen-test claim.","",
        "## Clean audio: exact local DER including overlap","","| Dataset | Model | DER, no collar | DER, ±250 ms | RTF |","|---|---|---:|---:|---:|"]
    for row in document["diarization"]:
        if row["condition"]=="clean" and row["status"]=="completed":
            lines.append(f"| {row['dataset_id']} | {row['model']} | {row['scores']['collar_0_overlap_True']['der']:.2%} | {row['scores']['collar_0.25_overlap_True']['der']:.2%} | {row['real_time_factor']:.3f} |")
    lines += ["","## Human-reference transcription","","| Condition | Serialized WER | CER | cpWER, Nemotron |","|---|---:|---:|---:|"]
    for row in document["recognition"]:
        if row["status"]=="completed":lines.append(f"| {row['condition']} | {row['chronological_serialization_scores']['wer']['error_rate']:.2%} | {row['chronological_serialization_scores']['cer']['error_rate']:.2%} | {row['speaker_attributed_scores']['native']['cpwer']:.2%} |")
    lines += ["","## Controlled separation",""]
    for row in document["separation"]:
        if row["status"]=="completed":lines.append(f"- {row['id']}: mean SI-SDRi {row['metrics']['mean_si_sdri_db']:.2f} dB; mean STOI {row['intelligibility']['mean_stoi']:.3f}. Oracle source permutation, not verified speaker attribution.")
    lines += ["","## Guarded overlap recovery",""]
    for row in document["recovery"]["cases"]:
        if row["status"]=="completed":lines.append(f"- {row['dataset_id']}: {row['counts']}; cpWER before {row['before_cpwer']['cpwer']:.2%}, after {row['after_cpwer']['cpwer']:.2%}. No accepted core is not evidence that no real overlap exists.")
    lines += ["","## Limits", "",*(f"- {item}" for item in document["limits"]),"","## Run again","", "```powershell",
        "python scripts/run_audio_benchmarks.py", "python scripts/profile_audio_benchmarks.py", "python scripts/export_audio_benchmark_report.py", "```","",
        "Use the project runtime and installed dependencies. The saved matrix and input/config fingerprints are checked before reuse; use a new run ID for changed experiments. No model downloads occur during inference."]
    (ROOT/settings["output_markdown"]).write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
    if args.prepare_site:
        site=(ROOT/settings["site_directory"]).resolve()
        if not site.is_relative_to((ROOT/"data/sites").resolve()):raise BenchmarkInputError("Unexpected Site checkout")
        manifest=json.loads((site/".openai/hosting.json").read_text(encoding="utf-8"))
        if not manifest.get("project_id") or manifest.get("static",{}).get("directory")!="dist":raise BenchmarkInputError("Open the existing static Site first")
        target=site/"dist";source=ROOT/"web/recording-viewer"
        encoded=json.dumps(document,ensure_ascii=False,allow_nan=False).replace("<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")
        html=(source/"benchmarks.html").read_text(encoding="utf-8").replace("__BENCHMARK_DATA__",encoded)
        (target/"benchmarks.html").write_text(html,encoding="utf-8",newline="\n")
        for name in ("benchmarks.js","benchmarks.json"):
            shutil.copyfile(source/name,target/name)
        shutil.copyfile(csv_path,target/"benchmark-summary.csv")
        audio=target/"benchmark-audio";audio.mkdir(exist_ok=True)
        primary=next(d for d in config.document["datasets"] if d["id"]=="ami60")
        shutil.copyfile(ROOT/primary["audio"],audio/"ami60.wav")
        for row in document["separation"]:
            if row["status"]!="completed":continue
            destination=audio/row["id"];destination.mkdir(exist_ok=True)
            for name in ("mixture","track_0","track_1"):shutil.copyfile(config.output/"separation"/row["id"]/f"{name}.wav",destination/f"{name}.wav")
        # Add links once without regenerating or replacing the saved podcast artifacts.
        for filename,anchor in (("index.html",'<div class="masthead-actions">'),("examples.html",'<div class="report-links">'),("pipeline.html",'<nav aria-label="Website">')):
            path=target/filename;existing=path.read_text(encoding="utf-8")
            if 'href="benchmarks.html"' not in existing:
                link='<a class="button" href="benchmarks.html">Benchmarks</a>' if filename!="pipeline.html" else '<a href="benchmarks.html">Benchmarks</a>'
                if anchor not in existing:raise BenchmarkInputError(f"Missing expected navigation in {filename}")
                path.write_text(existing.replace(anchor,anchor+link,1),encoding="utf-8",newline="\n")
    print(json.dumps({"diarization_runs":len(document["diarization"]),"ASR_conditions":len(document["recognition"]),
        "separation_cases":len(document["separation"]),"resource_repeats":len(document["resources"]["records"])},indent=2),flush=True)


if __name__=="__main__":main()
