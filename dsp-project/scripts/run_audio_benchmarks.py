"""Run a frozen local-model matrix, score human references, and preserve failures."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from math import gcd
from pathlib import Path
import platform
import os
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import numpy as np
from scipy.signal import resample_poly
from audio_benchmarks.config import SuiteConfig
from audio_benchmarks.adapters.file_inputs import fingerprint,read_audio,read_reference,read_transcript,write_audio
from audio_benchmarks.exceptions import BenchmarkError,BenchmarkProviderError
from audio_benchmarks.models import Interval,Utterance
from audio_benchmarks.services.diarization import activity_cells,score_diarization
from audio_benchmarks.services.transcription import cpwer,text_scores
from audio_benchmarks.services.separation import score_separation,intelligibility_scores


def save(path: Path, value: object) -> None:
    """Atomically persist generated results, not editable program source."""
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8",newline="\n")
    temporary.replace(path)


def canonical(path: Path) -> tuple[np.ndarray,int]:
    """Same mono/rate contract as the recognizer; do not peak-normalize ASR input."""
    values,rate=read_audio(path)
    mono=values.mean(axis=1)
    divisor=gcd(rate,16000)
    if rate!=16000:
        mono=resample_poly(mono,16000//divisor,rate//divisor).astype(np.float32)
    return mono,16000


def cases(config: SuiteConfig) -> list[dict[str,object]]:
    """Generate deterministic transforms before inference; all retain original reference times."""
    result:list[dict[str,object]]=[]
    c=config.document
    for dataset in c["datasets"]:
        source=config.root/dataset["audio"]
        values,rate=read_audio(source)
        for condition in c["conditions"] if dataset["conditions"] else ["clean"]:
            path=source
            if condition!="clean":
                path=config.output/"inputs"/f"{dataset['id']}_{condition}.wav"
                signature={"source_sha256":fingerprint(source),"condition":condition,"seed":c["noise_seed"],
                    "quiet_gain":c["quiet_gain"],"snr":c["noise_snr_db"],"resample_rate":c["resample_rate"]}
                metadata=path.with_suffix(".json")
                if path.exists():
                    recorded=json.loads(metadata.read_text(encoding="utf-8"))
                    if recorded["signature"]!=signature or recorded["sha256"]!=fingerprint(path):
                        raise BenchmarkProviderError("Generated fixture cache differs; choose a new run directory")
                else:
                    output=values.copy();out_rate=rate
                    if condition=="quiet":output*=c["quiet_gain"]
                    elif condition.startswith("noise_"):
                        rng=np.random.default_rng(c["noise_seed"])
                        noise=rng.standard_normal(output.shape).astype(np.float32)
                        rms=float(np.sqrt(np.mean(output.astype(np.float64)**2)))
                        noise*=rms/(10**(c["noise_snr_db"][condition]/20)*np.sqrt(np.mean(noise.astype(np.float64)**2)))
                        output+=noise
                        output/=max(1,float(np.max(np.abs(output)))/.98)
                    elif condition=="stereo_48k":
                        out_rate=c["resample_rate"];factor=gcd(rate,out_rate)
                        mono=resample_poly(output.mean(axis=1),out_rate//factor,rate//factor).astype(np.float32)
                        output=np.column_stack([mono,mono])
                        output/=max(1,float(np.max(np.abs(output)))/.98)
                    write_audio(path,output,out_rate)
                    save(metadata,{"signature":signature,"sha256":fingerprint(path)})
            result.append({"id":f"{dataset['id']}_{condition}","dataset":dataset,"condition":condition,
                           "audio":path,"duration":len(values)/rate})
    return result


def diarization(config: SuiteConfig) -> dict[str,object]:
    """Run existing pipeline entry point in isolated processes; never erase previous runs."""
    from scripts.benchmark_diarization import load_benchmark_config
    outputs:list[dict[str,object]]=[]
    matrix=cases(config)
    for case in matrix:
        duration=float(case["duration"]);dataset=case["dataset"]
        reference=read_reference(config.root/dataset["reference"],duration)
        for model in config.document["models"]:
            experiment=f"{case['id']}_{model['id']}"
            folder=config.output/"raw"/experiment
            report_path=folder/"report.json"
            expected=load_benchmark_config(config.root/model["config"],model["provider"],-1,None,None)
            try:
                if not report_path.exists():
                    log=config.output/"logs"/f"{experiment}.log";log.parent.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,str(ROOT/"scripts/benchmark_diarization.py"),str(case["audio"]),
                        "--reference",str(config.root/dataset["reference"]),"--config",str(config.root/model["config"]),
                        "--provider",model["provider"],"--tag",experiment,"--output",str(config.output/"raw")]
                    with log.open("w",encoding="utf-8") as stream:
                        subprocess.run(command,cwd=config.root,stdout=stream,stderr=subprocess.STDOUT,
                            check=True,shell=False,timeout=config.document["runtime_timeout_seconds"])
                raw=json.loads(report_path.read_text(encoding="utf-8"))
                if (raw["fingerprints"]["input"]["sha256"]!=fingerprint(case["audio"])
                        or raw["fingerprints"]["reference"]["sha256"]!=fingerprint(config.root/dataset["reference"])
                        or raw["local_config"]!=asdict(expected.local_model) or raw["provider"]!=model["provider"]):
                    raise BenchmarkProviderError("Stored inference does not match source/configuration")
                intervals=[Interval(r["start_time"],r["end_time"],r["preliminary_speaker_id"]) for r in raw["segments"]]
                scores={f"collar_{collar:g}_overlap_{include}":score_diarization(reference,intervals,duration,collar,include)
                    for collar in config.document["collars_seconds"] for include in config.document["score_overlap_modes"]}
                outputs.append({"id":experiment,"dataset_id":dataset["id"],"condition":case["condition"],
                    "model":model["id"],"status":"completed","duration_seconds":duration,
                    "elapsed_seconds":raw["elapsed_seconds"],"real_time_factor":raw["real_time_factor"],
                    "segments":len(intervals),"clip_validation_errors":raw["clip_validation_errors"],"scores":scores,
                    "fingerprints":{k:{"sha256":v["sha256"],"size_bytes":v["size_bytes"]} for k,v in raw["fingerprints"].items()},
                    "effective_config":raw["local_config"],"hypothesis":[asdict(r) for r in intervals]})
                score=scores["collar_0_overlap_True"]
                print(f"{experiment}: DER {score['der']:.4f}, overlap F1 {score['overlap_detection']['f1']}, RTF {raw['real_time_factor']:.3f}",flush=True)
            except (BenchmarkError,OSError,ValueError,subprocess.SubprocessError) as error:
                outputs.append({"id":experiment,"dataset_id":dataset["id"],"condition":case["condition"],
                    "model":model["id"],"status":"failed","error_type":type(error).__name__,
                    "detail":"See private run log; no missing result is counted as success"})
                print(f"FAILED {experiment}: {type(error).__name__}",flush=True)
            save(config.output/"diarization.json",{"cases":outputs,"matrix_complete":len(outputs)==len(matrix)*len(config.document["models"])})
    return {"cases":outputs}


def recognition(config: SuiteConfig) -> dict[str,object]:
    """Reuse the installed Whisper model; only human labels produce word accuracy."""
    from recording_viewer.adapters.faster_whisper import FasterWhisperTranscriber
    from recording_viewer.config import RecordingViewerConfig
    from recording_viewer.models import TranscriptWord,VoiceRegion
    from recording_viewer.services.alignment import align_words
    policy=RecordingViewerConfig.load(config.root/config.document["recognition_config"])
    recognizer=FasterWhisperTranscriber(policy)
    output=[]
    for case in cases(config):
        dataset=case["dataset"]
        if dataset["id"] not in config.document["recognition_datasets"]:continue
        reference=read_transcript(config.root/dataset["transcript"],float(case["duration"]))
        path=config.output/"recognition"/f"{case['id']}.json"
        signature={"audio":fingerprint(case["audio"]),"reference":fingerprint(config.root/dataset["transcript"]),
            "model":fingerprint(config.root/policy.model_directory/"model.bin"),"policy":asdict(policy),
            "adapter":fingerprint(ROOT/"src/recording_viewer/adapters/faster_whisper.py")}
        try:
            samples,rate=canonical(case["audio"])
            if path.exists():
                row=json.loads(path.read_text(encoding="utf-8"))
                if row["signature"]!=signature:raise BenchmarkProviderError("ASR cache signature mismatch")
                words=[TranscriptWord(**w) for w in row["words"]]
            else:
                started=perf_counter();words=recognizer.transcribe_samples(samples,rate);elapsed=perf_counter()-started
                row={"id":case["id"],"dataset_id":dataset["id"],"condition":case["condition"],"status":"completed",
                    "signature":signature,"elapsed_seconds":elapsed,"duration_seconds":float(case["duration"]),
                    "real_time_factor":elapsed/float(case["duration"]),"words":[asdict(w) for w in words]}
            expected=" ".join(r.text for r in sorted(reference,key=lambda r:(r.start,r.end)))
            hypothesis="".join(w.text for w in words)
            row["chronological_serialization_scores"]=text_scores(expected,hypothesis)
            row["reference_text"]=expected;row["hypothesis_text"]=hypothesis
            row["speaker_attributed_scores"]={}
            for model in config.document["models"]:
                raw=config.output/"raw"/f"{case['id']}_{model['id']}"/"report.json"
                if not raw.exists():continue
                segments=json.loads(raw.read_text(encoding="utf-8"))["segments"]
                regions=[VoiceRegion(r["segment_id"],r["preliminary_speaker_id"],r["start_time"],r["end_time"]) for r in segments]
                turns=align_words(words,regions,policy.minimum_word_coverage)
                row["speaker_attributed_scores"][model["id"]]=cpwer(reference,[Utterance(t.start,t.end,t.speaker,t.text) for t in turns])
            save(path,row);output.append(row)
            print(f"ASR {case['id']}: serialized WER {row['chronological_serialization_scores']['wer']['error_rate']:.3f}",flush=True)
        except (BenchmarkError,OSError,ValueError,RuntimeError) as error:
            output.append({"id":case["id"],"status":"failed","error_type":type(error).__name__})
    result={"cases":output,"notes":["Human transcript, not ASR-generated silver reference.",
        "Chronological WER is overlap-order-dependent; cpWER scores per-speaker streams separately.",
        "First vendor reference utterance begins -0.22 s: start clipped to zero; supplied text retained, with possible partial-word boundary bias.",
        "Model initialization is included in the first ASR call; later calls reuse the model. No latency percentile claim."]}
    save(config.output/"recognition.json",result)
    return result


def recovery(config: SuiteConfig) -> dict[str,object]:
    """Evaluate the ACTUAL guarded recovery, not an oracle speaker timeline."""
    from recording_viewer.adapters.faster_whisper import FasterWhisperTranscriber
    from recording_viewer.adapters.local_separator import LocalSpeechSeparator
    from recording_viewer.adapters.voice_embedder import OnnxVoiceEmbedder
    from recording_viewer.config import RecordingViewerConfig
    from recording_viewer.models import TranscriptWord,VoiceRegion
    from recording_viewer.recovery_config import RecoveryConfig
    from recording_viewer.services.overlap_recovery import OverlapRecovery
    from recording_viewer.services.recovery_timeline import splice_recovery
    policy=RecoveryConfig.load(config.root/config.document["recovery_config"])
    viewer=RecordingViewerConfig.load(config.root/config.document["recognition_config"])
    results=[]
    for dataset in config.document["datasets"]:
        if dataset["id"] not in config.document["recognition_datasets"]:continue
        target=config.output/"recovery.json"
        if target.exists():return json.loads(target.read_text(encoding="utf-8"))
        try:
            raw=json.loads((config.output/"raw"/f"{dataset['id']}_clean_native"/"report.json").read_text(encoding="utf-8"))
            asr=json.loads((config.output/"recognition"/f"{dataset['id']}_clean.json").read_text(encoding="utf-8"))
            samples,rate=canonical(config.root/dataset["audio"])
            regions=[VoiceRegion(r["segment_id"],r["preliminary_speaker_id"],r["start_time"],r["end_time"]) for r in raw["segments"]]
            words=[TranscriptWord(**r) for r in asr["words"]]
            reference=read_transcript(config.root/dataset["transcript"],len(samples)/rate)
            speaker_reference=read_reference(config.root/dataset["reference"],len(samples)/rate)
            service=OverlapRecovery(policy,LocalSpeechSeparator(config.root,policy,config.output/"recovery-scratch"),
                OnnxVoiceEmbedder(policy),FasterWhisperTranscriber(viewer))
            started=perf_counter();result=service.recover(samples,rate,regions);elapsed=perf_counter()-started
            turns=splice_recovery(words,regions,result,viewer.minimum_word_coverage)
            reports=[]
            for r in result.reports:
                clipped=[Interval(max(r.window.start,x.start)-r.window.start,min(r.window.end,x.end)-r.window.start,x.speaker)
                    for x in speaker_reference if min(r.window.end,x.end)>max(r.window.start,x.start)]
                overlap=sum(d for d,refs,_ in activity_cells(clipped,[],r.window.end-r.window.start) if len(refs)>=2)
                reports.append({"window":asdict(r.window),"status":r.status,
                    "reason":r.reason if r.status!="failed" else "provider_failure", "metrics":r.metrics,
                    "reference_overlap_fraction":overlap/(r.window.end-r.window.start)})
            row={"dataset_id":dataset["id"],"status":"completed","elapsed_seconds":elapsed,
                "counts":dict(Counter(r.status for r in result.reports)),"reasons":dict(Counter(r["reason"] for r in reports)),
                "reference_counts":result.reference_counts,"reports":reports,
                "before_cpwer":asr["speaker_attributed_scores"]["native"],
                "after_cpwer":cpwer(reference,[Utterance(t.start,t.end,t.speaker,t.text) for t in turns])}
            results.append(row)
            print(f"RECOVERY {dataset['id']}: {row['counts']}",flush=True)
        except Exception as error:
            results.append({"dataset_id":dataset["id"],"status":"failed","error_type":type(error).__name__})
            print(f"RECOVERY failed: {type(error).__name__}",flush=True)
    result={"cases":results,"note":"Quality-gate decisions, not oracle recovery. cpWER against supplied human text; no per-word human timestamps for overlap-only WER."}
    save(config.output/"recovery.json",result);return result


def separation(config: SuiteConfig) -> dict[str,object]:
    """Controlled source remix: exact waveform references, not a real-overlap benchmark."""
    from recording_viewer.adapters.local_separator import LocalSpeechSeparator
    from recording_viewer.recovery_config import RecoveryConfig
    spec=config.document["separation"];policy=RecoveryConfig.load(config.root/config.document["recovery_config"])
    dataset=next(d for d in config.document["datasets"] if d["id"]=="vbx_es2005a_150s")
    samples,rate=canonical(config.root/dataset["audio"])
    reference=read_reference(config.root/dataset["reference"],len(samples)/rate)
    # Use exact event boundaries to choose longest exclusive regions BEFORE separation.
    events=sorted({0.0,len(samples)/rate}|{v for r in reference for v in (r.start,r.end)})
    spans=[]
    for a,b in zip(events,events[1:]):
        active={r.speaker for r in reference if r.start<=(a+b)/2<r.end}
        if len(active)==1:
            speaker=next(iter(active))
            if spans and spans[-1][1]==a and spans[-1][2]==speaker:spans[-1]=(spans[-1][0],b,speaker)
            else:spans.append((a,b,speaker))
    selected={}
    for a,b,speaker in sorted(spans,key=lambda r:-(r[1]-r[0])):
        if b-a>=spec["minimum_reference_seconds"]:
            selected.setdefault(speaker,(a+spec["guard_seconds"],b-spec["guard_seconds"]))
    from itertools import combinations
    pairs=list(combinations(sorted(selected),2))[:spec["max_pairs"]]
    separator=LocalSpeechSeparator(config.root,policy,config.output/"separation-scratch")
    output=[]
    for pair_id,pair in enumerate(pairs):
        for level in spec["relative_levels_db"]:
            folder=config.output/"separation"/f"pair_{pair_id}_level_{level}"
            saved=folder/"result.json"
            if saved.exists():output.append(json.loads(saved.read_text(encoding="utf-8")));continue
            try:
                n=round(spec["clip_seconds"]*rate)
                sources=[samples[round(selected[s][0]*rate):round(selected[s][0]*rate)+n].copy() for s in pair]
                sources[1]*=10**(level/20)
                mixture=sources[0]+sources[1]
                gain=min(1,.95/float(np.max(np.abs(mixture))))
                sources=[x*gain for x in sources];mixture*=gain
                started=perf_counter();result=separator.separate(mixture,rate);elapsed=perf_counter()-started
                row={"id":folder.name,"status":"completed","speakers":pair,"relative_level_db":level,
                    "source_dataset":dataset["id"],"source_starts":[selected[s][0] for s in pair],
                    "duration_seconds":spec["clip_seconds"],"elapsed_seconds":elapsed,
                    "real_time_factor":elapsed/spec["clip_seconds"],"metrics":score_separation(mixture,tuple(sources),result.tracks)}
                if row["metrics"].get("defined"):
                    row["intelligibility"]=intelligibility_scores(tuple(sources),result.tracks,row["metrics"]["track_to_source"],rate)
                for name,values in [("mixture",mixture),*( (f"source_{i}",v) for i,v in enumerate(sources)),
                    *((f"track_{i}",v) for i,v in enumerate(result.tracks))]:write_audio(folder/f"{name}.wav",values,rate)
                save(saved,row);output.append(row)
                print(f"SEPARATION {folder.name}: SI-SDRi {row['metrics'].get('mean_si_sdri_db')}",flush=True)
            except Exception as error:
                output.append({"id":folder.name,"status":"failed","error_type":type(error).__name__})
                print(f"SEPARATION {folder.name} failed: {type(error).__name__}",flush=True)
    result={"cases":output,"note":"Synthetic mixtures of different reference-exclusive meeting spans; stems can contain background noise. Not real-overlap accuracy or speaker identification. Pairs chosen by reference duration before inference."}
    save(config.output/"separation.json",result);return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/benchmark_suite.json")
    parser.add_argument("--phase",choices=["all","diarization","recognition","recovery","separation"],default="all")
    args=parser.parse_args();config=SuiteConfig.load(args.config,ROOT)
    config.output.mkdir(parents=True,exist_ok=True)
    frozen=config.output/"frozen-config.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8"))!=config.document:
        raise BenchmarkProviderError("The run configuration changed; use a new run_id/output directory")
    save(frozen,config.document)
    versions={}
    for package in ("numpy","scipy","sherpa-onnx","faster-whisper","ctranslate2","psutil"):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    save(config.output/"environment.json",{"python":platform.python_version(),"platform":platform.platform(),
        "cpu_logical_count":os.cpu_count(),"versions":versions,"timing":"serial single-pass runs; disk cache uncontrolled; diarization includes model init/preprocessing/clip IO but excludes Python startup; no p95 latency claim"})
    for name,function in (("diarization",diarization),("recognition",recognition),("recovery",recovery),("separation",separation)):
        if args.phase in ("all",name):function(config)


if __name__=="__main__":main()
