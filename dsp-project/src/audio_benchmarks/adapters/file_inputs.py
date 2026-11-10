"""Explicit local file adapters; no downloads or hidden source changes."""

import hashlib
import json
import math
from pathlib import Path
import wave
import numpy as np
from audio_benchmarks.exceptions import BenchmarkInputError
from audio_benchmarks.models import Interval, Utterance


def fingerprint(path: Path) -> str:
    """SHA-256 of exact source/artifact bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read_audio(path: Path) -> tuple[np.ndarray,int]:
    """Read PCM16 as float channels; decoding additional formats stays outside metrics."""
    try:
        with wave.open(str(path),"rb") as stream:
            if stream.getsampwidth()!=2:
                raise BenchmarkInputError("Benchmark fixtures must be PCM16 WAV")
            values=np.frombuffer(stream.readframes(stream.getnframes()),dtype="<i2").astype(np.float32)/32768
            return values.reshape(-1,stream.getnchannels()),stream.getframerate()
    except (OSError,wave.Error,ValueError) as error:
        raise BenchmarkInputError(f"Cannot read benchmark WAV: {path.name}") from error


def write_audio(path: Path, values: np.ndarray, rate: int) -> None:
    """Generate a fixture without overwriting an existing experiment asset."""
    if path.exists():
        raise BenchmarkInputError("Refusing to overwrite an existing generated fixture")
    path.parent.mkdir(parents=True,exist_ok=True)
    samples=values if values.ndim==2 else values[:,None]
    if not np.isfinite(samples).all() or np.max(np.abs(samples),initial=0)>1:
        raise BenchmarkInputError("Generated samples must be finite and unclipped")
    with wave.open(str(path),"wb") as target:
        target.setparams((samples.shape[1],2,rate,0,"NONE","not compressed"))
        target.writeframes(np.rint(samples*32767).astype("<i2").tobytes())


def read_reference(path: Path, duration: float) -> list[Interval]:
    """Require one RTTM recording; explicitly clip reference intervals to the declared UEM."""
    try:
        result: list[Interval]=[]
        recordings:set[str]=set()
        for line in path.read_text(encoding="utf-8").splitlines():
            fields=line.split()
            if not fields or fields[0]!="SPEAKER":
                continue
            if len(fields)<8:
                raise ValueError("Malformed RTTM")
            recordings.add(fields[1])
            start,end=float(fields[3]),float(fields[3])+float(fields[4])
            if not all(math.isfinite(v) for v in (start,end)) or end<=start:
                raise ValueError("Invalid original RTTM interval")
            if min(duration,end)>max(0,start):
                result.append(Interval(max(0,start),min(duration,end),fields[7]))
        if len(recordings)!=1:
            raise ValueError("RTTM must describe exactly one recording")
        return result
    except (OSError,ValueError,IndexError) as error:
        raise BenchmarkInputError("Invalid benchmark RTTM") from error


def read_transcript(path: Path, duration: float) -> list[Utterance]:
    """Vendor human transcript; boundary-spanning utterance text retained as supplied."""
    try:
        rows=json.loads(path.read_text(encoding="utf-8"))
        result=[]
        for r in rows:
            a,b=r["start_time"],r["end_time"]
            if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (a,b))
                    or b<=a):
                raise ValueError("Invalid original transcript interval")
            if min(duration,b)>max(0,a):result.append(Utterance(max(0,a),min(duration,b),r["speaker"],r["words"]))
        return result
    except (OSError,KeyError,ValueError,TypeError) as error:
        raise BenchmarkInputError("Invalid human-reference transcript") from error
