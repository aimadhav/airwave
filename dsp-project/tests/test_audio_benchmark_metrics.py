"""Analytical references and mocked providers; no downloads or real models."""

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pytest

from audio_benchmarks.adapters.mock_adapter import MockDiarizationProvider
from audio_benchmarks.adapters.file_inputs import read_reference,read_transcript,write_audio,read_audio
from audio_benchmarks.config import SuiteConfig
from audio_benchmarks.exceptions import BenchmarkInputError,BenchmarkConfigurationError
from audio_benchmarks.models import Interval,Utterance
from audio_benchmarks.services.diarization import score_diarization,detection_counts
from audio_benchmarks.services.transcription import cpwer,text_scores,edit_counts,words
from audio_benchmarks.services.separation import si_sdr,score_separation
from dsp.evaluation import TimedSpeaker,diarization_error

ROOT=Path(__file__).resolve().parents[1]


def test_permutation_overlap_and_duplicate_regions_are_invariant() -> None:
    refs=[Interval(0,2,"a"),Interval(1,3,"b")]
    hyps=[Interval(0,2,"z"),Interval(1,3,"y"),Interval(.2,1.8,"z")]
    score=score_diarization(refs,hyps,4)
    assert score["der"]==score["jer"]==0
    assert score["reference_speaker_seconds"]==4
    assert score["reference_overlap_seconds"]==1
    assert score["overlap_detection"]["f1"]==1
    assert score["speech_detection"]["true_negative_seconds"]==1


def test_miss_confusion_false_alarm_and_full_uem() -> None:
    refs=[Interval(0,1,"a"),Interval(1,2,"b")]
    score=score_diarization(refs,[Interval(0,3,"x")],4)
    assert score["der"]==1
    assert score["confusion_seconds"]==score["false_alarm_seconds"]==1
    assert score["speech_detection"]["false_positive_seconds"]==1
    assert score["speech_detection"]["true_negative_seconds"]==1
    assert score_diarization(refs,[],4)["der"]==1
    assert score_diarization(refs,[],4)["jer"]==1


def test_empty_reference_is_undefined_not_perfect() -> None:
    score=score_diarization([],[Interval(0,1,"x")],2)
    assert score["der"] is None and score["jer"] is None
    assert score["false_alarm_seconds"]==1
    assert score["speech_detection"]["precision"]==0
    assert score["speech_detection"]["recall"] is None
    assert detection_counts(0,0,0,10)["f1"] is None


def test_explicit_half_width_collar_and_overlap_exclusion() -> None:
    refs=[Interval(0,2,"a")]
    assert score_diarization(refs,[Interval(.1,1.9,"x")],3,.25)["der"]==0
    score=score_diarization([Interval(0,2,"a"),Interval(1,3,"b")],[Interval(0,3,"x")],3,overlap_included=False)
    assert score["scored_wall_seconds"]==2
    assert score["reference_overlap_seconds"]==0
    assert score["overlap_included"] is False


@pytest.mark.parametrize("collar",[0,.25])
def test_der_cross_checks_existing_public_local_scorer(collar: float) -> None:
    refs=[Interval(0,2,"a"),Interval(1,4,"b")]
    hyps=[Interval(.1,2.5,"x"),Interval(1.5,3.5,"y"),Interval(4.1,4.6,"x")]
    old=diarization_error([TimedSpeaker(**asdict(r)) for r in refs],[TimedSpeaker(**asdict(r)) for r in hyps],collar)
    score=score_diarization(refs,hyps,5,collar)
    for key in ("der","miss_seconds","false_alarm_seconds","confusion_seconds","reference_speaker_seconds"):
        assert score[key]==pytest.approx(old[key])


@pytest.mark.parametrize("args",[(0,1,"a"),(-1,2,"a"),(1,1,"a"),(0,float('nan'),"a"),(0,1,"")])
def test_interval_validation(args: tuple) -> None:
    if args==(0,1,"a"):
        assert Interval(*args).end==1
    else:
        with pytest.raises(BenchmarkInputError):Interval(*args)


def test_out_of_bounds_and_invalid_collar_fail_explicitly() -> None:
    with pytest.raises(BenchmarkInputError):score_diarization([Interval(0,2,"a")],[],1)
    with pytest.raises(BenchmarkInputError):score_diarization([],[],1,-.1)


def test_word_edits_unicode_normalization_and_cer_policy() -> None:
    assert words("Hello, WE’RE_here!")==["hello","we're","here"]
    score=edit_counts(["a","b","c"],["a","d","c","e"])
    assert score["errors"]==2 and score["substitutions"]==1 and score["insertions"]==1
    assert text_scores("Hello, WORLD!","hello world")["wer"]["error_rate"]==0
    assert text_scores("", "invented")["wer"]["error_rate"] is None
    assert text_scores("ab","ac")["cer"]["error_rate"]==.5


def test_cpwer_permutation_and_extra_stream_insertions() -> None:
    refs=[Utterance(0,1,"a","hello there"),Utterance(1,2,"b","good day")]
    hyps=[Utterance(0,1,"y","hello there"),Utterance(1,2,"x","good day")]
    assert cpwer(refs,hyps)["cpwer"]==0
    extra=hyps+[Utterance(2,3,"z","invented words")]
    score=cpwer(refs,extra)
    assert score["errors"]==2 and score["cpwer"]==.5
    assert cpwer(refs,[])["cpwer"]==1
    assert cpwer([],hyps)["cpwer"] is None


def test_source_permutation_and_scale_invariance() -> None:
    rng=np.random.default_rng(123)
    a,b=rng.normal(size=16000),rng.normal(size=16000)
    mixture=(a+b)/2
    score=score_separation(mixture,(a,b),(2*b,3*a))
    assert score["track_to_source"]==[1,0]
    assert score["mean_si_sdri_db"]>100
    assert si_sdr(np.zeros_like(a),a) is None
    assert score_separation(mixture,(a,b),(np.zeros_like(a),b))["defined"] is False
    with pytest.raises(BenchmarkInputError):si_sdr(a,b[:-1])


def test_file_adapters_clamp_boundary_annotations_and_refuse_overwrite(tmp_path: Path) -> None:
    audio=tmp_path/"fixture.wav"
    write_audio(audio,np.zeros(1600,dtype=np.float32),16000)
    values,rate=read_audio(audio)
    assert values.shape==(1600,1) and rate==16000
    with pytest.raises(BenchmarkInputError):write_audio(audio,values,rate)
    ref=tmp_path/"ref.rttm"
    ref.write_text("SPEAKER one 1 -0.1 0.2 <NA> <NA> a <NA> <NA>")
    assert read_reference(ref,.1)==[Interval(0,.1,"a")]
    ref.write_text(ref.read_text()+"\nSPEAKER two 1 0 .1 <NA> <NA> b <NA> <NA>")
    with pytest.raises(BenchmarkInputError):read_reference(ref,.1)
    transcript=tmp_path/"ref.json"
    transcript.write_text(json.dumps([{"start_time":-.1,"end_time":.1,"speaker":"a","words":"hello"}]))
    assert read_transcript(transcript,.1)[0].start==0


def test_mock_provider_needs_no_files_or_models(tmp_path: Path) -> None:
    result=MockDiarizationProvider((Interval(0,1,"x"),)).predict(tmp_path/"not-needed.wav",tmp_path,"test")
    assert result.elapsed_seconds==0 and len(result.intervals)==1


def test_configuration_rejects_path_escape_and_invalid_scope(tmp_path: Path) -> None:
    doc=json.loads((ROOT/"configs/benchmark_suite.json").read_text())
    path=tmp_path/"suite.json";path.write_text(json.dumps(doc))
    assert SuiteConfig.load(path,ROOT).document["run_id"]=="comprehensive-20261104"
    for key,value in [("output_directory","../escape"),("quiet_gain",0),("runtime_timeout_seconds",float("nan"))]:
        altered=dict(doc);altered[key]=value;path.write_text(json.dumps(altered))
        with pytest.raises(BenchmarkConfigurationError):SuiteConfig.load(path,ROOT)
