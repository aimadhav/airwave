import pytest
from dsp.evaluation import TimedSpeaker, diarization_error

pytest.importorskip("scipy")


def test_label_permutation_and_overlap() -> None:
    reference = [TimedSpeaker(0, 2, "alice"), TimedSpeaker(1, 3, "bob")]
    result = diarization_error(reference, [TimedSpeaker(0, 2, "z"), TimedSpeaker(1, 3, "y")])
    assert result["der"] == 0
    assert result["reference_speaker_seconds"] == 4


def test_miss_false_alarm_and_confusion() -> None:
    reference = [TimedSpeaker(0, 1, "a"), TimedSpeaker(1, 2, "b")]
    result = diarization_error(reference, [TimedSpeaker(0, 3, "x")])
    assert result["confusion_seconds"] == 1
    assert result["false_alarm_seconds"] == 1
    assert result["der"] == 1
    missed = diarization_error(reference, [])
    assert missed["miss_seconds"] == 2
    assert missed["der"] == 1


def test_boundary_collar_is_explicit() -> None:
    result = diarization_error([TimedSpeaker(0, 2, "a")], [TimedSpeaker(0.1, 1.9, "x")], collar=0.25)
    assert result["der"] == 0
    assert result["reference_speaker_seconds"] == 1.5
    assert diarization_error([], [TimedSpeaker(0, 1, "x")])["der"] is None
