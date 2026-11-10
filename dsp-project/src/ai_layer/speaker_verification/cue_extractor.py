from ai_layer.common.llm_provider import LLMProvider
from ai_layer.models.speaker_models import IdentityCue
from ai_layer.models.transcription_models import StitchedTurn


class IdentityCueExtractor:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def extract_cues_from_turns(self, turns: list[StitchedTurn]) -> list[IdentityCue]:
        if not turns:
            return []

        dialogue = "\n".join(
            f"[{turn.turn_id}] {turn.speaker_id}: {turn.text}" for turn in turns
        )
        prompt = (
            "Extract identity cues (self_intro, direct_address, third_party) from this dialogue.\n"
            f"{dialogue}\n"
            'Return JSON: {"cues": [{"cue_type": "...", "detected_name": "...", '
            '"target_speaker_id": "...", "source_turn_id": "...", "confidence": 0.0}]}'
        )
        payload = self._llm_provider.generate_structured_json(prompt)
        cues: list[IdentityCue] = []

        for item in payload.get("cues", []):
            cues.append(
                IdentityCue(
                    cue_type=item["cue_type"],
                    detected_name=item["detected_name"],
                    target_speaker_id=item["target_speaker_id"],
                    source_turn_id=item["source_turn_id"],
                    confidence=float(item.get("confidence", 0.5)),
                )
            )

        if not cues:
            cues.extend(self._heuristic_extract(turns))

        return cues

    @staticmethod
    def _heuristic_extract(turns: list[StitchedTurn]) -> list[IdentityCue]:
        cues: list[IdentityCue] = []
        for turn in turns:
            lowered = turn.text.lower()
            if "i'm " in lowered or "i am " in lowered:
                name = turn.text.split("I'm")[-1].split("I am")[-1].strip(" .")
                if name:
                    cues.append(
                        IdentityCue(
                            cue_type="self_intro",
                            detected_name=name.split()[0],
                            target_speaker_id=turn.speaker_id,
                            source_turn_id=turn.turn_id,
                            confidence=0.85,
                        )
                    )
        return cues
