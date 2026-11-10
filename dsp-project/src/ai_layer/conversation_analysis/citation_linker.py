from ai_layer.models.analysis_models import UtteranceCitation
from ai_layer.models.speaker_models import VerifiedTurn


class CitationLinker:
    @staticmethod
    def find_matching_turn(turns: list[VerifiedTurn], turn_id: str) -> VerifiedTurn | None:
        if not turns:
            return None
        target = str(turn_id).strip().lower()
        for turn in turns:
            if turn.turn_id.lower() == target:
                return turn
        # Try matching by number e.g. "turn_2" vs "2"
        digits = "".join(ch for ch in target if ch.isdigit())
        if digits:
            for turn in turns:
                turn_digits = "".join(ch for ch in turn.turn_id if ch.isdigit())
                if turn_digits == digits:
                    return turn
        return turns[0] if turns else None

    @staticmethod
    def link_citation(turn: VerifiedTurn) -> UtteranceCitation:
        return UtteranceCitation(
            turn_id=turn.turn_id,
            speaker_name=turn.speaker_name,
            time_range=turn.time_range,
            spoken_text=turn.text,
        )
