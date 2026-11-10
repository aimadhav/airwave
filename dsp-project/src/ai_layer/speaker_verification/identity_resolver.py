from ai_layer.models.speaker_models import IdentityCue, SpeakerProfile, SpeakerReassignment


class IdentityResolver:
    def resolve_identities(
        self,
        cues: list[IdentityCue],
        known_profiles: list[SpeakerProfile],
        preliminary_speaker_ids: list[str],
    ) -> tuple[dict[str, str], dict[str, str], list[SpeakerReassignment]]:
        profile_by_name = {
            profile.canonical_name.lower(): profile for profile in known_profiles
        }
        for profile in known_profiles:
            for alias in profile.aliases:
                profile_by_name[alias.lower()] = profile

        speaker_id_to_name: dict[str, str] = {}
        speaker_id_to_resolved_id: dict[str, str] = {}
        reassignments: list[SpeakerReassignment] = []

        for speaker_id in preliminary_speaker_ids:
            speaker_id_to_name[speaker_id] = speaker_id
            speaker_id_to_resolved_id[speaker_id] = speaker_id

        for cue in cues:
            if cue.cue_type != "self_intro":
                continue

            detected = cue.detected_name.strip()
            profile = profile_by_name.get(detected.lower())
            resolved_id = profile.speaker_id if profile else f"speaker_{detected.lower()}"
            resolved_name = profile.canonical_name if profile else detected

            preliminary_id = cue.target_speaker_id
            previous_name = speaker_id_to_name.get(preliminary_id, preliminary_id)
            if previous_name != resolved_name:
                reassignments.append(
                    SpeakerReassignment(
                        preliminary_speaker_id=preliminary_id,
                        resolved_speaker_id=resolved_id,
                        resolved_name=resolved_name,
                        rationale=f"Detected self-introduction as {resolved_name}.",
                    )
                )

            speaker_id_to_name[preliminary_id] = resolved_name
            speaker_id_to_resolved_id[preliminary_id] = resolved_id

        return speaker_id_to_name, speaker_id_to_resolved_id, reassignments
