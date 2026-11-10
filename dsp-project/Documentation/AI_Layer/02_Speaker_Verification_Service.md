# Sublayer 02: Speaker Identity Verification

## 1. Role & Purpose

The **Speaker Identity Verification Sublayer** confirms or corrects the speaker identifiers assigned to dialogue segments. Voice characteristics alone can sometimes be uncertain or misclassified due to background noise or similar voices. This sublayer uses the semantic context of the conversation (what people are saying to each other) to verify identities and link unnamed speaker clusters to real contact profiles.

---

## 2. Inputs

- **Chronological Transcript**: The sequence of timestamped utterances produced by the STT sublayer.
- **Preliminary Speaker IDs**: The speaker label currently assigned to each utterance.
- **Registered Speakers List**: Stored profiles of known speakers, including names and previous interaction history.

---

## 3. Step-by-Step Working

1. **Dialogue Context Scanning**:
   - The sublayer reads through conversational turns, examining how participants refer to themselves and each other.
   - It identifies key identity markers in the text, such as:
     - **Self-Introductions**: When a speaker explicitly states their name (e.g., *"Hello everyone, this is Alex"*).
     - **Direct Addresses**: When one person speaks directly to another (e.g., *"Raj, what are your thoughts?"*) and the next speaker responds naturally.
     - **Third-Party References**: When a speaker mentions another person in the third person, establishing that the speaker cannot be that person.

2. **Cross-Referencing with Assigned Speaker IDs**:
   - The detected identity markers are compared against the preliminary speaker IDs assigned to those utterances:
     - **Agreement**: If the spoken words match the assigned speaker ID, the sublayer increases the confidence score for that speaker assignment.
     - **Correction**: If a speaker was assigned an anonymous label (e.g., `Speaker 2`) but introduces themselves as a known contact, the sublayer reassigns all utterances spoken by that voice in this session to the confirmed contact profile.
     - **Conflict Resolution**: If the preliminary speaker label contradicts the spoken content, the sublayer re-evaluates the segment, corrects the speaker ID, and records the explanation.

3. **Updating Speaker Profile Confidence**:
   - Every confirmed identity increases the system's long-term certainty for that speaker.
   - Newly identified speakers who do not yet exist in the contacts list are flagged so a new speaker profile can be created.

---

## 4. Outputs

- **Verified Transcript**: The complete conversation transcript updated with validated speaker names and IDs.
- **Identity Evidence Record**: Explanations for any speaker reassignments or confirmations attached to the conversation records.
- **Speaker Directory Updates**: Refined confidence scores and profile associations for registered speakers.
