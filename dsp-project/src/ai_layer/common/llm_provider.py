import json
import re
from typing import Any, Protocol


class LLMProvider(Protocol):
    def generate_text(self, prompt: str) -> str:
        ...

    def generate_structured_json(self, prompt: str) -> dict[str, Any]:
        ...


class MockLLMProvider:
    """Deterministic LLM mock for tests and offline development."""

    def generate_text(self, prompt: str) -> str:
        instruction = prompt.partition("\n")[0].lower()
        if "summarize" in instruction or "summary" in instruction:
            excerpt = re.search(r"Transcript excerpt:\s*(.+)", prompt)
            return "Offline extractive overview: " + excerpt.group(1) if excerpt else "Offline summary: no transcript excerpt supplied."
        if "answer" in instruction:
            lowered = prompt.lower()
            if "evidence:" in lowered:
                idx = lowered.find("evidence:") + len("evidence:")
                rest = prompt[idx:].split("If the evidence")[0]
                lines = [line.strip().lstrip("- ").strip() for line in rest.splitlines() if line.strip().startswith("-")]
                if lines:
                    return "Based on the conversation: " + "; ".join(lines[:3]).rstrip(".") + "."
            return "The supplied evidence does not contain this information."
        return "Mock LLM response."

    def generate_structured_json(self, prompt: str) -> dict[str, Any]:
        lowered = prompt.partition("\n")[0].lower()

        if "query intent" in lowered or "analyze intent" in lowered:
            query_line = ""
            for line in prompt.splitlines():
                if line.lower().startswith("query:"):
                    query_line = line[6:].strip().lower()
                    break
            target_q = query_line if query_line else lowered

            if "launch" in target_q or "decide" in target_q or "decision" in target_q:
                return {
                    "intent_category": "decision",
                    "target_person": None,
                    "target_topic": "launch date" if "launch" in target_q else None,
                }
            if "task" in target_q or "commit" in target_q or "action" in target_q or "deadline" in target_q or "redesign" in target_q:
                person = None
                for candidate in ["Sarah", "David", "Marcus", "Elena", "Aris", "Rahul", "Aman", "Alex"]:
                    if candidate.lower() in target_q:
                        person = candidate
                        break
                return {
                    "intent_category": "task",
                    "target_person": person,
                    "target_topic": "tasks",
                }
            if "ransac" in target_q or "spatial" in target_q or "mapping" in target_q:
                person = "Rahul" if "rahul" in target_q else None
                return {
                    "intent_category": "general",
                    "target_person": person,
                    "target_topic": "RANSAC",
                }
            return {
                "intent_category": "general",
                "target_person": None,
                "target_topic": None,
            }

        if "identity cue" in lowered or "self_intro" in lowered or "introduction" in lowered:
            cues = []
            for turn_id, speaker, text in self._dialogue_rows(prompt):
                intro = re.search(r"\b(?:I'm|I am)\s+([A-Z][a-z]+)\b", text)
                if intro:
                    cues.append({"cue_type": "self_intro", "detected_name": intro.group(1),
                                 "target_speaker_id": speaker, "source_turn_id": turn_id,
                                 "confidence": 0.9})
            return {"cues": cues}

        if "structured conversation intelligence" in lowered:
            dialogue = "\n".join(f"[{t}] {s}: {text}" for t, s, text in self._dialogue_rows(prompt))
            return {
                "facts": self.generate_structured_json("Extract facts\n" + dialogue).get("facts", []),
                "decisions": self.generate_structured_json("Extract decisions\n" + dialogue).get("decisions", []),
                "action_items": self.generate_structured_json("Extract action items\n" + dialogue).get("action_items", []),
            }

        if "extract facts" in lowered or "factual" in lowered:
            facts = []
            for line in prompt.split("\n"):
                match = re.search(r"\[(turn_\w+)\]\s*([^:]+):\s*(.*)", line)
                if match:
                    t_id, spk, text = match.group(1), match.group(2).strip(), match.group(3).strip()
                    if any(w in text.lower() for w in ["renewed", "tested", "operated", "verified", "calibrated", "threshold", "contract", "is complete", "tests are passing"]):
                        facts.append({
                            "statement": text,
                            "turn_id": t_id,
                        })
            return {"facts": facts}

        if "extract decisions" in lowered:
            decisions = []
            for line in prompt.split("\n"):
                match = re.search(r"\[(turn_\w+)\]\s*([^:]+):\s*(.*)", line)
                if match:
                    t_id, spk, text = match.group(1), match.group(2).strip(), match.group(3).strip()
                    if any(w in text.lower() for w in ["decid", "agreed", "confirmed", "scheduled"]):
                        decisions.append({
                            "decision": text,
                            "agreed_by": [spk],
                            "turn_id": t_id,
                        })
            return {"decisions": decisions}

        if "extract action items" in lowered:
            items = []
            for line in prompt.split("\n"):
                match = re.search(r"\[(turn_\w+)\]\s*([^:]+):\s*(.*)", line)
                if match:
                    t_id, spk, text = match.group(1), match.group(2).strip(), match.group(3).strip()
                    task_match = re.search(r"(?:I\s+will|will)\s+([^.]+?)(?:\s+by\s+([^.]+))?(?:\.|$)", text, re.IGNORECASE)
                    if task_match:
                        task_desc = task_match.group(1).strip()
                        deadline = task_match.group(2).strip() if task_match.group(2) else None
                        assignee = spk
                        if text.lower().startswith("david will"):
                            assignee = "David"
                        items.append({
                            "task_description": task_desc[:1].upper() + task_desc[1:],
                            "assignee_name": assignee,
                            "deadline": deadline,
                            "turn_id": t_id,
                        })
            return {"action_items": items}

        if "normalize" in lowered:
            match = re.search(r"statement:\s*(.+)", prompt, re.IGNORECASE)
            statement = match.group(1).strip() if match else prompt
            return {"normalized_statement": statement}

        if "relationship" in lowered or "classify" in lowered:
            return {"relationship_type": "novel", "rationale": "No prior matching record found."}

        json_match = re.search(r"\{.*\}", prompt, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    @staticmethod
    def _dialogue_rows(prompt: str) -> list[tuple[str, str, str]]:
        """Extract only supplied transcript lines; never invent fixture evidence."""
        return [(m.group(1), m.group(2).strip(), m.group(3).strip())
                for line in prompt.splitlines()
                if (m := re.match(r"\[(turn_\w+)\]\s*([^:]+):\s*(.*)", line))]
