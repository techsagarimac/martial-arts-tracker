"""Debounced real-time coaching cues so the UI does not spam every frame."""

from __future__ import annotations

from models.technique import FeedbackItem, FormScores, StanceResult, TechniqueDetection


class FeedbackManager:
    def __init__(self, cooldown_seconds: float = 2.8) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last: dict[str, float] = {}
        self.active: list[FeedbackItem] = []

    def reset(self) -> None:
        self._last.clear()
        self.active = []

    def update(
        self,
        timestamp: float,
        pose_ok: bool,
        pose_message: str,
        stance: StanceResult,
        detection: TechniqueDetection | None,
        form: FormScores,
    ) -> list[FeedbackItem]:
        emitted: list[FeedbackItem] = []

        def maybe(key: str, message: str, level: str, extra_cd: float = 0.0) -> None:
            last = self._last.get(key, -1e9)
            if timestamp - last < self.cooldown_seconds + extra_cd:
                return
            self._last[key] = timestamp
            item = FeedbackItem(message=message, level=level, key=key, timestamp=timestamp)  # type: ignore[arg-type]
            emitted.append(item)

        if not pose_ok:
            maybe("pose", pose_message or "Technique not detected clearly", "red", extra_cd=0.6)
        else:
            for note in stance.feedback:
                if "too close" in note.lower():
                    maybe("feet", note, "yellow")
                elif "guard" in note.lower():
                    maybe("guard", "Keep your guard higher", "yellow")
                elif "inward" in note.lower():
                    maybe("knee", note, "yellow")
                elif "uneven" in note.lower():
                    maybe("align", note, "yellow")
                elif "shifted" in note.lower():
                    maybe("base", "Stay over your base", "yellow")

            if detection and detection.uncertain:
                maybe("uncertain", "Technique uncertain", "red")
            elif detection and detection.name:
                if form.extension >= 84:
                    maybe("ext_ok", "Good extension", "green", extra_cd=1.5)
                elif form.extension < 70 and detection.name != "hook":
                    maybe("ext_low", "Extend through the technique", "yellow")
                if detection.name == "roundhouse_kick" and form.accuracy < 80:
                    maybe("hip", "Rotate your hip more", "yellow")
                if form.balance < 70:
                    maybe("bal", "Re-center after the technique", "yellow")

        # Keep a short live list (unique keys).
        by_key = {item.key: item for item in self.active if timestamp - item.timestamp < 4.0}
        for item in emitted:
            by_key[item.key] = item
        self.active = list(by_key.values())
        return emitted
