"""Optional camera / body-scale calibration for distance estimates."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import CalibrationSettings


CALIBRATION_DISCLAIMER = (
    "Accuracy depends on camera position, lighting, perspective, frame rate, "
    "and calibration. Uncalibrated webcams cannot report exact meters/second "
    "or striking impact force."
)


@dataclass
class CalibrationResult:
    settings: CalibrationSettings
    usable: bool
    message: str


def build_calibration(
    athlete_height_cm: float | None,
    reference_distance_m: float | None,
    camera_distance_m: float | None,
    pixels_per_meter: float | None,
) -> CalibrationResult:
    settings = CalibrationSettings(
        athlete_height_cm=athlete_height_cm if athlete_height_cm and athlete_height_cm > 50 else None,
        reference_distance_m=reference_distance_m if reference_distance_m and reference_distance_m > 0 else None,
        camera_distance_m=camera_distance_m if camera_distance_m and camera_distance_m > 0 else None,
        pixels_per_meter=pixels_per_meter if pixels_per_meter and pixels_per_meter > 1 else None,
    )
    usable = bool(settings.athlete_height_cm or settings.pixels_per_meter)
    if usable:
        message = "Calibration is active for estimated physical speed. " + CALIBRATION_DISCLAIMER
    else:
        message = (
            "No scale set. Speed is shown as normalized visual movement, not m/s. "
            + CALIBRATION_DISCLAIMER
        )
    return CalibrationResult(settings=settings, usable=usable, message=message)
