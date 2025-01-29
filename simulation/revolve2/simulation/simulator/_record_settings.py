from dataclasses import dataclass


@dataclass
class RecordSettings:
    """Settings for recording a simulation."""

    video_directory: str
    overwrite: bool = False

    fps: int = 24

    width: int | None = None
    height: int | None = None

    camera_id: int | None = None
    camera_type: str | None = None
