from dataclasses import dataclass

from revolve2.simulation.scene.geometry import Geometry


@dataclass
class Terrain:
    """Terrain consisting of only static geometry."""

    static_geometry: list[Geometry]
    """The static geometry that defines the terrain."""
