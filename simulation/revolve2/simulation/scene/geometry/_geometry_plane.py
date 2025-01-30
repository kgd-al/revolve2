from dataclasses import dataclass, field

from .._color import Color
from ..vector2 import Vector2
from ._geometry import Geometry
from .textures import MapType, Texture


def _default_texture():
    return Texture(
        base_color=Color(100, 100, 100, 255),
        map_type=MapType.MAP2D,
    )


@dataclass(kw_only=True)
class GeometryPlane(Geometry):
    """A flat plane geometry."""

    size: Vector2
    texture: Texture = field(default_factory=_default_texture)

    def __post_init__(self):
        if self.texture is None:
            self.texture = _default_texture()
