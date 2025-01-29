from dataclasses import dataclass, field
from typing import Tuple, Optional, Any

from ._multi_body_system import MultiBodySystem
from ._simulation_handler import SimulationHandler


@dataclass(kw_only=True)
class Scene:
    """Description of a scene that can be simulated."""

    handler: SimulationHandler
    _multi_body_systems: list[MultiBodySystem] = field(default_factory=list, init=False)
    """
    Multi-body system in this scene.

    Don't add to this directly, but use `add_multi_body_system` instead.
    """

    _mujoco_specifics: list[Tuple[Optional[str], str, dict[str, Any]]] = field(default_factory=list, init=False)
    """
    Mujoco-specific configuration. Ugly hack until a better solution is implemented
    """

    def add_multi_body_system(self, multi_body_system: MultiBodySystem) -> None:
        """
        Add a multi-body system to the scene.

        :param multi_body_system: The multi-body system to add.
        """
        self._multi_body_systems.append(multi_body_system)

    @property
    def multi_body_systems(self) -> list[MultiBodySystem]:
        """
        Get the multi-body systems in scene.

        Do not make changes to this list.

        :returns: The multi-body systems in the scene.
        """
        return self._multi_body_systems[:]

    def add_mujoco_element(self, parent: Optional[str], tag: str, kwargs):
        """
        Add a mujoco-specific element to the scene
        :param parent: The parent to attach the element to or None for the worldbody
        :param tag: The tag of the element
        :param kwargs: The arguments to create the element with
        """
        self._mujoco_specifics.append((parent, tag, kwargs))

    @property
    def mujoco_specifics(self) -> list[Tuple[Optional[str], str, dict[str, Any]]]:
        return self._mujoco_specifics[:]
