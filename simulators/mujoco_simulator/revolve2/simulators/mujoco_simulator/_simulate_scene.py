import logging
import math
from typing import List, Dict, Callable

import cv2
import mujoco
import numpy as np
import numpy.typing as npt

from revolve2.simulation.scene import Scene, SimulationState
from revolve2.simulation.simulator import RecordSettings
from revolve2.simulation.simulator._simulator import Callback
from ._abstraction_to_mujoco_mapping import CameraSensorMujoco
from ._control_interface_impl import ControlInterfaceImpl
from ._open_gl_vision import OpenGLVision
from ._render_backend import RenderBackend
from ._scene_to_model import scene_to_model
from ._simulation_state_impl import SimulationStateImpl
from .viewers import CustomMujocoViewer, NativeMujocoViewer, ViewerType


def simulate_scene(
    scene_id: int,
    scene: Scene,
    callbacks: Dict[Callback, List[Callable]],
    headless: bool,
    record_settings: RecordSettings | None,
    start_paused: bool,
    control_step: float,
    sample_step: float | None,
    simulation_time: int | None,
    simulation_timestep: float,
    cast_shadows: bool,
    fast_sim: bool,
    viewer_type: ViewerType,
    render_backend: RenderBackend = RenderBackend.EGL,
) -> list[SimulationState]:
    """
    Simulate a scene.

    :param scene_id: An id for this scene, unique between all scenes ran in parallel.
    :param scene: The scene to simulate.
    :param headless: If False, a viewer will be opened that allows a user to manually view and manually interact with the simulation.
    :param record_settings: If not None, recording will be done according to these settings.
    :param start_paused: If true, the simulation will start in a paused state. Only makessense when headless is False.
    :param control_step: The time between each call to the handle function of the scene handler. In seconds.
    :param sample_step: The time between each state sample of the simulation. In seconds.
    :param simulation_time: How long to simulate for. In seconds.
    :param simulation_timestep: The duration to integrate over during each step of the simulation. In seconds.
    :param cast_shadows: If shadows are cast.
    :param fast_sim: If fancy rendering is disabled.
    :param viewer_type: The type of viewer used for the rendering in a window.
    :param render_backend: The backend to be used for rendering (EGL by default and switches to GLFW if no cameras are on the robot).
    :returns: The results of simulation. The number of returned states depends on `sample_step`.
    :raises ValueError: If the viewer is not able to record.
    """
    logging.info(f"Simulating scene {scene_id}")

    """Define mujoco data and model objects for simulating."""
    model, mapping = scene_to_model(
        scene, simulation_timestep, cast_shadows=cast_shadows, fast_sim=fast_sim
    )
    data = mujoco.MjData(model)

    offscreen_render = (headless and record_settings is not None)
    """Initialize viewer object if we need to render the scene."""
    if not headless:
        render_backend = RenderBackend.GLFW
        import glfw
        glfw.init()

        match viewer_type:
            case viewer_type.CUSTOM:
                viewer = CustomMujocoViewer
            case viewer_type.NATIVE:
                viewer = NativeMujocoViewer
            case _:
                raise ValueError(
                    f"Viewer of type {viewer_type} not defined in _simulate_scene."
                )

        viewer = viewer(
            model,
            data,
            width=None if record_settings is None else record_settings.width,
            height=None if record_settings is None else record_settings.height,
            backend=render_backend,
            start_paused=start_paused,
            render_every_frame=False,
            hide_menus=(record_settings is not None),
        )

        cam = None
        if viewer_type == viewer_type.CUSTOM:
            cam = viewer._viewer_backend.cam
        elif viewer_type == viewer_type.NATIVE:
            cam = viewer._viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = 0

    """Define a control interface for the mujoco simulation (used to control robots)."""
    control_interface = ControlInterfaceImpl(
        data=data, abstraction_to_mujoco_mapping=mapping
    )
    """Make separate viewer for camera sensors."""
    camera_viewers = {
        camera.camera_id: OpenGLVision(
            model=model, camera=camera, headless=headless, open_gl_lib=render_backend
        )
        for camera in mapping.camera_sensor.values()
    }

    if offscreen_render:
        camera_viewers[-1] = OpenGLVision(
            model=model, camera=CameraSensorMujoco(
                1 if (cid := record_settings.camera_id) is None else cid,
                (record_settings.width, record_settings.height)),
            headless=headless, open_gl_lib=render_backend,
        )

        if record_settings.camera_type is not None:
            camera_viewers[-1]._camera.type = dict(
                free=mujoco.mjtCamera.mjCAMERA_FREE,
                tracking=mujoco.mjtCamera.mjCAMERA_TRACKING,
                fixed=mujoco.mjtCamera.mjCAMERA_FIXED,
                user=mujoco.mjtCamera.mjCAMERA_USER,
            )[record_settings.camera_type.lower()]
        else:
            camera_viewers[-1]._camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    """Define some additional control variables."""
    last_control_time = 0.0
    last_sample_time = 0.0
    last_video_time = 0.0  # time at which last video frame was saved

    simulation_states: list[SimulationState] = (
        []
    )  # The measured states of the simulation

    """Record the scene if we want to record."""
    if record_settings is not None:
        if not offscreen_render and not viewer.can_record:
            raise ValueError(
                f"Selected Viewer {type(viewer).__name__} has no functionality to record."
            )
        video_step = 1 / record_settings.fps
        video_file_path = f"{record_settings.video_directory}/{scene_id}.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_size = ((record_settings.width, record_settings.height)
                      if offscreen_render else
                      viewer.current_viewport_size())
        video = cv2.VideoWriter(
            video_file_path,
            fourcc,
            record_settings.fps,
            video_size,
        )

    # callbacks = {k: [] for k in Callback}

    for cb in callbacks[Callback.START]:
        cb(model=model, data=data, mapping=mapping, handler=scene.handler)

    if not headless:
        for cb in callbacks[Callback.RENDER_START]:
            cb(model=model, data=data, viewer=viewer)

    """
    Compute forward dynamics without actually stepping forward in time.
    This updates the data so we can read out the initial state.
    """
    mujoco.mj_forward(model, data)
    images = {
        camera_id: camera_viewer.process(model, data)
        for camera_id, camera_viewer in camera_viewers.items()
    }

    def sample(log=False):
        _state = SimulationStateImpl(
            data=data, abstraction_to_mujoco_mapping=mapping, camera_views=images
        )
        if log:
            simulation_states.append(_state)
        return _state

    def control():
        for _cb in callbacks[Callback.PRE_CONTROL]:
            _cb(model, data)

        scene.handler.handle(sample(), control_interface, control_step)

        for _cb in callbacks[Callback.POST_CONTROL]:
            _cb(model, data)

    # Sample initial state.
    if sample_step is not None:
        sample(log=True)

    if control_step is not None:
        control()

    """After rendering the initial state, we enter the rendering loop."""
    while (time := data.time) <= (
        float("inf") if simulation_time is None else simulation_time
    ):
        for cb in callbacks[Callback.PRE_STEP]:
            cb(model, data)

        # do control if it is time
        if time >= last_control_time + control_step:
            last_control_time = math.floor(time / control_step) * control_step

            control()

        # sample state if it is time
        if sample_step is not None:
            if time >= last_sample_time + sample_step:
                last_sample_time = int(time / sample_step) * sample_step
                sample(log=True)

        # step simulation
        mujoco.mj_step(model, data)
        # extract images from camera sensors.
        images = {
            camera_id: camera_viewer.process(model, data)
            for camera_id, camera_viewer in camera_viewers.items()
        }

        # render if not headless. also render when recording and if it time for a new video frame.
        if not headless or (
            record_settings is not None and time >= last_video_time + video_step
        ) and not offscreen_render:
            _status = viewer.render(callbacks)

            # Check if simulation was closed
            if _status == -1:
                break

        # capture video frame if it's time
        if record_settings is not None and time >= last_video_time + video_step:
            last_video_time = int(time / video_step) * video_step

            if offscreen_render:
                img = images[-1]

            else:
                # https://github.com/deepmind/mujoco/issues/285 (see also record.cc)
                img: npt.NDArray[np.uint8] = np.empty(
                    (video_size[1], video_size[0], 3),
                    dtype=np.uint8,
                )

                mujoco.mjr_readPixels(
                    rgb=img,
                    depth=None,
                    viewport=viewer.view_port,
                    con=viewer.context,
                )

            # Flip the image and map to OpenCV colormap (BGR -> RGB)
            img = np.flipud(img)[:, :, ::-1]
            video.write(img)

        for cb in callbacks[Callback.POST_STEP]:
            cb(model, data)

    for cb in callbacks[Callback.END]:
        cb(model, data)

    if not headless:
        for cb in callbacks[Callback.RENDER_END]:
            cb(model=model, data=data, viewer=viewer)

    """Once simulation is done we close the potential viewer and release the potential video."""
    if (not headless or record_settings is not None) and not offscreen_render:
        viewer.close_viewer()

    if record_settings is not None:
        video.release()

    # Sample one final time.
    if sample_step is not None:
        sample(log=True)

    for camera_viewer in camera_viewers.values():
        camera_viewer.free()

    logging.info(f"Scene {scene_id} done.")
    return simulation_states
