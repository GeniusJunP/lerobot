"""
Debug script to load a real pretrained ACT policy (GeniusJunP/act_grab_candy) and run it in the standard
`record_loop` using a synthetic (black image + zero state) virtual robot, so we can inspect produced
actions without real hardware.

Goals:
- Verify the policy loads correctly from Hugging Face Hub.
- Auto-infer action dim, camera names, image shapes, and (optional) state dim from the policy config.
- Feed constant black images (or optionally random noise) so vision path executes deterministically.
- Log the actions the policy outputs each step.

Usage (examples):
    # Default: 1 episode, 2 seconds, 5 FPS, black images
    python debug_record_with_act_policy.py

    # Two short episodes at 10 FPS with random images
    python debug_record_with_act_policy.py --episodes 2 --ep-seconds 1.5 --fps 10 --random-images

    # Force CPU
    python debug_record_with_act_policy.py --device cpu

Notes:
- Requires internet access for first download of the model (cached afterwards).
- Dataset is created under a unique repo_id (not pushed) to reuse existing machinery; you can ignore its contents.
- If normalization stats were not packaged with the model, the script will warn and exit.
"""
from __future__ import annotations

import argparse
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import init_logging, log_say, get_safe_torch_device
from lerobot.record import record_loop
from lerobot.robots.robot import Robot, RobotConfig
from lerobot.configs.types import FeatureType

MODEL_REPO_ID = "GeniusJunP/act_grab_candy"

# ----------------------------- Virtual Robot ----------------------------- #
@dataclass
class VirtualRobotConfig(RobotConfig):
    id: str | None = "virtual_act_robot"


class VirtualRobot(Robot):
    name = "virtual_act_robot"
    config_class = VirtualRobotConfig

    def __init__(self, action_dim: int, camera_specs: dict[str, tuple[int, int, int]], state_dim: int | None):
        super().__init__(self.config_class())
        self._connected = False
        self._t0 = 0.0
        # Build hardware-style features dictionary
        # Observation: joint-like scalars + camera images
        self._obs_joint_names = [f"s{i}" for i in range(state_dim or 0)]
        self._observation_features: dict[str, Any] = {
            **{jn: float for jn in self._obs_joint_names},
            **{cam: (h, w, c) for cam, (h, w, c) in camera_specs.items()},
        }
        # Actions: scalar targets
        self._action_names = [f"a{i}" for i in range(action_dim)]
        self._action_features = {an: float for an in self._action_names}
        self._camera_specs = camera_specs
        self.last_action = {}
        self.random_state = np.random.RandomState(0)

    # --- Required interface --- #
    @property
    def observation_features(self) -> dict:
        return self._observation_features

    @property
    def action_features(self) -> dict:
        return self._action_features

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True):  # noqa: D401 (simple)
        logging.info("VirtualRobot connected (no hardware).")
        self._connected = True
        self._t0 = time.perf_counter()

    @property
    def is_calibrated(self) -> bool:  # Always true for virtual robot
        return True

    def calibrate(self) -> None:  # No-op
        pass

    def configure(self) -> None:  # No-op
        pass

    def disconnect(self):
        logging.info("VirtualRobot disconnected.")
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise RuntimeError("Robot not connected")
        obs = {}
        # Joint-like scalars: simple time-based ramp (normalized ~seconds)
        t = time.perf_counter() - self._t0
        for jn in self._obs_joint_names:
            obs[jn] = float(t)
        # Camera images: created externally via factory each call (injected later)
        for cam, (h, w, c) in self._camera_specs.items():
            obs[cam] = self._image_provider(cam, h, w, c)
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected:
            raise RuntimeError("Robot not connected")
        logging.info(f"VirtualRobot received action: {action}")
        self.last_action = action
        return action

    # Injection point for image generation mode (set externally)
    def set_image_provider(self, fn):
        self._image_provider = fn


# ----------------------------- Helpers ----------------------------- #

def infer_robot_specs_from_policy(policy: PreTrainedPolicy):
    cfg = policy.config
    # Action dim
    action_dim = cfg.output_features["action"].shape[0]
    # Image cameras
    camera_specs: dict[str, tuple[int, int, int]] = {}
    for key, ft in cfg.input_features.items():
        if ft.type is FeatureType.VISUAL and key.startswith("observation.images."):
            cam = key.removeprefix("observation.images.")
            c, h, w = ft.shape  # channel-first in config
            camera_specs[cam] = (h, w, c)  # store as HWC for numpy
    # State feature (optional)
    state_dim = None
    if cfg.robot_state_feature is not None:
        state_dim = cfg.robot_state_feature.shape[0]
    return action_dim, camera_specs, state_dim


def create_policy(repo_id: str, device: str | None):
    logging.info(f"Loading policy from {repo_id} ...")
    policy: ACTPolicy = ACTPolicy.from_pretrained(repo_id)
    if device:
        policy.to(get_safe_torch_device(device))
        policy.config.device = device
    # Sanity: check normalization buffers not inf
    bad = []
    for name, buf in policy.named_buffers():
        if torch.isinf(buf).any():
            bad.append(name)
    if bad:
        logging.error("Some normalization stats are inf (not loaded): %s", bad)
        raise SystemExit(
            "Normalization stats missing. Ensure the pretrained weights include buffers or provide stats manually."
        )
    logging.info("Policy loaded OK.")
    return policy


def build_dataset(robot: VirtualRobot, fps: int, use_video: bool) -> LeRobotDataset:
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video)
    repo_id = f"lerobot/act_debug_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    ds = LeRobotDataset.create(repo_id, fps=fps, robot_type=robot.name, features={**action_features, **obs_features})
    return ds


# ----------------------------- Main Script ----------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--ep-seconds", type=float, default=20.0)
    p.add_argument("--reset-seconds", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--device", type=str, default=None, help="Force device (cpu|cuda)")
    p.add_argument("--random-images", action="store_true", help="Use random noise instead of black images")
    p.add_argument("--model", type=str, default=MODEL_REPO_ID, help="Model repo id or local path")
    return p.parse_args()


def main():
    args = parse_args()
    init_logging()
    log_say("Load ACT policy", play_sounds=False)
    policy = create_policy(args.model, args.device)

    action_dim, camera_specs, state_dim = infer_robot_specs_from_policy(policy)
    logging.info(
        "Inferred specs -> action_dim=%d, cameras=%s, state_dim=%s", action_dim, list(camera_specs), state_dim
    )

    robot = VirtualRobot(action_dim, camera_specs, state_dim)

    # Image provider selection
    if args.random_images:
        def provider(_cam, h, w, c):
            return (np.random.rand(h, w, c) * 255).astype(np.uint8)
    else:
        def provider(_cam, h, w, c):
            return np.zeros((h, w, c), dtype=np.uint8)
    robot.set_image_provider(provider)

    dataset = build_dataset(robot, args.fps, use_video=False)

    robot.connect()

    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < args.episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}", play_sounds=False)
            policy.reset()
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                policy=policy,
                dataset=dataset,
                control_time_s=args.ep_seconds,
            )
            if not events["stop_recording"] and recorded_episodes < args.episodes - 1:
                log_say("Reset environment (no policy)", play_sounds=False)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    policy=None,  # simulate reset without policy
                    dataset=dataset,
                    control_time_s=args.reset_seconds,
                )
            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording", play_sounds=False)
    robot.disconnect()
    logging.info("Done.")


if __name__ == "__main__":
    main()
