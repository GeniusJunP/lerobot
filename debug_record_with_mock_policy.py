"""
Debug script to verify the action generation and propagation in `record.py`.

This script uses:
1.  `LoggingDummyRobot`: A virtual robot that logs any action it's asked to perform.
2.  `MockPolicy`: A virtual policy that generates predictable, non-zero actions.

Purpose:
- To confirm that the `record` function correctly uses a policy to generate
  non-zero actions during an episode.
- To verify that these actions are passed to the robot's `send_action` method.
- To isolate whether the "arm not moving" issue is in the software logic
  (action generation) or hardware/driver layer.

How to run:
conda activate lerobot-dev
# Run for 2 episodes, generating a ramp-up action pattern
python debug_record_with_mock_policy.py --episodes 2 --ep-seconds 1 --fps 5 --pattern ramp

# Run and observe the reset phase without the policy (will show log spam if unpatched)
python debug_record_with_mock_policy.py --episodes 2 --ep-seconds 1 --fps 5 --pattern ramp --no-reset-use-policy


Expected output:
- During "Recording episode...", you should see logs like:
  "INFO ... DummyRobot received action: {'target': 0.1}"
  "INFO ... DummyRobot received action: {'target': 0.2}"
- If the action is always `{'target': 0.0}` or the log never appears,
  it points to a software issue.
- If non-zero actions are logged, the software side is likely working correctly.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.record import record_loop
from lerobot.robots.robot import Robot, RobotConfig
from lerobot.utils.utils import init_logging, log_say, get_safe_torch_device

# ################## MOCK/DUMMY COMPONENTS ##################

@dataclass
class MockPolicyConfig:
    device: str = "cpu"
    use_amp: bool = False
    pattern: str = "zero"  # zero|ramp|sine

class MockPolicy:
    name = "mock_policy"  # for logging if needed
    def __init__(self, config: MockPolicyConfig):
        self.config = config
        self.device = get_safe_torch_device(config.device)
        self._step = 0
    def reset(self):
        self._step = 0
    @torch.inference_mode()
    def select_action(self, observation: dict[str, torch.Tensor]):  # returns (batch, action_dim)
        pat = self.config.pattern
        s = self._step
        if pat == "ramp":
            val = min(s * 0.1, 1.0)
        elif pat == "sine":
            val = float(torch.sin(torch.tensor(s * 0.5)) * 0.5 + 0.5)
        elif pat == "zero":
            val = 0.0
        else:
            val = 0.0
        self._step += 1
        # Return shape (1,1) so predict_action() squeeze(0) -> (1,)
        return torch.tensor([[val]], dtype=torch.float32, device=self.device)

@dataclass
class DummyRobotConfig(RobotConfig):
    """A minimal config for our DummyRobot."""
    id: str | None = "dummy_robot_0"


class LoggingDummyRobot(Robot):
    """A robot that logs actions sent to it instead of moving hardware."""
    name = "logging_dummy_robot"
    config_class = DummyRobotConfig

    def __init__(self, config: DummyRobotConfig = None):
        if config is None:
            config = self.config_class()
        super().__init__(config)
        self._connected = False
        self._t0 = 0.0
        self.last_action = {}

    @property
    def observation_features(self) -> dict:
        return {"image": (480, 640, 3), "state": 1}

    @property
    def action_features(self) -> dict:
        return {"target": float}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True):
        logging.info(f"{self.name} connected.")
        self._connected = True
        self._t0 = time.perf_counter()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self):
        logging.info(f"{self.name} disconnected.")
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise RuntimeError("Robot not connected")
        return {
            "image": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            "state": np.array([time.perf_counter() - self._t0], dtype=np.float32),
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected:
            raise RuntimeError("Robot not connected")
        logging.info(f"DummyRobot received action: {action}")
        self.last_action = action
        return action


# ################## MAIN SCRIPT LOGIC ##################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    p.add_argument("--ep-seconds", type=float, default=20.0, help="Duration of each episode")
    p.add_argument("--reset-seconds", type=float, default=1.0, help="Duration of reset phase")
    p.add_argument("--fps", type=int, default=10, help="Target FPS for the record loop")
    p.add_argument("--pattern", type=str, default="ramp", choices=["zero", "ramp", "sine"], help="Action pattern for mock policy")
    p.add_argument("--reset-use-policy", action="store_true", help="Pass the policy to the reset loop")
    p.add_argument("--no-reset-use-policy", dest="reset_use_policy", action="store_false")
    p.set_defaults(reset_use_policy=True)
    return p.parse_args()


def main():
    args = parse_args()
    init_logging()
    robot = LoggingDummyRobot()
    policy = MockPolicy(MockPolicyConfig(pattern=args.pattern))
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=False)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=False)
    dataset = LeRobotDataset.create(
        "lerobot/dummy_dataset", fps=args.fps, robot_type=robot.name, features={**action_features, **obs_features}
    )
    robot.connect()
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    recorded_episodes = 0
    with VideoEncodingManager(dataset):
        while recorded_episodes < args.episodes and not events["stop_recording"]:
            log_say(f"Recording episode {dataset.num_episodes}", play_sounds=False)
            policy.reset()
            record_loop(robot=robot, events=events, fps=args.fps, policy=policy, dataset=dataset, control_time_s=args.ep_seconds)
            if not events["stop_recording"] and ((recorded_episodes < args.episodes - 1) or events["rerecord_episode"]):
                log_say("Reset the environment", play_sounds=False)
                policy_for_reset = policy if args.reset_use_policy else None
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    policy=policy_for_reset,
                    dataset=dataset,
                    control_time_s=args.reset_seconds,
                )
            if events["rerecord_episode"]:
                log_say("Re-record episode", play_sounds=False)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            dataset.save_episode()
            recorded_episodes += 1
    log_say("Stop recording", play_sounds=False)
    robot.disconnect()
    logging.info("Finished debug run.")


if __name__ == "__main__":
    main()
