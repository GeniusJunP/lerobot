#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import platform
from functools import wraps

import pytest


# Platform detection utilities
def is_apple_silicon():
    """Check if running on Apple Silicon (M1, M2, etc.)."""
    return platform.system() == "Darwin" and platform.machine().lower() in ["arm64", "aarch64"]


def is_intel_mac():
    """Check if running on Intel-based Mac."""
    return platform.system() == "Darwin" and platform.machine().lower() in ["x86_64", "amd64"]


def is_macos():
    """Check if running on macOS (any architecture)."""
    return platform.system() == "Darwin"


def is_linux():
    """Check if running on Linux."""
    return platform.system() == "Linux"


def get_normalized_machine():
    """Get normalized machine architecture string."""
    machine = platform.machine().lower()
    # Normalize architecture names
    if machine in ["amd64"]:
        return "x86_64"
    elif machine in ["aarch64"]:
        return "arm64"
    return machine


# Import torch and other dependencies after utility functions
import torch

from lerobot import available_cameras, available_motors, available_robots
from lerobot.utils.import_utils import is_package_available

DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"

TEST_ROBOT_TYPES = []
for robot_type in available_robots:
    TEST_ROBOT_TYPES += [(robot_type, True), (robot_type, False)]

TEST_CAMERA_TYPES = []
for camera_type in available_cameras:
    TEST_CAMERA_TYPES += [(camera_type, True), (camera_type, False)]

TEST_MOTOR_TYPES = []
for motor_type in available_motors:
    TEST_MOTOR_TYPES += [(motor_type, True), (motor_type, False)]

# Camera indices used for connecting physical cameras
OPENCV_CAMERA_INDEX = int(os.environ.get("LEROBOT_TEST_OPENCV_CAMERA_INDEX", 0))
INTELREALSENSE_SERIAL_NUMBER = int(os.environ.get("LEROBOT_TEST_INTELREALSENSE_SERIAL_NUMBER", 128422271614))

DYNAMIXEL_PORT = os.environ.get("LEROBOT_TEST_DYNAMIXEL_PORT", "/dev/tty.usbmodem575E0032081")
DYNAMIXEL_MOTORS = {
    "shoulder_pan": [1, "xl430-w250"],
    "shoulder_lift": [2, "xl430-w250"],
    "elbow_flex": [3, "xl330-m288"],
    "wrist_flex": [4, "xl330-m288"],
    "wrist_roll": [5, "xl330-m288"],
    "gripper": [6, "xl330-m288"],
}

FEETECH_PORT = os.environ.get("LEROBOT_TEST_FEETECH_PORT", "/dev/tty.usbmodem585A0080971")
FEETECH_MOTORS = {
    "shoulder_pan": [1, "sts3215"],
    "shoulder_lift": [2, "sts3215"],
    "elbow_flex": [3, "sts3215"],
    "wrist_flex": [4, "sts3215"],
    "wrist_roll": [5, "sts3215"],
    "gripper": [6, "sts3215"],
}


def require_x86_64_kernel(func):
    """
    Decorator that skips the test if platform device is not an x86_64 cpu.
    
    Note: This decorator is primarily used for backward compatibility tests
    where test artifacts were generated on x86_64 systems and may have
    floating point precision differences on other architectures.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if get_normalized_machine() != "x86_64":
            pytest.skip("requires x86_64 platform (test artifacts generated on x86_64)")
        return func(*args, **kwargs)

    return wrapper


def require_compatible_kernel(func):
    """
    Decorator that skips the test if platform is not compatible with the functionality.
    
    This allows tests to run on both x86_64 and arm64 (Apple Silicon) architectures,
    which are the primary supported platforms for lerobot.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        machine = get_normalized_machine()
        if machine not in ["x86_64", "arm64"]:
            pytest.skip(f"unsupported platform: {machine}")
        return func(*args, **kwargs)

    return wrapper


def require_arm64_kernel(func):
    """
    Decorator that skips the test if platform device is not an arm64/aarch64 cpu.
    
    Useful for testing Apple Silicon specific functionality.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if get_normalized_machine() != "arm64":
            pytest.skip("requires arm64/aarch64 platform (Apple Silicon)")
        return func(*args, **kwargs)

    return wrapper


def require_apple_silicon(func):
    """
    Decorator that skips the test if not running on Apple Silicon.
    
    This is specifically for Apple Silicon Mac functionality.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_apple_silicon():
            pytest.skip("requires Apple Silicon Mac")
        return func(*args, **kwargs)

    return wrapper


def require_macos(func):
    """
    Decorator that skips the test if not running on macOS.
    
    This works for both Intel and Apple Silicon Macs.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_macos():
            pytest.skip("requires macOS")
        return func(*args, **kwargs)

    return wrapper


def require_cpu(func):
    """
    Decorator that skips the test if device is not cpu.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEVICE != "cpu":
            pytest.skip("requires cpu")
        return func(*args, **kwargs)

    return wrapper


def require_cuda(func):
    """
    Decorator that skips the test if cuda is not available.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("requires cuda")
        return func(*args, **kwargs)

    return wrapper


def require_env(func):
    """
    Decorator that skips the test if the required environment package is not installed.
    As it need 'env_name' in args, it also checks whether it is provided as an argument.
    If 'env_name' is None, this check is skipped.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if 'env_name' is provided and extract its value
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        if "env_name" in arg_names:
            # Get the index of 'env_name' and retrieve the value from args
            index = arg_names.index("env_name")
            env_name = args[index] if len(args) > index else kwargs.get("env_name")
        else:
            raise ValueError("Function does not have 'env_name' as an argument.")

        # Perform the package check
        package_name = f"gym_{env_name}"
        if env_name is not None and not is_package_available(package_name):
            pytest.skip(f"gym-{env_name} not installed")

        return func(*args, **kwargs)

    return wrapper


def require_package_arg(func):
    """
    Decorator that skips the test if the required package is not installed.
    This is similar to `require_env` but more general in that it can check any package (not just environments).
    As it need 'required_packages' in args, it also checks whether it is provided as an argument.
    If 'required_packages' is None, this check is skipped.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Determine if 'required_packages' is provided and extract its value
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
        if "required_packages" in arg_names:
            # Get the index of 'required_packages' and retrieve the value from args
            index = arg_names.index("required_packages")
            required_packages = args[index] if len(args) > index else kwargs.get("required_packages")
        else:
            raise ValueError("Function does not have 'required_packages' as an argument.")

        if required_packages is None:
            return func(*args, **kwargs)

        # Perform the package check
        for package in required_packages:
            if not is_package_available(package):
                pytest.skip(f"{package} not installed")

        return func(*args, **kwargs)

    return wrapper


def require_package(package_name):
    """
    Decorator that skips the test if the specified package is not installed.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_package_available(package_name):
                pytest.skip(f"{package_name} not installed")
            return func(*args, **kwargs)

        return wrapper

    return decorator
