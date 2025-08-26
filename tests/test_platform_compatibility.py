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

import platform

import pytest

from tests.utils import (
    get_normalized_machine,
    is_apple_silicon,
    is_intel_mac,
    is_linux,
    is_macos,
    require_apple_silicon,
    require_arm64_kernel,
    require_compatible_kernel,
    require_macos,
    require_x86_64_kernel,
)


def test_platform_detection():
    """Test that platform detection works correctly."""
    machine = get_normalized_machine()
    assert machine in ["x86_64", "arm64", "i386", "i686", "armv7l"], f"Unknown platform: {machine}"


def test_apple_silicon_detection():
    """Test Apple Silicon detection."""
    if platform.system() == "Darwin" and platform.machine().lower() in ["arm64", "aarch64"]:
        assert is_apple_silicon()
        assert is_macos()
        assert not is_intel_mac()
        assert not is_linux()
    else:
        assert not is_apple_silicon()


def test_intel_mac_detection():
    """Test Intel Mac detection."""
    if platform.system() == "Darwin" and platform.machine().lower() in ["x86_64", "amd64"]:
        assert is_intel_mac()
        assert is_macos()
        assert not is_apple_silicon()
        assert not is_linux()
    else:
        assert not is_intel_mac()


@require_compatible_kernel
def test_compatible_kernel_decorator():
    """Test that compatible kernel decorator allows x86_64 and arm64."""
    # This test should pass on x86_64 and arm64
    assert True


def test_x86_64_decorator_behavior():
    """Test that x86_64 decorator behavior is correct."""
    machine = get_normalized_machine()
    
    @require_x86_64_kernel
    def dummy_test():
        return True
    
    if machine == "x86_64":
        # Should not skip on x86_64
        assert dummy_test() is True
    else:
        # Should skip on non-x86_64
        with pytest.raises(pytest.skip.Exception):
            dummy_test()


def test_arm64_decorator_behavior():
    """Test that arm64 decorator behavior is correct."""
    machine = get_normalized_machine()
    
    @require_arm64_kernel
    def dummy_test():
        return True
    
    if machine == "arm64":
        # Should not skip on arm64
        assert dummy_test() is True
    else:
        # Should skip on non-arm64
        with pytest.raises(pytest.skip.Exception):
            dummy_test()


@require_apple_silicon  
def test_apple_silicon_specific_functionality():
    """Test that runs only on Apple Silicon."""
    # This test should only run on Apple Silicon Macs
    assert is_apple_silicon()
    assert is_macos()


@require_macos
def test_macos_specific_functionality():
    """Test that runs only on macOS (any architecture)."""
    # This test should run on both Intel and Apple Silicon Macs
    assert is_macos()


def test_normalized_machine_function():
    """Test that machine normalization works correctly."""
    machine = get_normalized_machine()
    raw_machine = platform.machine().lower()
    
    if raw_machine == "amd64":
        assert machine == "x86_64"
    elif raw_machine == "aarch64":
        assert machine == "arm64"
    else:
        assert machine == raw_machine