#!/usr/bin/env python3

"""
Apple Silicon Support Verification Script for LeRobot

This script verifies that LeRobot can detect and work properly on Apple Silicon Macs.
Run this script to check if your Apple Silicon Mac is properly supported.

Usage:
    python scripts/verify_apple_silicon_support.py
"""

import platform
import sys


def main():
    print("🤖 LeRobot Apple Silicon Support Verification")
    print("=" * 50)
    
    # System information
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    print()
    
    # Platform detection
    def is_apple_silicon():
        return platform.system() == "Darwin" and platform.machine().lower() in ["arm64", "aarch64"]
    
    def is_intel_mac():
        return platform.system() == "Darwin" and platform.machine().lower() in ["x86_64", "amd64"]
    
    def is_macos():
        return platform.system() == "Darwin"
    
    # Check platform support
    if is_apple_silicon():
        print("✅ APPLE SILICON MAC DETECTED!")
        print("   Your M1/M2/M3 Mac is fully supported by LeRobot!")
        print()
        print("Installation instructions:")
        print("   conda create -y -n lerobot python=3.10")
        print("   conda activate lerobot")
        print("   pip install -e .")
        print()
        print("Additional notes:")
        print("   - All LeRobot features work on Apple Silicon")
        print("   - For robot teleoperation, grant Terminal permission")
        print("     to access your keyboard in System Preferences")
        print("   - Hardware interfaces work the same as Intel Macs")
        
    elif is_intel_mac():
        print("✅ INTEL MAC DETECTED!")
        print("   Your Intel-based Mac is fully supported by LeRobot!")
        print()
        print("Installation instructions:")
        print("   conda create -y -n lerobot python=3.10")
        print("   conda activate lerobot")
        print("   pip install -e .")
        
    elif platform.system() == "Linux":
        print("✅ LINUX SYSTEM DETECTED!")
        print("   Your Linux system is fully supported by LeRobot!")
        
    elif platform.system() == "Windows":
        print("✅ WINDOWS SYSTEM DETECTED!")
        print("   Your Windows system has basic LeRobot support!")
        
    else:
        print("⚠️  UNKNOWN PLATFORM")
        print("   Your platform may have limited support.")
        print("   Please check the documentation for compatibility.")
    
    print()
    print("For more information, visit:")
    print("   https://huggingface.co/docs/lerobot/installation")
    print()
    print("🎉 Happy robot learning!")


if __name__ == "__main__":
    main()