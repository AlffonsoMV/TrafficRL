#!/usr/bin/env python
"""
Test script to verify Traffic RL installation.
"""

import sys
import importlib.util

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Run installation tests."""
    print("Testing Traffic RL installation...")
    
    # List of required modules
    required_modules = [
        "numpy", "torch", "matplotlib", "gymnasium", 
        "pygame", "pandas", "seaborn", "tqdm",
        "traffic_rl"
    ]
    
    # Check each module
    all_ok = True
    for module in required_modules:
        if check_module(module):
            print(f"✅ {module} is installed.")
        else:
            print(f"❌ {module} is NOT installed!")
            all_ok = False
    
    # Test traffic_rl submodules
    if check_module("traffic_rl"):
        try:
            # Try importing a few key submodules
            from traffic_rl import cli
            from traffic_rl.agents.base import BaseAgent
            print("✅ Traffic RL core modules are accessible.")
        except ImportError as e:
            print(f"❌ Traffic RL submodule import failed: {e}")
            all_ok = False
    
    # Final verdict
    if all_ok:
        print("\n✅ All dependencies are installed correctly.")
        print("✅ Traffic RL installation looks good!")
        print("\nYou can run 'traffic_rl --help' to get started.")
        return 0
    else:
        print("\n❌ Some dependencies are missing or not working correctly.")
        print("Please check the output above and reinstall if necessary.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 