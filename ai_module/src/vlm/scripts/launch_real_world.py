#!/usr/bin/env python3
"""
Real-World Launch Script

This script reads topic_remap.yaml and launches vlm.launch with proper remapping.
This keeps the main launch file clean while still using config files.

Usage:
    python3 launch_real_world.py
    # or
    rosrun vlm launch_real_world.py
"""

import yaml
import subprocess
import sys
import os


def load_config(config_file):
    """Load the topic remapping configuration."""
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def launch_with_remapping(config):
    """Launch vlm.launch with remapping arguments."""
    
    # Build the roslaunch command
    cmd = [
        "roslaunch", "vlm", "vlm_real.launch",
        "real_world:=true",
        f"sensor_scan_topic:={config['sensor_scan_topic']}",
        f"registered_scan_topic:={config['registered_scan_topic']}",
        f"camera_image_topic:={config['camera_image_topic']}",
        f"odom_topic:={config['odom_topic']}",
        f"robot_pose_topic:={config['robot_pose_topic']}",
        f"navigation_goal_topic:={config['navigation_goal_topic']}",
        f"cmd_vel_topic:={config['cmd_vel_topic']}",
        f"robot_type:={config['robot_type']}"
    ]
    
    print("=" * 60)
    print("LAUNCHING AI MODULE WITH REAL-WORLD REMAPPING")
    print("=" * 60)
    print(f"Config file: {config_file}")
    print(f"Robot type: {config['robot_type']}")
    print(f"Environment: {config['environment']}")
    print()
    print("Topic Remapping:")
    print(f"  /sensor_scan -> {config['sensor_scan_topic']}")
    print(f"  /state_estimation -> {config['odom_topic']}")
    print(f"  /cmd_vel -> {config['cmd_vel_topic']}")
    print(f"  /way_point_with_heading -> {config['navigation_goal_topic']}")
    print()
    print("Launching...")
    print("=" * 60)
    
    # Execute the command
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error launching: {e}")
        sys.exit(1)


def main():
    """Main function."""
    # Find the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../config/topic_remap.yaml')
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        print("Please create the config file or check the path.")
        sys.exit(1)
    
    # Load config and launch
    config = load_config(config_file)
    launch_with_remapping(config)


if __name__ == '__main__':
    main() 