"""
Training Scripts Package

This package contains training and management scripts for the Medical AI Assistant project.

Available Scripts:
- resume_training.py: CLI for resuming training with checkpoint management

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

__version__ = "1.0.0"
__all__ = ["resume_training_script"]

def get_available_scripts():
    """Get list of available training scripts"""
    return {
        "resume_training": {
            "script": "resume_training.py",
            "description": "Resume training with advanced checkpoint management",
            "commands": ["resume", "list", "monitor", "backup", "validate", "show"]
        }
    }

def main():
    """Main entry point for training scripts"""
    import sys
    from pathlib import Path
    
    print("Training Scripts Package v1.0.0")
    print("Available scripts:")
    
    scripts = get_available_scripts()
    for script_name, script_info in scripts.items():
        print(f"\n{script_name}:")
        print(f"  Description: {script_info['description']}")
        print(f"  Script: {script_info['script']}")
        print(f"  Commands: {', '.join(script_info['commands'])}")
    
    print("\nUsage:")
    print("  python -m training.scripts.resume_training resume --help")
    print("  python resume_training.py list --experiment my_experiment")

if __name__ == "__main__":
    main()