#!/usr/bin/env python3
"""
RTradez CLI Quick Demo.

Shows key CLI features with a focused demonstration.
"""

import subprocess
import sys

def demo_cli():
    """Run a focused CLI demonstration."""
    
    print("🚀 RTradez CLI Quick Demo")
    print("=" * 50)
    
    base_cmd = f"{sys.executable} -m rtradez.cli.main"
    
    # Key demonstrations
    demos = [
        (f"{base_cmd} version", "System Information"),
        (f"{base_cmd} risk position-size --capital 100000 --method kelly --return 0.12 --volatility 0.20", "Position Sizing"),
        (f"{base_cmd} portfolio status", "Portfolio Dashboard"),
        (f"{base_cmd} data align --simulate --frequency daily", "Data Processing"),
        (f"{base_cmd} analysis optimize --strategy kelly --trials 20", "Strategy Optimization"),
    ]
    
    for cmd, description in demos:
        print(f"\n📋 {description}")
        print("-" * 30)
        print(f"$ rtradez {cmd.split(base_cmd)[1].strip()}")
        print()
        
        # Run command
        result = subprocess.run(cmd, shell=True, capture_output=False)
        
        if result.returncode == 0:
            print("✅ Success")
        else:
            print("❌ Failed")
    
    print(f"\n🎯 RTradez CLI provides comprehensive tools for:")
    print("  • Risk management and position sizing")
    print("  • Portfolio management and coordination")
    print("  • Multi-frequency data processing")
    print("  • Strategy analysis and optimization")
    print("  • System configuration and monitoring")

if __name__ == "__main__":
    demo_cli()