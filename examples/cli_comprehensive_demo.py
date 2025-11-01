#!/usr/bin/env python3
"""
RTradez CLI Comprehensive Demo.

Demonstrates all major CLI capabilities including risk management,
portfolio management, data processing, analysis, and configuration.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a CLI command and display the description."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    # Run the command
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    print("-" * 60)
    print(f"Exit code: {result.returncode}")
    
    # Add a small delay between commands
    time.sleep(1)
    return result.returncode == 0

def main():
    """Run comprehensive CLI demonstration."""
    
    print("🚀 RTradez CLI Comprehensive Demo")
    print("=" * 60)
    
    # Get the base command
    base_cmd = f"{sys.executable} -m rtradez.cli.main"
    
    # Demo commands in logical order
    demos = [
        # System Information
        (f"{base_cmd} version", "Show RTradez version and system information"),
        (f"{base_cmd} status", "Display system status and component health"),
        
        # Risk Management Demos
        (f"{base_cmd} risk position-size --capital 250000 --method kelly --return 0.12 --volatility 0.18", 
         "Calculate Kelly Criterion position size"),
        
        (f"{base_cmd} risk compare-methods --capital 500000 --return 0.15 --volatility 0.22", 
         "Compare different position sizing methods"),
        
        (f"{base_cmd} risk risk-limits --capital 1000000 --max-exposure 1.1 --max-var 0.04", 
         "Configure portfolio risk limits"),
        
        (f"{base_cmd} risk monitor --capital 750000 --simulate --duration 5", 
         "Run simulated risk monitoring"),
        
        # Portfolio Management Demos
        (f"{base_cmd} portfolio create --capital 2000000 --max-strategies 8 --rebalance-freq weekly", 
         "Create portfolio configuration"),
        
        (f"{base_cmd} portfolio add-strategy --name 'Iron_Condor_SPY' --allocation 0.30 --return 0.08 --volatility 0.12", 
         "Add strategy to portfolio"),
        
        (f"{base_cmd} portfolio status --detailed", 
         "Show detailed portfolio status"),
        
        (f"{base_cmd} portfolio rebalance --dry-run", 
         "Show portfolio rebalancing plan"),
        
        (f"{base_cmd} portfolio performance --period 3M", 
         "Analyze portfolio performance"),
        
        # Data Processing Demos
        (f"{base_cmd} data align --simulate --frequency weekly --method outer --fill interpolate", 
         "Demonstrate temporal data alignment"),
        
        (f"{base_cmd} data bucket --simulate --type trading --size 1H --agg ohlc", 
         "Show time series bucketing"),
        
        (f"{base_cmd} data profile --analyze-gaps", 
         "Profile dataset temporal characteristics"),
        
        (f"{base_cmd} data sources", 
         "List available data sources"),
        
        # Analysis Demos
        (f"{base_cmd} analysis optimize --strategy kelly --trials 50 --objective sharpe", 
         "Optimize strategy parameters"),
        
        (f"{base_cmd} analysis backtest --strategy momentum --start-date 2023-01-01 --capital 500000", 
         "Run strategy backtest"),
        
        (f"{base_cmd} analysis compare --strategies kelly momentum mean_reversion --metric sharpe", 
         "Compare multiple strategies"),
        
        # Configuration Demos
        (f"{base_cmd} config show", 
         "Display current configuration"),
        
        (f"{base_cmd} config validate", 
         "Validate system configuration"),
    ]
    
    # Track results
    successful = 0
    total = len(demos)
    
    print(f"\nRunning {total} CLI demonstrations...\n")
    
    for i, (cmd, description) in enumerate(demos, 1):
        print(f"\n[{i}/{total}] Starting demo...")
        
        success = run_command(cmd, description)
        if success:
            successful += 1
            print("✅ Demo completed successfully")
        else:
            print("❌ Demo failed")
        
        # Add extra spacing between major sections
        if i in [2, 6, 11, 15, 18]:  # End of each major section
            print("\n" + "🔹" * 60)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 DEMO SUMMARY")
    print(f"{'='*60}")
    print(f"Total Demos: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success Rate: {successful/total:.1%}")
    
    print(f"\n🎯 CLI CAPABILITIES DEMONSTRATED:")
    capabilities = [
        "🛡️ Risk Management & Position Sizing",
        "🏦 Portfolio Management & Coordination", 
        "📊 Multi-Frequency Data Processing",
        "📈 Strategy Analysis & Optimization",
        "⚙️ Configuration Management",
        "🎮 Interactive Command Interface",
        "📋 Rich Text Formatting & Tables",
        "🔍 System Status & Health Monitoring"
    ]
    
    for capability in capabilities:
        print(f"  {capability}")
    
    print(f"\n💡 NEXT STEPS:")
    next_steps = [
        "Initialize configuration: rtradez config init",
        "Set up API keys: rtradez config set-api <provider> <key>",
        "Create portfolio: rtradez portfolio create",
        "Optimize strategies: rtradez analysis optimize",
        "Monitor risk: rtradez risk monitor",
        "Process data: rtradez data align --input <files>"
    ]
    
    for step in next_steps:
        print(f"  • {step}")
    
    print(f"\n🔧 PRODUCTION FEATURES:")
    features = [
        "Comprehensive CLI with rich formatting",
        "Pydantic data validation throughout",
        "Interactive configuration management",
        "Real-time risk monitoring",
        "Multi-strategy portfolio coordination",
        "Advanced data processing capabilities",
        "Strategy optimization and backtesting",
        "Extensible plugin architecture"
    ]
    
    for feature in features:
        print(f"  ✨ {feature}")
    
    print(f"\n🚀 RTradez CLI Demo Complete!")
    
    if successful == total:
        print("🎉 All demonstrations completed successfully!")
    else:
        print(f"⚠️ {total - successful} demonstrations had issues - check logs above")


if __name__ == "__main__":
    main()