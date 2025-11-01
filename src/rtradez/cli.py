"""
RTradez CLI Entry Point.

Main entry point for the RTradez command-line interface.
"""

def main():
    """Main CLI entry point."""
    from .cli.main import app
    app()

if __name__ == "__main__":
    main()