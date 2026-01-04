"""Main entry point for the Smart Glasses application."""
import argparse
import sys
from app.orchestrator import SmartGlassesOrchestrator
from app.layers.layer5_output import OutputMode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Smart Glasses Computer Vision System")
    parser.add_argument(
        "--mode",
        choices=["navigation", "description"],
        default="navigation",
        help="Output mode: navigation (short commands) or description (rich context)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = SmartGlassesOrchestrator()
    
    # Set output mode
    mode = OutputMode.NAVIGATION if args.mode == "navigation" else OutputMode.DESCRIPTION
    orchestrator.set_output_mode(mode)
    
    # Update camera ID if specified
    if args.camera != 0:
        orchestrator.sensor.camera_id = args.camera
    
    try:
        # Run main loop
        orchestrator.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

