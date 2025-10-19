"""Serve command - starts the dashboard backend."""

import os
import sys


def add_serve_parser(parser):
    """Add arguments for the serve command."""
    parser.add_argument(
        "--port",
        type=int,
        default=8124,
        help="Port to bind the server to (default: 8124)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker processes (default: 10)",
    )


def handle_serve(args):
    """Start the dashboard backend using uvicorn."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required to run the server.")
        print("Install it with: pip install uvicorn")
        return 1

    # Get the path to the backend module
    backend_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dashboard", "backend"
    )

    # Add backend directory to Python path so imports work correctly
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    print(f"Starting OpenWeights dashboard backend on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")

    # Change to backend directory so relative imports work
    original_dir = os.getcwd()
    os.chdir(backend_path)

    try:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=False,
        )
        return 0
    finally:
        os.chdir(original_dir)
