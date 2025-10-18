"""Fetch file content CLI command."""

import os
import sys

from openweights import OpenWeights


def add_fetch_parser(parser):
    """Add arguments for the fetch command."""
    parser.add_argument(
        "file_id",
        help="File ID to fetch",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (if not specified, writes to stdout)",
    )


def get_openweights_client():
    """Get OpenWeights client with authentication."""
    auth_token = os.getenv("OPENWEIGHTS_API_KEY")

    if not auth_token:
        print(
            "Error: OPENWEIGHTS_API_KEY environment variable not set", file=sys.stderr
        )
        print("Please set your API key:", file=sys.stderr)
        print("  export OPENWEIGHTS_API_KEY=your_token_here", file=sys.stderr)
        sys.exit(1)

    try:
        return OpenWeights(auth_token=auth_token)
    except Exception as e:
        print(f"Error initializing OpenWeights client: {str(e)}", file=sys.stderr)
        sys.exit(1)


def handle_fetch(args) -> int:
    """Handle the fetch command - download and display file content."""
    try:
        ow = get_openweights_client()

        # Fetch the file content
        try:
            content = ow.files.content(args.file_id)
        except Exception as e:
            print(
                f"Error: Failed to fetch file {args.file_id}: {str(e)}", file=sys.stderr
            )
            return 1

        # Write to output file or stdout
        if args.output:
            try:
                with open(args.output, "wb") as f:
                    f.write(content)
                print(f"File saved to: {args.output}", file=sys.stderr)
            except Exception as e:
                print(f"Error writing to file {args.output}: {str(e)}", file=sys.stderr)
                return 1
        else:
            # Write to stdout as binary
            sys.stdout.buffer.write(content)

        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
