import re
from typing import Optional, Tuple

from huggingface_hub import HfApi


def is_progress_line(line):
    """
    Check if a line is likely a progress indicator that should be skipped
    """
    # Common patterns in progress lines
    progress_patterns = [
        r"⠋|⠙|⠹|⠸|⠼|⠴|⠦|⠧|⠇|⠏",  # Spinner characters
        r"\[#*\s*\]",  # Progress bars
        r"idealTree:.*",  # npm specific progress
        r"reify:.*",  # npm specific progress
        r"\([^)]*\)\s*�[⠀-⣿]",  # Progress with spinner
    ]

    return any(re.search(pattern, line) for pattern in progress_patterns)


def clean_ansi(text):
    # Step 1: Remove ANSI escape sequences
    ansi_escape = re.compile(
        r"""
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI
            \[
            [0-?]*  # parameter bytes
            [ -/]*  # intermediate bytes
            [@-~]   # final byte
        )
    """,
        re.VERBOSE,
    )

    text = ansi_escape.sub("", text)

    # Step 2: Split into lines and process terminal control characters
    lines = text.splitlines()
    screen = []
    current_line = ""

    for line in lines:
        # Handle carriage return (simulate line overwrites)
        if "\r" in line:
            parts = line.split("\r")
            # Process each part
            for part in parts:
                # Handle backspaces in this part
                while "\x08" in part:
                    part = re.sub(r".\x08", "", part, 1)

                # Overwrite current line from the start
                current_line = part
        else:
            # Handle backspaces
            while "\x08" in line:
                line = re.sub(r".\x08", "", line, 1)
            current_line = line

        # Only add non-empty lines that aren't just progress indicators
        if current_line.strip() and not is_progress_line(current_line):
            screen.append(current_line)

    # Remove duplicate consecutive lines
    unique_lines = []
    prev_line = None
    for line in screen:
        if line != prev_line:
            unique_lines.append(line)
            prev_line = line

    # If we have 0 lines because all where progress lines, return the last one
    if not unique_lines:
        return current_line

    # Join lines and clean up any remaining control characters
    cleaned_output = "\n".join(unique_lines) + "\n"
    return cleaned_output


def validate_huggingface_secrets(
    hf_token: str, hf_user: Optional[str] = None, hf_org: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validate HuggingFace secrets by checking:
    1. The token is valid
    2. If hf_user is provided, verify it matches the token's user
    3. If hf_org is provided, verify the token has access to the organization

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Initialize HF API with the token
        api = HfApi(token=hf_token)

        # Get user info to verify token is valid and get username
        user_info = api.whoami()
        if not user_info:
            return False, "Invalid HuggingFace token"

        token_username = user_info.get("name")

        # Verify username if provided
        if hf_user and hf_user.lower() != token_username.lower():
            return False, f"Token belongs to user '{token_username}', not '{hf_user}'"

        # Verify organization access if provided
        if hf_org:
            # Get organizations from user info
            organizations = user_info.get("orgs", [])
            org_names = [org["name"] for org in organizations]

            if hf_org not in org_names:
                return False, f"Token does not have access to organization '{hf_org}'"

        return True, "Valid HuggingFace credentials"

    except Exception as e:
        return False, f"Error validating HuggingFace credentials: {str(e)}"


def validate_organization_secrets(secrets: dict) -> Tuple[bool, str]:
    """
    Validate all organization secrets.
    Currently validates:
    - HuggingFace credentials (HF_TOKEN, HF_USER, HF_ORG)

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Validate HuggingFace secrets if any are provided
    hf_token = secrets.get("HF_TOKEN")
    hf_user = secrets.get("HF_USER")
    hf_org = secrets.get("HF_ORG")

    if hf_token:
        is_valid, error_message = validate_huggingface_secrets(
            hf_token, hf_user, hf_org
        )
        if not is_valid:
            return False, error_message

    # Add validation for other secret types here

    return True, "All secrets are valid"
