#!/usr/bin/env python3
import os
import re
import sys


def process_file(path, visited=None, with_tags=True):
    print(path)
    if visited is None:
        visited = set()
    if path in visited:
        return ""
    visited.add(path)

    if not os.path.exists(path):
        return f"<{path}>\n[missing file]\n</{path}>\n"

    if os.path.isdir(path):
        # For directories: list contents, recurse into README.md if present
        entries = os.listdir(path)
        out = [f"<{path}>"]
        out.append("\n".join(sorted(entries)))
        readme_path = os.path.join(path, "README.md")
        if os.path.exists(readme_path):
            out.append(process_file(readme_path, visited))
        out.append(f"</{path}>")
        return "\n".join(out) + "\n"

    # For files
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if with_tags:
        out = [f"<{path}>", content, f"</{path}>"]
    else:
        out = [content]

    # If it's markdown, follow relative links
    if path.endswith(".md"):
        link_pattern = re.compile(r"\[.*?\]\((?!http)(.*?)\)")
        for link in link_pattern.findall(content):
            link_path = os.path.normpath(os.path.join(os.path.dirname(path), link))
            out.append(process_file(link_path, visited))

    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 2:
        print("Usage: python llm_dot_txt.py README.md")
        sys.exit(1)

    root = sys.argv[1]
    combined = process_file(root, with_tags=False)

    # Cutoff stuff before first occurrence of "# OpenWeights"
    combined = "# OpenWeights" + combined.split("# OpenWeights", 1)[1]

    with open("llm.txt", "w", encoding="utf-8") as f:
        f.write(combined)

    with open("dev.md", "r") as f:
        dev = f.read()

    with open("LLM.md", "w", encoding="utf-8") as f:
        f.write(combined + "\n" + dev)


if __name__ == "__main__":
    main()
