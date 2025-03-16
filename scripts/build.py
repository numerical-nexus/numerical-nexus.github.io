#!/usr/bin/env python3

import os
import subprocess
import argparse
from typing import List
from pathlib import Path


def export_html_wasm(notebook_path: str, output_dir: str, as_app: bool = False) -> bool:
    """Export a single marimo notebook to HTML format.

    Returns:
        bool: True if export succeeded, False otherwise
    """
    output_path = notebook_path.replace(".py", ".html")

    cmd = ["marimo", "export", "html-wasm"]
    if as_app:
        print(f"Exporting {notebook_path} to {output_path} as app")
        cmd.extend(["--mode", "run", "--no-show-code"])
    else:
        print(f"Exporting {notebook_path} to {output_path} as notebook")
        cmd.extend(["--mode", "edit"])

    try:
        output_file = os.path.join(output_dir, output_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        cmd.extend([notebook_path, "-o", output_file])
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error exporting {notebook_path}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error exporting {notebook_path}: {e}")
        return False


def generate_index(all_notebooks: List[str], output_dir: str) -> None:
    """Generate the index.html file with sidebar and iframe."""
    print("Generating index.html")

    index_path = os.path.join(output_dir, "index.html")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(index_path, "w") as f:
            f.write(
                """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marimo Notebooks</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <nav class="col-md-3 col-lg-2 d-md-block bg-light sidebar p-3">
                <h4 class="text-center">Marimo Notebooks</h4>
                
                <div class="mb-3">
                    <label for="notebook-dropdown" class="form-label">Select a Notebook</label>
                    <select id="notebook-dropdown" class="form-select">
                        <option value="">Select...</option>
"""
            )

            # Add options dynamically
            for notebook in all_notebooks:
                notebook_name = notebook.split("/")[-1].replace(".py", "")
                display_name = notebook_name.replace("_", " ").title()
                notebook_path = notebook.replace(".py", ".html")

                f.write(f'                        <option value="{notebook_path}">{display_name}</option>\n')

            f.write(
                """                    </select>
                </div>
            </nav>

            <!-- Main Content -->
            <main class="col-md-9 col-lg-10 p-4">
                <iframe id="notebook-frame" src="about:blank" class="w-100 border rounded shadow" style="height: 90vh;"></iframe>
            </main>
        </div>
    </div>

    <script>
        document.addEventListener("DOMContentLoaded", function () {
            const frame = document.getElementById("notebook-frame");
            const notebookDropdown = document.getElementById("notebook-dropdown");

            notebookDropdown.addEventListener("change", function () {
                frame.src = this.value || "about:blank";
            });
        });
    </script>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
            )

    except IOError as e:
        print(f"Error generating index.html: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build marimo notebooks")
    parser.add_argument(
        "--output-dir", default="_site", help="Output directory for built files"
    )
    args = parser.parse_args()

    all_notebooks: List[str] = []
    for directory in ["notebooks", "apps"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        all_notebooks.extend(str(path) for path in dir_path.rglob("*.py"))

    if not all_notebooks:
        print("No notebooks found!")
        return

    # Export notebooks sequentially
    for nb in all_notebooks:
        export_html_wasm(nb, args.output_dir, as_app=nb.startswith("apps/"))

    # Generate index only if all exports succeeded
    generate_index(all_notebooks, args.output_dir)


if __name__ == "__main__":
    main()
