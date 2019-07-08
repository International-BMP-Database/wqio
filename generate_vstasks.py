import sys
import os
from pathlib import Path

TEMPLATE = """\
{{
    "version": "0.1.0",
    "isShellCommand": false,
    "args": [],
    "showOutput": "always",
    "echoCommand": false,
    "suppressTaskName": false,
    "tasks": [
        {{
            "taskName": "test",
            "command": "{pyexec:s}",
            "args": [
                "check_{modulename:s}.py",
                "--cov",
                "--pep8"
            ]
        }},
        {{
            "taskName": "notebooks",
            "options": {{
                "cwd": "${{workspaceRoot}}/docs/tutorial"
            }},
            "command": "{pyexec:s}",
            "args": ["make.py"]
        }},
        {{
            "taskName": "docs",
            "options": {{
                "cwd": "${{workspaceRoot}}/docs"
            }},
            "command": "make.bat",
            "args": ["html"]
        }}
    ]
}}
"""


if __name__ == '__main__':
    configdir = Path(".vscode")
    configdir.mkdir(exists_ok=True, parents=True)
    configpath = configdir / "tasks.json"

    if len(sys.argv) < 2:
        name = Path.cwd().name
    else:
        name = sys.argv[1]

    python = str(Path(sys.executable))
    config = TEMPLATE.format(pyexec=python, modulename=name)

    with configpath.open('w') as configfile:
        configfile.write(config)
