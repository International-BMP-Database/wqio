import sys
from pathlib import Path

TEMPLATE = """\
{{
    "version": "2.0.0",
    "args": [],
    "echoCommand": false,
    "runner": "terminal",
    "tasks": [
        {{
            "label": "format-code",
            "type": "shell",
            "command": "black",
            "args": [
                "*.py",
                "{modulename:s}/*.py",
                "{modulename:s}/*/*.py",
            ],
        }},
        {{
            "label": "test",
            "type": "shell",
            "command": "{pyexec:s}",
            "args": [
                "check_{modulename:s}.py",
                "--tb=short",
                "--strict",
                "--cov-report=xml:cov.xml",
                "--cov={modulename:s}"
            ],
            "group": {{
                "kind": "test",
                "isDefault": true
            }}
        }},
        {{
            "label": "test-lastfailed",
            "command": "{pyexec:s}",
            "args": [
                "check_{modulename:s}.py",
                "--strict",
                "--tb=short",
                "--lf",
            ],
            "group": {{
                "kind": "test",
                "isDefault": true
            }},
            "type": "shell"
        }},
        {{
            "label": "test-debuggin",
            "command": "{pyexec:s}",
            "args": [
                "check_{modulename:s}.py",
                "--strict",
                "--tb=short",
                "--lf",
                "--pdb"
            ],
            "group": {{
                "kind": "test",
                "isDefault": true
            }},
            "type": "shell"
        }},
        {{
            "label": "notebooks",
            "options": {{
                "cwd": "${{workspaceRoot}}/docs/tutorial"
            }},
            "command": "{pyexec:s}",
            "args": ["make.py"]
        }},
        {{
            "label": "docs",
            "options": {{
                "cwd": "${{workspaceRoot}}/docs"
            }},
            "command": "make.bat",
            "args": ["html"]
        }}
    ]
}}
"""


if __name__ == "__main__":
    configdir = Path(".vscode")
    configdir.mkdir(exist_ok=True, parents=True)
    configpath = configdir / "tasks.json"

    python = str(Path(sys.executable))
    if len(sys.argv) < 2:
        name = Path.cwd().name
    else:
        name = sys.argv[1]

    config = TEMPLATE.format(pyexec=python, modulename=name)
    with configpath.open("w") as configfile:
        configfile.write(config)
