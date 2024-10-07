import os
from pathlib import Path
import logging
# eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJodHRwczovL2lkZW50aXR5dG9vbGtpdC5nb29nbGVhcGlzLmNvbS9nb29nbGUuaWRlbnRpdHkuaWRlbnRpdHl0b29sa2l0LnYxLklkZW50aXR5VG9vbGtpdCIsImlhdCI6MTcyNjI2NDgyNywiZXhwIjoxNzI2MjY4NDI3LCJpc3MiOiJmaXJlYmFzZS1hZG1pbnNkay02cjM0eUB0YWJuaW5lLWF1dGgtMzQwMDE1LmlhbS5nc2VydmljZWFjY291bnQuY29tIiwic3ViIjoiZmlyZWJhc2UtYWRtaW5zZGstNnIzNHlAdGFibmluZS1hdXRoLTM0MDAxNS5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsInVpZCI6IkU5MXhCSnFhQ0FoSmJTY2dVbEhYNktseFdRRzMifQ.d-y3GIzgPuInX6wZnq-rnwpgDSJE_-hNVODBx0gcAS5p2k_Mn8_K5D1ltTl3dTtSxIV9aMEF687HLHm2DbYBkO07EFTb8449EDtGbt2czndfyseptaDCE9_6wel7xWthNGF3_lOOOLlCg0ZLhwnir3ObrW9iBfFvT8b9UfmOY3UngMyfiU-Hab6B4szr6ui8z9VwNL-BKttepamNjyhXodHv0xElWbWUWAzAEefFI9Y9x4uvydFzM53W5Z0oHiJ49gBsQGjFVWe1QauBKTMn31iQwBZdopS9wDQJbfavGYIZTWFNco9wWy9Q2YUbMnB9r-k3PJoqObza8B0HCtFmcw
logging.basicConfig(level = logging.INFO, format = '[%(asctime)s]: %(message)s:')

project_name = "text_Summarizer"  

list_of_files =[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb", 
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not  os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    
    else:
        logging.info(f"File {filename } already exists")