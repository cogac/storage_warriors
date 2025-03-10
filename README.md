# Dev-Phase - this README reflects the planned End-State of the project
# Storage Warriors -- The Solution for Storage Optimization

# Installation

To install our program, clone or download the REPO and go to the directory

```
git clone https://github.com/JustWatcher124/storage_warriors
cd storage_warriors
```

It is recommended to install all dependencies in a separate python virtual environment.
To achieve this recommendation:

```
python -m venv ./venv_storage_warriors
## For Windows
# In cmd.exe
./venv_storage_warriors\Scripts\activate.bat
# In PowerShell
./venv_storage_warriors\Scripts\Activate.ps1

## For GNU+Linux / MacOs Systems
source ./venv_storage_warriors/bin/activate
```

We supply a `requirements.txt` file that contains all necessary python modules.

You can install from this file with

```
pip install -r requirements.txt
```

See _Usage_ to see how to use the software

# Usage

Now the program is usable with `python3 -m start.py DATA`

`DATA` is the path or file to a data file or directory containing parts.

Supported file formats for DATA files: _parquet, csv, xlsx, xls, ods_
