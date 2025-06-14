## Setup

1. Follow the installation instructions at https://docs.astral.sh/uv/getting-started/installation/ . Test that the installation works by executing the following command (without the " ") in the Windows Powershell: `` uv --version``

1. In Powershell go to the directory where the code is located and the "pyproject.toml" file is. You can use "cd \<PATH\>" where \<PATH\> is the folder path to change the directory.

1. Install all the dependencies: ``uv sync``

1. Run a script: ``uv run python <PATH>`` where \<PATH\> is the file path to a *.py file. For example: ``uv run python run_data_preparation.py``

1. Close Windows Powershell.

## Run project

### Data preparation

We prepare our data by running the `run_data_preparation.py` script. Open `run_data_preparation.py` in your text editor of choice and search the line:

```python
INPUTFOLDER_PATH: Path = Path('/home/kepiej/Dropbox/ATarnaud/')
```

Replace the path between quotes '' to your project Dropbox folder location on your pc.

Now open your terminal (or powershell):
1. Navigate to the folder with the code: ``cd <PATH>`` where \<PATH\> is the folder path. This path should end with the folder "MultitimeMultimomentRating".

2. Run the script:
``
uv run python run_data_preparation.py
``

When the script finishes you should find a 'prices.parquet' file in the same folder.

### Run fundrating

Open your terminal (or powershell):
1. Navigate to the folder with the code: ``cd <PATH>`` where \<PATH\> is the folder path. This path should end with the folder "MultitimeMultimomentRating".

2. Run the script:
``
uv run python run_fundrating.py
``
