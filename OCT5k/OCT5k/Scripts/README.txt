# Build the environment

python3 -m venv venv

source venv/bin/activate

pip install -U pip

pip install -U build

pip install -r requirements.txt

python -m jupyterlab

# Run Prepare_Images_Download.ipynb and execute every cell for downloading the images
