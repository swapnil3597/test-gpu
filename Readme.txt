# >> Code to Test GPU

# >> Use Python version 3 (code tested on 3.7.6)
# Make sure current directory contains following files
# Readme.txt, requirements.txt and run.py

# >> Create and activate a virtual environment
# Use pip install package_name, 
# pip3 install package_name or
# python3 -m pip install package_name 
# whichever suitable for install packages
pip install virtualenv
python -m venv gpu-check-env
source gpu-check-env/bin/activate

# >> Install required packages inside the environment
pip install -r requirements.txt

# >> Run code inside the environment
python run.py --batch-size 256 --train-steps 5000 --device gpu
# Change values of batch size and train steps as per requirement

# >> To check GPU utilization, on another terminal
watch -n 1 nvidia-smi

# >> Leave/Deactivate the virtual enviroment
deactivate