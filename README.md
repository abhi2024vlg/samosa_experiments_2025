## Steps for installation

- Make executable using `chmod +x install.sh`
- `./install.sh`
- `conda env create -f environment.yaml -n FoR`
- Activate conda env using `conda activate FoR`
- Install some dependencies `pip install tensorboard`
- Setup HF credentials using `huggingface-cli login --token <TOKEN>`
- Run script using `nohup python3 main.py`