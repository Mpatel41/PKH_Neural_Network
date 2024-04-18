import subprocess
import os
packages = ["matplotlib", "scikit-learn", "tensorflow", "scikeras", "keras"]

for package in packages: 
    install_pack = f'pip install {package}'
    os.system(install_pack) 
