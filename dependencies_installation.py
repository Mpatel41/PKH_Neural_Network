import subprocess
packages = ["matplotlib", "scikit-learn", "tensorflow", "scikeras"]

for package in packages: 
      install_pack = f'pip install {package}'
      os.system(install_pack) 
