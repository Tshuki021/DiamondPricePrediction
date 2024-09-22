from setuptools import find_packages, setup  # Used to package and distribute the project
from typing import List  # For specifying the return type of the function

HYPEN_E_DOT = '-e .'  # Represents the local directory for editable installs

def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements file and returns a list of dependencies.
    Removes '-e .' if present, which is used for local installs.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()  # Read all lines from the requirements file
        requirements = [req.replace("\n", "") for req in requirements]  # Remove newline characters

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)  # Remove '-e .' from the list
    return requirements

# Setup function to define project metadata and requirements
setup(
    name='DiamondPricePrediction',  # Project name
    version='0.0.1',  # Initial version
    author='Shubhanshu',  # Author name
    author_email='shubhanshukrsharma@gmail.com',  # Author email
    install_requires=get_requirements('requirements.txt'),  # List of required packages
    packages=find_packages()  # Automatically find and include all packages
)
