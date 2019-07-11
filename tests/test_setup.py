"""
This is to test setup.py

It's really just to get as close to 100% code coverage as possible.
"""
import subprocess

def test_setup():
    """Install xenith"""
    subprocess.run(["pip", "install", "-e", "setup.py"])
