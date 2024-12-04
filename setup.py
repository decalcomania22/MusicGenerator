# setup.py

from music21 import *

# Initialize and create user settings
us = environment.UserSettings()
us.create()

print("music21 environment setup complete.")
for key in sorted(us.keys()):
    print(key)