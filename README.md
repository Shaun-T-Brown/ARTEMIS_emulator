# ARTEMIS_emulator

##Installation

Currently the package must be install `by hand'. 

The package requires the following:
- python3 (version 3.10.x or higher)
- numpy
- scipy
- h5py
- matplotlib
- dill
- scikit-learn (version 1.1.2 or higher)

First clone the github repository, ` got clone https://github.com/Shaun-T-Brown/ARTEMIS_emulator.git`

To allow the package to be called from other python scripts the source code needs to be added to your python path. This is most easily done by editing your .basrc (or similar) file

export PYTHONPATH="${PYTHONPATH}:/PATH/TO/PACKAGE/ARTEMIS_emulator/src/"


##Usage

The main functions to interface with the emualtors can be imported using the following:
```
from Emulator_functions import emulator #import emulator package

em = emulator() #initalise the emulation class
```

Currently there is no documentation. However, there are two example jupyter notebooks (found within ./examples) that demonstrate how to use the existing emulators and how to build new ones for statistics not already calculated.


