# Generation of Synthetic Chest X-Ray Images and Detection of COVID-19: a Deep Learning based Approach

This is the implementation of the paper Generation of Synthetic Chest X-Ray Images and Detection of COVID-19: a Deep Learning based Approach. 

`loader.py` contains the loading requirements for the dataset.

Required Directory Structure:

```
.
+--Train
|  +--.
|  +--COVID
|  +--NORMAL
+--Test
|  +--.
|  +--COVID
|  +--NORMAL

```

`main.py` contains the discriminator, generator, and the acgan function.

`trainer.py` contains the training methodology for the ACGAN, trained for `1200` epochs.

`utils.py` has the label smoothing function, the print logs function, and the function to generate noise and labels.

`generate.py` loads the trained generator weights and generates the CXR image.

