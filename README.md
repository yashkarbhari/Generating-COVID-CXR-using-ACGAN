# Generation of Synthetic Chest X-Ray Images and Detection of COVID-19: a Deep Learning based Approach

This is the official implementation of the paper [Generation of Synthetic Chest X-Ray Images and Detection of COVID-19: a Deep Learning based Approach.](https://www.mdpi.com/2075-4418/11/5/895) 

## Usage


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

`loader.py` contains the loading requirements for the dataset.

`main.py` contains the discriminator, generator, and the acgan function.

`trainer.py` contains the training methodology for the ACGAN, trained for `1200` epochs.

`utils.py` has the label smoothing function, the print logs function, the function to generate noise and labels, and the function to plot the loss graph.

`generate.py` loads the trained generator weights and generates the CXR image.

We have added 50 synthetic images in `COVID-19 (Synthetic)`. Remaining synthetic images are available on request, mail yashkarbhari17@gmail.com .

Training dataset was used from the following:
1) https://github.com/ieee8023/covid-chestxray-dataset
2) https://github.com/agchung/Figure1-COVID-chestxray-dataset
3) https://github.com/agchung/Actualmed-COVID-chestxray-dataset
4) https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

## Model structure

<p align="center">
  <img src="images/acgan.png" width="800" height="400"/>
</p>

## Generated Images



<table>
  <tr>
    <td>COVID-19 CXR</td>
     <td>Normal CXR</td>
     
  </tr>
  <tr>
    <td><img src="images/covid_grid.png"></td>
    <td><img src="images/normal_grid.png"></td>
  </tr>
 </table>

## Citation
```
@Article{diagnostics11050895,
author = {Karbhari, Yash and Basu, Arpan and Geem, Zong Woo and Han, Gi-Tae and Sarkar, Ram},
title = {Generation of Synthetic Chest X-ray Images and Detection of COVID-19: A Deep Learning Based Approach},
journal = {Diagnostics},
volume = {11},
year = {2021},
ARTICLE-NUMBER = {895}
}
```
