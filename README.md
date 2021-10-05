# Vertebrae_segmentation

This is an associated repository of the work **Vertebrae segmentation, identification and localization using a graph optimization and a synergistic sycle**.

Pre-print at [link]().

It includes the proposed pipeline which is able to segment and label vertebrae from CT images of varient resolution, orientation and field of view. You could test the pipeline following the steps below.  

![visu](visu.png)

## Download the repository

``
git clone https://gitlab.inria.fr/spine/vertebrae_segmentation.git
``

## Set up a virtual environment

Firstly install the [anaconda or miniconda](https://docs.anaconda.com/anaconda/install/index.html).

Once you have the conda installed, create a virtual environment using the env.yml file in the repo.

``
conda env create -f env.yml
``

The environment is named as verse20, you could change it in the env.yml.

Then activate the environment by

``
conda activate verse20
``

Now you should hava all the dependencies that the pipeline needs.

## Get the data

The method is developed using the [VerSe20](https://verse2020.grand-challenge.org/) challenge public training set. The released models were trained using 80 scans of it. To test the pipeline on VerSe20 public and hidden testset, download the data from its [Github](https://github.com/anjany/verse). More details about the dataset can be found in the publication [Sekuboyina A et al., VerSe: A Vertebrae Labelling and Segmentation Benchmark for Multi-detector CT Images, 2021.](https://doi.org/10.1016/j.media.2021.102166).

## Run the pipeline on VerSe20 testset

The challenge data structure you downloaded should be like

> verse20_miccai_challenge
>
>       01_training
>
>               ...
>
>       02_validation (public testset)
>
>               GL017
>
>               GL045
>
>               ...
>
>       03_test (hidden testset)
>
>               ...
>

To process the testset, run

``
python test.py -D <path_to_folder>/02_validation
``

You could specify the save folder by adding -S <path_to_folder>. Otherwise, the results would be saved directly in results/.

