# Untitled

# Vertebrae segmentation

This is an associated repository of the work 

**Di Meng et al., Vertebrae segmentation, identification and localization using a graph optimization and a synergistic cycle, 2021.**

Pre-print: [link]().

It includes the proposed pipeline which is able to segment and label vertebrae from CT images of variant resolution, orientation and field of view. You could test the pipeline following the steps below.

## Download the repository

```bash
git clone https://gitlab.inria.fr/spine/vertebrae_segmentation.git
```

## Set up a virtual environment

Firstly install the [anaconda or miniconda](https://docs.anaconda.com/anaconda/install/index.html).

Once you have the conda installed, create a virtual environment using the env.yml file in the repo.

```bash
conda env create -f env.yml
```

The environment is named as verse20, you could change it in the env.yml.

Then activate the environment by

```bash
conda activate verse20
```

Now you should have all the dependencies that the pipeline needs.

## Get the data

The method is developed using the [VerSe20](https://verse2020.grand-challenge.org/) challenge public training set. The released models were trained using 80 scans of it. To test the pipeline on VerSe20 public and hidden testset, download the data from its [Github](https://github.com/anjany/verse). More details about the dataset can be found in the publication [Sekuboyina A et al., VerSe: A Vertebrae Labelling and Segmentation Benchmark for Multi-detector CT Images, 2021.](https://doi.org/10.1016/j.media.2021.102166).

## Run the pipeline on VerSe20 testset

The challenge data structure you downloaded should be like

```
../verse20_miccai_challenge
		01_training
				...
		02_validation (public testset)
				GL017
						GL017_CT_ax.nii.gz
						GL017_CT_ax_seg.nii.gz
						GL017_CT_ax_iso-ctd.json
						GL017_CT_ax.png
				...
		03_test (hidden testset)
				...
```

To process the public or hidden testset, run

```bash
python test.py -D <path to folder>/02_validation
```

The output for each input CT scan is a multi-label segmentation mask and a json file with labels and centroid coordinates.

You could specify the save folder by 

```bash
python test.py -D <path to folder>/02_validation -S <path to save folde>
```

Otherwise, the results would be saved directly in a subfolder results/.

To process one scan instead of the whole dataset, provide the scan ID, eg. GL017

```bash
python test.py -D <path to folder>/02_validation -V GL017
```