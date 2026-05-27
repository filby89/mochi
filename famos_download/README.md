# FaMoS download scripts

MOCHI is trained and evaluated on the **FaMoS** dataset, which is released as part of
**[TEMPEH](https://tempeh.is.tue.mpg.de/)** (Bolkart et al., CVPR 2023). The scripts in this
folder are copied **verbatim from the TEMPEH release** and are provided here only for
convenience — all credit goes to the TEMPEH authors.

> **You must register at <https://tempeh.is.tue.mpg.de/> and agree to the TEMPEH/FaMoS license
> terms before downloading.** Each script prompts for your TEMPEH username and password.

## Requirements

`wget`, `unzip`, and [`7z`](https://www.7-zip.org/) must be installed.

## Usage

Run the scripts **from inside this folder**; data is downloaded to `./data/downloads` and
extracted to `./data/{training_data,test_data,test_data_subset}`.

```bash
cd famos_download

# Quick start — small paper test subset (images + calibrations + frame list)
bash fetch_test_subset.sh

# Full test set (images + calibrations + scans)
bash fetch_test_data.sh

# Full training set (images + calibrations + scans + FLAME registrations, ~500 GB extracted)
bash fetch_training_data.sh
```

`fetch_training_data.sh` / `fetch_test_data.sh` are umbrellas that call the component scripts
(`fetch_training_images.sh`, `fetch_training_scans.sh`, `fetch_registrations.sh`,
`fetch_test_images.sh`, `fetch_test_scans.sh`); run those individually if you only need part of
the data.

## Next step

These scripts give you the **raw** FaMoS captures. MOCHI consumes **, pre-rendered
multi-view grids**, so after downloading, run the preprocessing described in
[`../datasets/preprocess.md`](../datasets/preprocess.md) to produce the inputs the trainer
expects.
