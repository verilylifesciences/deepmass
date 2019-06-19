# Running DeepMass:Prism on Google Cloud ML

## Overview

Peptide fragmentation spectra are routinely predicted in the interpretation of
mass spectrometry-based proteomics data. Unfortunately, the generation of
fragment ions is not well enough understood to estimate fragment ion intensities
accurately. In collaboration with the Max Planck Institute, we demonstrated that
machine learning can predict peptide fragmentation patterns in mass
spectrometers with accuracy within the uncertainty of the measurements. Here, we
provide a set of instructions for how to use the model on the Google Cloud
Machine Learning Engine (CMLE).

Briefly, running DeepMass:Prism consists of three steps:

*   Preprocess your input peptide table (this step is usually run locally).
*   Submit a prediction job to DeepMass’ CMLE.
*   Postprocess the outputs (this step is also usually run locally).

## Prerequisites

*   Clone this repository to your local machine, using the following command: \
    `git clone https://github.com/verilylifesciences/deepmass.git`
*   There are no system-specific requirements.

## Install dependencies

*   Numpy (`pip install numpy`)
*   Pandas (`pip install pandas`)
*   Tensorflow v1.7 (`pip install tensorflow==1.7.0`)
*   Google Cloud SDK ([link](https://cloud.google.com/sdk/))

## Data format

Input data table should be written in CSV (comma-separated) file format and
contain at least the following columns:

*   Peptide sequence:
    *   Amino-acid modifications should be given in “(modification)” format -
        for example, “ACDM(ox)FK” is a valid format,
    *   All cysteine residues are considered as carbamylated.
    *   N-terminal acetylations are ignored.
    *   Stay tuned, a larger list of supported modifications will be added soon.
*   Charge:
    *   Our model can handle charges up to 7.
*   Fragmentation type:
    *   Our model currently supports HCD and CID fragmentations.
*   Mass analyzer:
    *   Our model currently supports FTMS and ITMS analyzers.

The input table can contain any number of columns, and the 4 required columns
can be arbitrarily named - the preprocessing step will map the column names as
needed. Here’s an example input:

```
ModifiedSequence,Charge,Fragmentation,MassAnalyzer
AKM(ox)LIVR,3,HCD,ITMS
ILFWYK,2,CID,FTMS
```

## Pre-processing step

The preprocessing step converts the input table into a CMLE friendly format. To
do the pre-processing locally, run the following commands:

```
DATA_DIR="./data"  # Location to folder with input table.
python preprocess.py \
  --input_data="${DATA_DIR}/input_table.csv" \
  --output_data_dir="${DATA_DIR}" \
  --sequence_col="ModifiedSequence" \
  --charge_col="Charge" \
  --fragmentation_col="Fragmentation" \
  --analyzer_col="MassAnalyzer"
```

The preprocessing step will write two files into the `"${DATA_DIR}"` folder:
input.json and metadata.tsv. The first one should be uploaded to a Google Cloud
Storage bucket, and the latter one will be used in the post-processing step.

In case your input sequences contain modifications in `[modification]` format
instead of the expected `(modification)`, add the following parameter to the
command above:

```
  --clean_peptides=True
```

## Submit prediction job to CMLE

CMLE supports two modes of job submission: online and batch. You can read more
about it
[here](https://cloud.google.com/ml-engine/docs/tensorflow/online-vs-batch-prediction),
but briefly: the online mode is easier to set up, but can only process a limited
number of peptide inputs (up to ~300). The batch mode setup, on the other hand,
is more involved, but can easily process unlimited input sizes.

### Submit predictions in the online mode

Run the following command to submit a CMLE online job (input file size is
limited to 1.5 Mb, which corresponds to ~300 peptides - support for batch
predictions with unlimited input sizes is coming up soon):

```
gcloud ml-engine predict \
    --model deepmass_prism \
    --project deepmass-204419 \
    --format json \
    --json-instances "${DATA_DIR}/input.json" > "${DATA_DIR}/prediction.results"
```

Runtime will depend on the number of input examples, but it should generally be
completed within minutes. The output at this step should look like this:

```
{
  "predictions": [
    {
      "key": 0,
      "outputs": [
        [
          0.0,
          0.0
        ],
        ...
        [
          3219.9737692221674,
          0.0
        ]
      ]
    },
    {
      "key": 1,
      "outputs": [
        [
          0.0,
          0.0
        ],
        ...
        [
          1663.7626936434697,
          0.0
        ]
      ]
    }
  ]
}

```

### Submit predictions in the batch mode

To setup your system for batch predictions, do the folowing:

*   Make sure you have your Google Cloud Project set up
    ([link](https://cloud.google.com/)).
*   Add the following DeepMass service account to your Cloud project and grant
    it “Storage Admin” access role
    ([help](https://cloud.google.com/iam/docs/how-to)):

    `service-114490567449@cloud-ml.google.com.iam.gserviceaccount.com`

*   You will then need to copy the preprocessed inputs from step 1 into a Google
    Cloud Storage (GCS) bucket under your Google Cloud Project
    ([link](https://cloud.google.com/storage/docs/creating-buckets)).

Then go to console/terminal and configure the following gcloud settings:

```
PROJECT_NAME="project_name"  # Modify.
gcloud auth login
gcloud auth application-default login
gcloud config set account "${ACCOUNT_NAME}"  # ACCOUNT_NAME is email address you
                                             # used to set up the Cloud Projects
gcloud config set project "${PROJECT_NAME}"

```

Now you're ready to submit a CMLE batch job as follows:

```
CLOUD_DIR="gs://path-to-gcs-bucket"  # Modify.
INPUTS="${CLOUD_DIR}/input.json"  # Modify.
JOB_NAME="prediction_job_name"  # Modify.
REGION="us-central1"  # Modify.

gcloud ml-engine jobs submit prediction "${JOB_NAME}" \
    --model deepmass_prism \
    --input-paths "${INPUTS}" \
    --output-path "${CLOUD_DIR}" \
    --region "${REGION}" \
    --data-format text \
    --project deepmass-204419
```

This run will write several files (depending on the number of peptides in the
input table) into `"${CLOUD_DIR}"`, with the following filename pattern:
prediction.results* and prediction.errors_stats*. If the post-processing step
will be carried out locally, then you should transfer the outputs to a local
folder (for example, `"${DATA_DIR}"`).

**Privacy note:** Running the predictions in batch mode will log the email
address registered under user's Google Cloud Project, as well as the input and
output paths. If this is a concern, please run the prediction in the online
mode.

## Post-processing step

The post-processing step merges the input table and the model outputs, as
follows:

```
python postprocess.py \
    --metadata_file="${DATA_DIR}/metadata.tsv" \
    --input_data_pattern="${DATA_DIR}/prediction.results*" \
    --output_data_dir="${DATA_DIR}" \
    --batch_prediction=False
```

This step will write the final table into the `"${DATA_DIR}"`. The output table
will be identical to the input one, with two additional columns: `FragmentIons`
(a “;”-separated list of fragment ions), and `FragmentIntensities` (a
“;”-separated list of fragment ion peak intensities).

## Contact

Please post any questions or comments to our Google Group at:
DeepMass@google.com.
