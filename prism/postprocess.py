#
# Copyright 2017 Verily Life Sciences LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

r"""Infers YB spectra and outputs TSV spectral library.

This script post-processes DeepMass:Prism outputs. It requires two inputs:
output table from the preprocess.py run, and json files from the Cloud ML
inference.

Example usage for the noloss model:
  DATA_DIR="./data"
  python postprocess.py \
    --metadata_file="${DATA_DIR}/..." \
    --input_data_pattern="${DATA_DIR}/prediction.results*" \
    --output_data_dir="${DATA_DIR}" \
    --label_dim=2 \
    --neutral_losses=False \
    --batch_prediction=True
"""

import json
import os

import numpy as np
import pandas as pd
from tensorflow.compat.v1 import app
from tensorflow.compat.v1 import flags
from tensorflow.compat.v1 import gfile

import utils




_FLOAT_META_SUFFIX = '_pos'
_PRECURSOR_METADATA_FEATURES = ['Charge' + _FLOAT_META_SUFFIX,
                                'Length' + _FLOAT_META_SUFFIX,
                                'MassAnalyzer_FTMS', 'Fragmentation_HCD',
                                'Fragmentation_CID', 'MassAnalyzer_ITMS']
_ION_COLUMNS = ['b', 'b-H2O', 'b-NH3', 'y', 'y-H2O', 'y-NH3']
_MOD_SEQUENCE = 'ModifiedSequence'
_PRECURSOR_CHARGE = 'PrecursorCharge'
_CHARGE = 'Charge'
_FRAGMENTATION = 'Fragmentation'
_MASS_ANALYZER = 'MassAnalyzer'
_LENGTH = 'Length'

_POSITION_COL = 'FragmentNumber'
_ION_COL = 'FragmentType'
_ABUNDANCE_COL = 'RelativeIntensity'
_LOSS_TYPE = 'FragmentLossType'
_FRAG_CHARGE = 'FragmentCharge'

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'metadata_file',
    None,
    'Path to a TSV file with metadata.')
flags.DEFINE_string(
    'input_data_pattern',
    None,
    'Input data filename pattern.')
flags.DEFINE_enum(
    'label_dim',
    '2',
    ['2', '6'],
    'Number of features in the output/label time step')
flags.DEFINE_string(
    'output_data_dir',
    None,
    'Directory with prediction outputs.')
flags.DEFINE_bool(
    'neutral_losses',
    False,
    'True if H2O and NH3 losses are modeled.')
flags.DEFINE_bool(
    'batch_prediction',
    True,
    'True if batch prediction instead of online was used to generate outputs.')
flags.DEFINE_string(
    'add_input_data_pattern',
    None,
    ('Input data filename pattern for additional features to-be included to '
     'the final outptu. These inputs should be formatted in the same way as'
     'the model outputs - ie, JSON format with "key" and "output" values,'
     'where the key is an integer and output is a list of feature values.'))
flags.DEFINE_list(
    'add_feature_names',
    None,
    'A comma-separated list of additional feature names.')


def reformat_outputs(row, label_dim, neutral_losses):
  """Reformats output from the spectral model into a TSV shape.

  Args:
    row: A pandas series.
    label_dim: A dimensionality of output time (ion type) point.
    neutral_losses: True if NH3/H2O losses should be included, False otherwise.

  Raises:
    ValueError: label_dim is not 2 or 6.

  Returns:
    A pandas series with predicted intensities and ion types added to the input.
  """
  pep_len = row[_LENGTH]
  if label_dim == 2:
    pred_labels = pd.DataFrame(
        np.array(row['outputs'])[:pep_len],
        columns=['y', 'b'])
  elif label_dim == 6:
    pred_labels = pd.DataFrame(
        np.array(row['outputs'])[:pep_len],
        columns=_ION_COLUMNS)
  else:
    raise ValueError('label_dim is not 2 or 6')

  # Reverse order of y ions.
  for name, col in pred_labels.filter(like='y', axis=1).iteritems():
    pred_labels[name] = col[::-1].reset_index()[name]
  pred_labels = pd.melt(pred_labels.reset_index(), id_vars='index',
                        var_name=_ION_COL, value_name=_ABUNDANCE_COL)
  # Ion type counts should be 1-based not 0-based.
  pred_labels[_POSITION_COL] = pred_labels['index'] + 1
  pred_labels[_FRAG_CHARGE] = 1
  pred_labels.drop('index', axis=1, inplace=True)

  # Split ion type column.
  if neutral_losses:
    (pred_labels[_ION_COL],
     pred_labels[_LOSS_TYPE]) = pred_labels[_ION_COL].str.split('-', 1).str
    pred_labels = pred_labels.fillna('noloss')
  else:
    pred_labels[_LOSS_TYPE] = 'noloss'
    pred_labels[_ION_COL] = pred_labels[_ION_COL]

  # Add fragment m/z values. Note, only charge +1 are considered for now.
  pred_labels = pred_labels.merge(
      utils.calculate_yb_series(row[_MOD_SEQUENCE], utils.MOL_WEIGHTS),
      how='left',
      on=[_ION_COL, _POSITION_COL])

  ions = (pred_labels[_ION_COL] + pred_labels[_POSITION_COL].map(str) +
          '_charge' + pred_labels[_FRAG_CHARGE].map(str) + '-' +
          pred_labels[_LOSS_TYPE])

  precursor_mz = (
      (pred_labels['FragmentMz'].max() + ((row[_CHARGE]) - 1)
       * utils.MOL_WEIGHTS['groupH+'])
      / row[_CHARGE])

  ions = ';'.join(ions.values)
  intensities = ';'.join(map(str, pred_labels[_ABUNDANCE_COL]))
  fragment_mzs = ';'.join(map(str, pred_labels['FragmentMz']))
  row = row.append(
      pd.Series({'FragmentIons': ions,
                 'FragmentIntensities': intensities,
                 'FragmentMZs': fragment_mzs,
                 'PrecursorMZ': precursor_mz}))
  row = row.drop(
      ['outputs', 'Unnamed: 0', 'index'] + _PRECURSOR_METADATA_FEATURES)
  return row


def main(unused_argv):

  metadata = pd.read_csv(gfile.Open(FLAGS.metadata_file), sep='\t')

  # Read DeepMass:Prism outputs and merge with metadata.
  outputs = []
  for filen in gfile.Glob(FLAGS.input_data_pattern):
    with gfile.Open(filen) as infile:
      if FLAGS.batch_prediction:
        out_df = pd.read_json(infile, lines=True)
      else:
        out_df = json.load(infile)
        out_df = pd.DataFrame(out_df['predictions'])
      out_df = out_df.merge(
          metadata, left_on='key', right_on='index', how='left')
      outputs.append(out_df)
  outputs = pd.concat(outputs)
  outputs = outputs.apply(
      reformat_outputs,
      args=(int(FLAGS.label_dim), FLAGS.neutral_losses),
      axis=1)

  # Read additional features.
  if FLAGS.add_feature_names is not None:
    outputs_drip = []
    for filen in gfile.Glob(FLAGS.add_input_data_pattern):
      with gfile.Open(filen) as infile:
        if FLAGS.batch_prediction:
          out_df = pd.read_json(infile, lines=True)
          out_df = pd.DataFrame(
              out_df['outputs'].tolist(),
              columns=FLAGS.add_feature_names,
              index=out_df['key'])
        else:
          pass  
        outputs_drip.append(out_df)
    outputs_drip = pd.concat(outputs_drip)
    outputs = outputs.merge(
        outputs_drip, how='left', left_on='key', right_index=True)

  # Write to a file.
  with gfile.Open(
      os.path.join(FLAGS.output_data_dir, 'outputs.tsv'), 'w') as outf:
    outputs.to_csv(outf, sep='\t', index=False)


if __name__ == '__main__':
  app.run(main)
