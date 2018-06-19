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

This script reads data from a text file containing at least two columns:
peptide sequence and charge (which can be nones), and then outputs a JSON file
with inputs for mass spectrum prediction by DeepMass:Prism.

Example usage for the noloss model:
  DATA_DIR="./data"
  python preprocess.py \
    --input_data="${DATA_DIR}/test_mq.txt" \
    --output_data_dir="${DATA_DIR}" \
    --sequence_col="ModifiedSequence" \
    --charge_col="Charge" \
    --fragmentation_col="Fragmentation" \
    --analyzer_col="MassAnalyzer"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re
import sys

import numpy as np
import pandas as pd

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile


import utils


_FLOAT_META_SUFFIX = '_pos'
_PRECURSOR_METADATA_FEATURES = ['Charge' + _FLOAT_META_SUFFIX,
                                'Length' + _FLOAT_META_SUFFIX,
                                'MassAnalyzer_FTMS', 'Fragmentation_HCD',
                                'Fragmentation_CID', 'MassAnalyzer_ITMS']
_MOD_SEQUENCE = 'ModifiedSequence'
_PRECURSOR_CHARGE = 'PrecursorCharge'
_CHARGE = 'Charge'
_FRAGMENTATION = 'Fragmentation'
_MASS_ANALYZER = 'MassAnalyzer'
_LENGTH = 'Length'
_REQUIRED_COLUMNS = [_MOD_SEQUENCE, _CHARGE, _FRAGMENTATION, _MASS_ANALYZER]
_RESIDUE = re.compile(r'[A-Z](\((\w+)\))?')

_GROUP = 'group'
_GROUP_H = _GROUP + 'H'
_GROUP_OH = _GROUP + 'OH'
_GROUP_NH3 = _GROUP + 'NH3'
_GROUP_H2O = _GROUP + 'H2O'

_MOL_WEIGHTS = {
    'A': 71.03711,
    'C': 103.00919 + 57.02146,  # Add fixed CME modification to the Cys mass.
    'E': 129.04259,
    'D': 115.02694,
    'G': 57.02146,
    'F': 147.06841,
    'I': 113.08406,
    'H': 137.05891,
    'K': 128.09496,
    'M': 131.04049,
    'L': 113.08406,
    'N': 114.04293,
    'Q': 128.05858,
    'P': 97.05276,
    'S': 87.03203,
    'R': 156.10111,
    'T': 101.04768,
    'W': 186.07931,
    'V': 99.06841,
    'Y': 163.06333,
    'M(ox)': 147.035405,
    'groupCH3': 14.01565,
    'groupOH': 17.00274,
    'groupH': 1.007825,
    'groupH2O': 18.01057,
    'groupCH3CO': 42.01057,
    'groupO': 15.994915,
    'groupNH3': 17.02655}


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'input_data',
    '',
    'Input data filepath.')
flags.DEFINE_string(
    'output_data_dir',
    '',
    'Input data filepath.')
flags.DEFINE_bool(
    'clean_peptides',
    True,
    'True if peptide modifications are in [x] format.')
flags.DEFINE_string(
    'sequence_col',
    _MOD_SEQUENCE,
    'Modified sequence column name in the input file.')
flags.DEFINE_string(
    'charge_col',
    _CHARGE,
    'Charge column name in the input file.')
flags.DEFINE_string(
    'fragmentation_col',
    _FRAGMENTATION,
    'Fragmentation column name in the input file.')
flags.DEFINE_string(
    'analyzer_col',
    _MASS_ANALYZER,
    'Mass analyzer column name in the input file.')


def generate_json_inputs(data, encoding):
  """Generates inputs to-be stored into a JSON file.

  Args:
    data: A pandas dataframe with modified sequence and metadata features.
    encoding: A Pandas data frame with one-hot-encoded alphabet (alphabet
              characters are indices and one-hot encoding are columns).

  Yields:
    A dictionary with inputs, context, and key features.
  """

  max_length = data[_LENGTH].max()
  for _, row in data.iterrows():
    peptide = row[_MOD_SEQUENCE]
    input_features = []
    for residue in re.finditer(_RESIDUE, peptide):
      input_list = encoding[residue.group()]
      input_features.append(input_list)
    # Pad the sequence.
    input_features = np.array(input_features, dtype=np.float64)
    input_features = np.pad(
        input_features,
        ((0, max_length - input_features.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0)
    metas = []
    for meta in _PRECURSOR_METADATA_FEATURES:
      metas.append(np.array([row[meta]], dtype=np.float64).T)
    metas = np.array(metas, dtype=np.float64)
    yield {
        'inputs': input_features.tolist(),
        'context': metas.tolist(),
        'key': row['index']}


def check_inputs(data, alphabet):
  """Validate that the column names, amino-acid alphabet, and metadata.

  Args:
    data: A pandas data frame.
    alphabet: A list of amino-acid residues currently considered by the model.

  Raises:
    ValueError: In case the input table does not contain the required columns.
    ValueError: In case the peptide sequences contain novel amino-acid residues.
    ValueError: In case unknown fragmentation types are provided.
    ValueError: In case unknown mass analyzer types are provided.
  """
  columns = data.columns
  missing_columns = []
  for column in _REQUIRED_COLUMNS:
    if column not in columns:
      missing_columns.append(column)
  if missing_columns:
    raise ValueError(
        'The following columns are not in input table: %s.' %
        ', '.join(missing_columns))

  input_alphabet = set()
  for peptide in data[_MOD_SEQUENCE].values:
    input_alphabet.update([i.group() for i in re.finditer(_RESIDUE, peptide)])
  if input_alphabet.difference(set(alphabet)):
    raise ValueError(
        'Sequences contain these new residues not in model alphabet: %s' %
        ', '.join(list(input_alphabet.difference(set(alphabet)))))
  if set(data[_FRAGMENTATION].unique()).difference(set(['HCD', 'CID'])):
    raise ValueError(
        'Inputs contain invalid fragmentation type(s): %s' %
        ', '.join(list(set(data[_FRAGMENTATION].unique()).difference(
            set(['HCD', 'CID'])))))
  if set(data[_MASS_ANALYZER].unique()).difference(set(['FTMS', 'ITMS'])):
    raise ValueError(
        'Inputs contain invalid mass analyzer type(s): %s' %
        ', '.join(list(set(data[_MASS_ANALYZER].unique()).difference(
            set(['FTMS', 'ITMS'])))))


def preprocess_peptides(data, clean_peptides=True):
  """Preprocesses peptide sequences and turns them into TF sequence examples.

  The TSV file must contain a column named `_MOD_SEQUENCE`, and preferably also
  `_PRECURSOR_CHARGE`. The inputs will be turned into tf.sequence_examples and
  stored into a TFRecord format.

  Args:
    data: A pandas data frame.
    clean_peptides: A bool, True if modifications are in '[modification]' format
        (default True).

  Returns:
    A tuple with pandas data frame containing peptide and its meta data, and a
        set of peptide meta data features.
  """
  def _get_length(sequence):
    return len([aa.group() for aa in re.finditer(_RESIDUE, sequence)])

  if data.empty:  # Some of the input files can be empty, so exit.
    sys.exit()

  # Modify modification labels to match the input.
  if clean_peptides:
    data[_MOD_SEQUENCE] = utils.clean_peptides(data[_MOD_SEQUENCE])

  # Add metadata (peptide sequence, fragmentation, and mass analyzer type).
  data[_LENGTH] = data[_MOD_SEQUENCE].apply(_get_length)
  metadata, precursor_meta_features = utils.process_metadata(data)
  return metadata, precursor_meta_features


def main(unused_argv):

  # Get one-hot encoding.
  mol_weights = pd.Series(_MOL_WEIGHTS)
  alphabet = [k for k in mol_weights.keys() if not k.startswith(_GROUP)]
  alphabet = sorted(alphabet)
  one_hot_encoding = pd.get_dummies(alphabet).astype(int).to_dict(orient='list')

  with gfile.Open(FLAGS.input_data) as inputf:
    input_data = pd.read_csv(inputf, sep=',')
  input_data.rename(
      columns={FLAGS.sequence_col: _MOD_SEQUENCE,
               FLAGS.charge_col: _CHARGE,
               FLAGS.fragmentation_col: _FRAGMENTATION,
               FLAGS.analyzer_col: _MASS_ANALYZER},
      inplace=True)

  check_inputs(input_data, alphabet)

  metadata, _ = preprocess_peptides(input_data, FLAGS.clean_peptides)
  metadata = metadata.reset_index()

  
  # length.
  json_inputs = generate_json_inputs(metadata, one_hot_encoding)
  with gfile.Open(
      os.path.join(FLAGS.output_data_dir, 'input.json'), 'w') as outf:
    for json_input in json_inputs:
      outf.write(json.dumps(json_input) + '\n')
  with gfile.Open(
      os.path.join(FLAGS.output_data_dir, 'metadata.csv'), 'w') as outf:
    metadata.to_csv(outf, sep='\t')


if __name__ == '__main__':
  app.run(main)
