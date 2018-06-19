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
r"""This module contains utility functions for DeepMass Prism."""

import re

import numpy as np
import pandas as pd


_RE_AA = re.compile(r'[A-Z](\[(\w+)\])?')
_PERCENTILES = pd.DataFrame([
    [7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10,
     10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12,
     12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15,
     15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18,
     18, 19, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 23, 23, 24, 25, 26, 27,
     28, 30],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
     7]], index=['Length', 'Charge']).T
_ALPHABETS = {'Fragmentation': ['HCD', 'CID'], 'MassAnalyzer': ['ITMS', 'FTMS']}


def _range_normalize(vector, lower=None, upper=None):
  """Normalizes values to a range (eg [0, 1]).

  Used to range normalize a subset of a larger set of data (where the total
  range exceeds the [min, max] of the values in vector).

  Note that this can return values outside of [0, 1] if lower or upper are
  within the range of vector. This throws no error as it can be useful in some
  cases.

  Args:
    vector: Numpy array-like with values to normalize.
    lower: The minimum value in the complete dataset. None to use vector.min()
    upper: The maximum value in the complete dataset. None to use vector.max()

  Returns:
    An object of type vector with values correspondingly normalized.
  """

  if lower is None:
    lower = vector.min()
  if upper is None:
    upper = vector.max()

  return (vector - lower) / (upper - lower)


def clean_peptides(peptide_list):
  """Cleans peptide sequeces by replacing [x] modification notation for (x).

  Args:
    peptide_list: A list of strings.

  Returns:
    A python list with input peptides but with their modifications in (x)
        format.
  """
  clean_peptide_list = []
  for peptide in peptide_list:
    clean_peptide = []
    for residue in re.finditer(_RE_AA, peptide):
      residue = residue.group()
      if '[' in residue:
        if residue == 'C[Carbamidomethyl]':
          residue = 'C'
        elif residue == 'M[Oxidation]':
          residue = 'M(ox)'
        else:
          residue = residue[0]
      clean_peptide.append(residue)
    clean_peptide_list.append(''.join(clean_peptide))
  return clean_peptide_list


def generate_encoding(alphabet):
  """Generates one-hot encoding for a given alphabet.

  Args:
    alphabet: A Python set, list, or dict with alphabet characters.

  Returns:
    A Pandas data frame with one-hot-encoded alphabet.
  """
  return pd.DataFrame(
      np.eye(len(alphabet)),
      index=alphabet, columns=alphabet, dtype=int).to_dict(orient='list')


def process_metadata(metadata):
  """Further processes precursor metadata features.

  Used eg to clip by percentiles, log-scale or unit-range-normalize feature
  columns; also one-hot encodes string features.

  Args:
    metadata: a dataframe as returned by get_meta_data()

  Raises:
    ValueError: If a precursor feature is not found in the table or if a process
    is not one of {range,log,low N,high N}.

  Returns:
    A copy of metadata with processed features appended;
    A set of the names of processed precursor features.
  """

  feat_input = [
      ['Charge', 'Charge_pos', 'int', 'range 0 max'],
      ['Length', 'Length_pos', 'int', 'range 0 max'],
      ['Fragmentation', 'Fragmentation_OH', 'string'],
      ['MassAnalyzer', 'MassAnalyzer_OH', 'string']]

  precursor_features = set()
  for feat_line in feat_input:
    feature = feat_line.pop(0)
    # Check for bad precursor requests:
    if feature not in metadata.columns:
      raise ValueError('{} not found in metadata table'.format(feature))
    feat_dest = feat_line.pop(0)
    feat_type = feat_line.pop(0)
    if feat_type == 'string':
      # Generate one hot echidna:
      encoding = generate_encoding(_ALPHABETS[feature])
      df = pd.DataFrame(
          [encoding[v] for v in metadata[feature]],
          columns=[feature + '_' + v for v in _ALPHABETS[feature]])
      metadata = pd.concat([metadata, df], 1)
      precursor_features = precursor_features.union(df.columns.values)
    elif feat_type in ['int', 'float']:
      precursor_features.add(feat_dest)
      feat_percentiles = _PERCENTILES[feature]
      metadata[feat_dest] = metadata[feature]
      p_lo = feat_percentiles.iloc[0]
      p_hi = feat_percentiles.iloc[100]
      for process in feat_line:
        if process.startswith('range'):
          process = process.split()
          if process[1] != 'min':
            p_lo = int(process[1])
          if process[2] != 'max':
            p_hi = int(process[2])
          metadata[feat_dest] = _range_normalize(metadata[feat_dest],
                                                 p_lo, p_hi)
        elif process.startswith('low'):
          p_lo = _PERCENTILES[feature].iloc[int(process.split()[1])]
          metadata[feat_dest] = metadata[feat_dest].clip_lower(p_lo)
        elif process.startswith('high'):
          p_hi = _PERCENTILES[feature].iloc[int(process.split()[1])]
          metadata[feat_dest] = metadata[feat_dest].clip_upper(p_hi)
        elif process == 'log':
          metadata[feat_dest] = np.log(metadata[feat_dest])
          feat_percentiles = np.log(feat_percentiles)
          p_lo = np.log(p_lo)
          p_hi = np.log(p_hi)
        else:
          raise ValueError('Bad precursor feature process {}'.format(process))

  return metadata, precursor_features
