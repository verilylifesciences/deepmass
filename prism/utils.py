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

import itertools
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

_RESIDUE = re.compile(r'[A-Z](\((\w+)\))?')

_GROUP = 'group'
_GROUP_H = _GROUP + 'H'
_GROUP_HP = _GROUP + 'H+'
_GROUP_OH = _GROUP + 'OH'
_GROUP_NH3 = _GROUP + 'NH3'
_GROUP_H2O = _GROUP + 'H2O'

MOL_WEIGHTS = {
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
    'groupH+': 1.007276,
    'groupH2O': 18.01057,
    'groupCH3CO': 42.01057,
    'groupO': 15.994915,
    'groupNH3': 17.02655}

_POSITION_COL = 'FragmentNumber'
_RESIDUE_COL = 'residue'
_ION_COL = 'FragmentType'
_ABUNDANCE_COL = 'RelativeIntensity'
_MZ_THEORY = 'FragmentMz'


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
        if residue in ['C[Carbamidomethyl]', 'C[+57]']:
          residue = 'C'
        elif residue in ['M[Oxidation]', 'M[+16]']:
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


def calculate_yb_series(sequence, molecular_weights, neutral_losses=False):
  """Calculates theoretical MS2 spectrum of y- and b-ion fragments.

  MS2 spectrum is a result of the peptide fragmentation. By far the most common
  fragmentations occur at YB sites in the peptide backbone chain. This function
  calculates a theoretical YB fragmentation patterns for +1 and +2 charged
  fragment ions, with an optional ions with neutral loss of water and ammonium
  molecules. The accuracy of this function was tested against this server:
  http://db.systemsbiology.net:8080/proteomicsToolkit

  Args:
    sequence: A string denoting peptide sequence (use 1-letter AA code, defined
              in molecular_weights input).
    molecular_weights: A Pandas series with weights of residues and neutral
                       groups.
    neutral_losses: True if NH3/H2O losses should be included, False otherwise.

  Returns:
    An array with theoretical y/b mz values.

  Raises:
    ValueError: In case molecular_weights does not contain one or more of
                required neutral groups (ie, 'groupH', 'groupOH', 'groupNH3',
                'groupH2O').
  """

  # Get peptide alphabet first.
  peptide = [aa.group() for aa in re.finditer(_RESIDUE, sequence)]

  # Check molecular_weights dictionary.
  required_groups = set([_GROUP_H, _GROUP_OH, _GROUP_NH3, _GROUP_H2O])
  if required_groups.difference(molecular_weights.keys()):
    raise ValueError('molecular_weights does not contain all required groups')

  peaks = pd.DataFrame()
  peaks[_RESIDUE_COL] = peptide
  peaks[_POSITION_COL] = np.arange(len(peptide)) + 1
  # Get theoretical spectrum for single-charged b- and y-ions.
  peaks['b'] = _calculate_b_series(peptide, molecular_weights)
  peaks['y'] = _calculate_y_series(peptide, molecular_weights)

  # Get theoretical spectra for NH3- and H2O-loss ions.
  if neutral_losses:
    for ion, grp in itertools.product(['b', 'y'], [_GROUP_NH3, _GROUP_H2O]):
      peaks['{}-{}'.format(ion, grp.replace(_GROUP, ''))] = (
          peaks[ion] - molecular_weights[grp])

  # Reverse order of y ions, as they are generated backwards from C-terminus.
  for name, col in peaks.filter(like='y', axis=1).iteritems():
    peaks[name] = col[::-1].reset_index()[name]

  peaks = pd.melt(peaks,
                  id_vars=[_POSITION_COL, _RESIDUE_COL],
                  var_name=_ION_COL,
                  value_name=_MZ_THEORY)
  return peaks


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


def _calculate_b_series(peptide, molecular_weights):
  """Calculates theoretical MS2 B-ion series for charge 1 fragments.

  For a peptide ACDEF the B-ion series would be: A, AC, ACD, ACDE, and ACDEF

  Args:
    peptide: A list of amino-acid resides (use 1-letter AA code, as defined
              in molecular_weights input).
    molecular_weights: A Pandas series with weights of residues and neutral
                       groups.

  Returns:
    A python list of M/Z values representing the B-ion series.
  """
  b_series = np.cumsum(
      ([molecular_weights[_GROUP_HP]] +  # Add H to N-terminus.
       [molecular_weights[i] for i in peptide]))
  return b_series[1:]  # Remove the H peak.


def _calculate_y_series(peptide, molecular_weights):
  """Calculates theoretical MS2 Y-ion series for charge 1 fragments.

  For a peptide ACDEF the Y-ion series would be: A, AC, ACD, ACDE, and ACDEF, so
  similar as in B-series, but with an additional H3O+ ion, and reversed order.

  Args:
    peptide: A list of amino-acid resides (use 1-letter AA code, as defined
              in molecular_weights input).
    molecular_weights: A Pandas series with weights of residues and neutral
                       groups.

  Returns:
    A python list of M/Z values representing the B-ion series.
  """
  y_series = np.cumsum(
      ([molecular_weights[_GROUP_OH] +  # Add OH group to C-terminus.
        molecular_weights[_GROUP_H]  + molecular_weights[_GROUP_HP]] +
       [molecular_weights[i] for i in reversed(peptide)]))
  return np.flipud(y_series[1:])  # Remove the H peak and turn around.

