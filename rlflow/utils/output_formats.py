import sys
import datetime
import json
import os
import tempfile
import warnings
from typing import Dict, List, TextIO, Union, Any, Optional, Tuple

import numpy as np

class KVWriter(object):
    """
    Key Value writer
    """

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        """
        Write a dictionary to file

        :param key_values: (dict)
        :param key_excluded: (dict)
        :param step: (int)
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close owned resources
        """
        raise NotImplementedError


class SeqWriter(object):
    """
    sequence writer
    """

    def write_sequence(self, sequence: List):
        """
        write_sequence an array to file

        :param sequence: (list)
        """
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file: Union[str, TextIO]):
        """
        log to a file, in a human readable format

        :param filename_or_file: (str or File) the file to write the log to
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'write'), f'Expected file or str, got {filename_or_file}'
            self.file = filename_or_file
            self.own_file = False

    def write(self, key_values: Dict, key_excluded: Dict, step: int = 0) -> None:
        # Create strings for printing
        key2str = {}
        tag = None
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and 'stdout' in excluded:
                continue

            if isinstance(value, float):
                # Align left
                value_str = f'{value:<8.3g}'
            else:
                value_str = str(value)

            if key.find('/') > 0:  # Find tag and add it to the dict
                tag = key[:key.find('/') + 1]
                key2str[self._truncate(tag)] = ''
            # Remove tag from key
            if tag is not None and tag in key:
                key = str('   ' + key[len(tag):])

            key2str[self._truncate(key)] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn('Tried to write empty key-value dict')
            return
        else:
            key_width = max(map(len, key2str.keys()))
            val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (key_width + val_width + 7)
        lines = [dashes]
        for key, value in key2str.items():
            key_space = ' ' * (key_width - len(key))
            val_space = ' ' * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    @classmethod
    def _truncate(cls, string: str, max_length: int = 23) -> str:
        return string[:max_length - 3] + '...' if len(string) > max_length else string

    def write_sequence(self, sequence: List) -> None:
        sequence = list(sequence)
        for i, elem in enumerate(sequence):
            self.file.write(elem)
            if i < len(sequence) - 1:  # add space unless this is the last one
                self.file.write(' ')
        self.file.write('\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename: str):
        """
        log to a file, in the JSON format

        :param filename: (str) the file to write the log to
        """
        self.file = open(filename, 'wt')

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and 'json' in excluded:
                continue

            if hasattr(value, 'dtype'):
                if value.shape == () or len(value) == 1:
                    # if value is a dimensionless numpy array or of length 1, serialize as a float
                    key_values[key] = float(value)
                else:
                    # otherwise, a value is a numpy array, serialize as a list or nested lists
                    key_values[key] = value.tolist()
        self.file.write(json.dumps(key_values) + '\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """

        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename: str):
        """
        log to a file, in a CSV format

        :param filename: (str) the file to write the log to
        """

        self.file = open(filename, 'w+t')
        self.keys = []
        self.separator = ','

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        # Add our current row to the history

        extra_keys = key_values.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, key) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(key)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.separator * len(extra_keys))
                self.file.write('\n')
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            value = key_values.get(key)
            if value is not None:
                self.file.write(str(value))
        self.file.write('\n')
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    def __init__(self, folder: str):
        """
        Dumps key/value pairs into TensorBoard's numeric format.

        :param folder: (str) the folder to write the log to
        """

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            SummaryWriter = None
        assert SummaryWriter is not None, ("tensorboard is not installed, you can use "
                                           "pip install tensorboard to do so")
        self.writer = SummaryWriter(log_dir=folder)

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:

        for (key, value), (_, excluded) in zip(sorted(key_values.items()),
                                               sorted(key_excluded.items())):

            if excluded is not None and 'tensorboard' in excluded:
                continue

            if isinstance(value, np.ScalarType):
                self.writer.add_scalar(key, value, step)

            import torch as th
            if isinstance(value, th.Tensor):
                self.writer.add_histogram(key, value, step)

        # Flush the output to the file
        self.writer.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.writer:
            self.writer.close()
            self.writer = None


def make_output_format(_format: str, log_dir: str, log_suffix: str = '') -> KVWriter:
    """
    return a logger for the requested format

    :param _format: (str) the requested format to log to ('stdout', 'log', 'json' or 'csv' or 'tensorboard')
    :param log_dir: (str) the logging directory
    :param log_suffix: (str) the suffix for the log file
    :return: (KVWriter) the logger
    """
    os.makedirs(log_dir, exist_ok=True)
    if _format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif _format == 'log':
        return HumanOutputFormat(os.path.join(log_dir, f'log{log_suffix}.txt'))
    elif _format == 'json':
        return JSONOutputFormat(os.path.join(log_dir, f'progress{log_suffix}.json'))
    elif _format == 'csv':
        return CSVOutputFormat(os.path.join(log_dir, f'progress{log_suffix}.csv'))
    elif _format == 'tensorboard':
        return TensorBoardOutputFormat(log_dir)
    else:
        raise ValueError(f'Unknown format specified: {_format}')
