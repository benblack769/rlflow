import sys
import datetime
import json
import os
import tempfile
import warnings
from collections import defaultdict
from typing import Dict, List, TextIO, Union, Any, Optional, Tuple

import numpy as np

from .output_formats import make_output_format, KVWriter

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


# ================================================================
# Backend
# ================================================================

class Logger(object):
    def __init__(self, folder: Optional[str], output_formats: List[KVWriter]):
        """
        the logger class

        :param folder: (str) the logging location
        :param output_formats: ([str]) the list of output format
        """
        self.name_to_value = defaultdict(float)  # values this iteration
        self.name_to_count = defaultdict(int)
        self.name_to_excluded = defaultdict(str)
        self.level = INFO
        self.dir = folder
        self.output_formats = output_formats

    # Logging API, forwarded
    # ----------------------------------------
    def record(self, key: str, value: Any,
               exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: (Any) save to log this key
        :param value: (Any) save to log this value
        :param exclude: (str or tuple) outputs to be excluded
        """
        self.name_to_value[key] = value
        self.name_to_excluded[key] = exclude

    def record_mean(self, key: str, value: Any,
                    exclude: Optional[Union[str, Tuple[str, ...]]] = None) -> None:
        """
        The same as record(), but if called many times, values averaged.

        :param key: (Any) save to log this key
        :param value: (Number) save to log this value
        :param exclude: (str or tuple) outputs to be excluded
        """
        if value is None:
            self.name_to_value[key] = None
            return
        old_val, count = self.name_to_value[key], self.name_to_count[key]
        self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
        self.name_to_count[key] = count + 1
        self.name_to_excluded[key] = exclude

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        if self.level == DISABLED:
            return
        for _format in self.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(self.name_to_value, self.name_to_excluded, step)

        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    def log(self, *args, level: int = INFO) -> None:
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).

        level: int. (see logger.py docs) If the global logger level is higher than
                    the level argument here, don't print to stdout.

        :param args: (list) log the arguments
        :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level: int) -> None:
        """
        Set logging threshold on current logger.

        :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        self.level = level

    def get_dir(self) -> str:
        """
        Get directory that log files are being written to.
        will be None if there is no output directory (i.e., if you didn't call start)

        :return: (str) the logging directory
        """
        return self.dir

    def close(self) -> None:
        """
        closes the file
        """
        for _format in self.output_formats:
            _format.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args) -> None:
        """
        log to the requested format outputs

        :param args: (list) the arguments to log
        """
        for _format in self.output_formats:
            if isinstance(_format, SeqWriter):
                _format.write_sequence(map(str, args))


def make_logger(folder: str, format_strings: Optional[List[str]] = None) -> None:
    """
    configure the current logger

    :param folder: (str) the save location
    :param format_strings: (Optional[List[str]]) the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    """
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ''
    if format_strings is None:
        format_strings = 'stdout,log,csv'.split(',')

    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]

    print(f'Creating logger to {folder}')
    return Logger(folder=folder, output_formats=output_formats)
