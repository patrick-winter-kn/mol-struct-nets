import math
import shutil


class LogLevel:
    DEBUG = 1
    VERBOSE = 2
    INFO = 3
    WARNING = 4
    ERROR = 5


global_log_level = LogLevel.INFO


def log(string, log_level=LogLevel.INFO):
    if log_level >= global_log_level:
        if log_level == LogLevel.WARNING:
            print('\033[93mWARNING: ' + string + '\033[0m')
        elif log_level == LogLevel.ERROR:
            print('\033[91mERROR: ' + string + '\033[0m')
        else:
            print(string)


def do_print(log_level):
    return log_level >= global_log_level


def divider(character='_', nr_lines=1, log_level=LogLevel.INFO):
    if character == '_':
        log(('_' * column_width() + '\n') * nr_lines, log_level)
    else:
        log('\n' + (character * column_width() + '\n') * nr_lines, log_level)


def header(string, log_level=LogLevel.INFO):
    padding = ' ' * math.floor((column_width() - len(string)) / 2)
    log(padding + string, log_level)


def column_width():
    return shutil.get_terminal_size((100, 1)).columns
