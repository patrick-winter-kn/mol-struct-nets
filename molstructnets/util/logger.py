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
