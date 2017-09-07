class LogLevel:

    DEBUG = 1
    VERBOSE = 2
    INFO = 3
    WARNING = 4
    ERROR = 5


global_log_level = LogLevel.INFO


def log(string, log_level=LogLevel.INFO):
    if log_level >= global_log_level:
        print(string)


def do_print(log_level):
    return log_level >= global_log_level
