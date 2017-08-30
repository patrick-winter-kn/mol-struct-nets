class LogLevel:

    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


global_log_level = LogLevel.DEBUG


def log(string, log_level=LogLevel.INFO):
    if log_level >= global_log_level:
        print(string)


def do_print(log_level):
    return log_level >= global_log_level
