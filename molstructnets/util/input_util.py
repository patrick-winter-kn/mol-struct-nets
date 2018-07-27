import re

from util import misc


def read_bool(text, optional=False, description=None):
    optional_text = ''
    if optional:
        optional_text = 'empty or '
    if description is None:
        description = 'No description available.'
    while True:
        value = input(text)
        value = value.lower()
        if value == '' and optional:
            return None
        elif value in ['y', 'yes', 'true']:
            return True
        elif value in ['n', 'no', 'false']:
            return False
        elif value == '?':
            print(description)
        else:
            print('Input is invalid. Must be ' + optional_text + 'one of (y|n).')


def read_int(text, optional=False, description=None, min_=None, max_=None):
    return read_number('integer', int, text, optional, description, min_, max_)


def read_float(text, optional=False, description=None, min_=None, max_=None):
    return read_number('float', float, text, optional, description, min_, max_)


def read_number(number_name, number_type, text, optional=False, description=None, min_=None, max_=None):
    optional_text = ''
    if optional:
        optional_text = 'empty or '
    if description is None:
        description = 'No description available.'
    range_text = ''
    if min_ is not None and max_ is None:
        range_text = ' (' + str(min_) + '<=x)'
    elif min_ is None and max_ is not None:
        range_text = ' (x<=' + str(max_) + ')'
    elif min_ is not None and max_ is not None:
        range_text = ' (' + str(min_) + '<=x<=' + str(max_) + ')'
    while True:
        value = input(text)
        if value == '' and optional:
            return None
        elif value == '?':
            print(description)
        else:
            try:
                value = number_type(value)
                if misc.in_range(value, min_, max_):
                    return value
            except ValueError:
                pass
            print('Input is invalid. Must be ' + optional_text + number_name + range_text + '.')


def read_string(text, optional=False, description=None, regex=None):
    optional_text = ''
    if optional:
        optional_text = 'empty or '
    if description is None:
        description = 'No description available.'
    regex_text = ''
    if regex is not None:
        regex_text = ' that matches regex \'' + regex + '\''
        regex = re.compile(regex)
    while True:
        value = input(text)
        if value == '' and optional:
            return None
        elif value == '?':
            print(description)
        elif value != '' and (regex is None or regex.match(value)):
            return value
        else:
            print('Input is invalid. Must be ' + optional_text + 'string' + regex_text + '.')
