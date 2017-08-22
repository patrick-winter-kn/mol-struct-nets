import os
from os import path


def get_filename(file_path, with_extension=True):
    name = path.basename(file_path)
    if not with_extension:
        name = name[:name.rfind('.')]
    return name


def file_exists(file_path):
    return path.exists(file_path)


def resolve_path(file_path):
    return path.abspath(file_path)


def resolve_subpath(folder_path, child_paths):
    if not isinstance(child_paths, list):
        child_paths = [child_paths]
    full_path = resolve_path(folder_path)
    for child_path in child_paths:
        full_path += path.sep + child_path
    return full_path


def make_folders(file_path):
    folder_path = path.dirname(resolve_path(file_path))
    if not path.exists(folder_path):
        os.makedirs(folder_path)
