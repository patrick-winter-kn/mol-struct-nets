import os
import shutil
import tempfile
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


def resolve_subpath(folder_path, *child_paths):
    full_path = resolve_path(folder_path)
    for child_path in child_paths:
        full_path += path.sep + child_path
    return full_path


def make_folders(file_path, including_this=False):
    file_path = resolve_path(file_path)
    if including_this:
        file_path += path.sep
    folder_path = path.dirname(file_path)
    if not path.exists(folder_path):
        os.makedirs(folder_path)


def list_files(folder_path):
    return os.listdir(folder_path)


def get_parent(file_path):
    return path.dirname(resolve_path(file_path))


def get_temporary_file_path(prefix=None):
    file_path = tempfile.mkstemp(prefix=prefix)[1]
    os.remove(file_path)
    return file_path


def move_file(source, destination, safe=True):
    make_folders(destination)
    remove_file(destination)
    if safe:
        shutil.copy(source, destination)
        remove_file(source)
    else:
        shutil.move(source, destination)


def is_folder(file_path):
    return path.isdir(file_path)


def copy_file(source, destination):
    make_folders(destination)
    remove_file(destination)
    shutil.copy(source, destination)


def remove_file(file_path):
    if file_exists(file_path):
        if path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)


def read_file(file_path):
    file_path = resolve_path(file_path)
    if not file_exists(file_path):
        return None
    with open(file_path, 'r') as file:
        string = file.read()
    return string
