from util import file_util

def get_experiement_folder(root_path):
    return file_util.resolve_subpath(root_path, 'experiments')
