from util import file_util


class Status:

    queued = 'queued'
    failed = 'failed'
    success = 'success'


class ExecutionResults:

    def __init__(self, path, number_experiments):
        self.path = file_util.resolve_path(path)
        self.status = list()
        if file_util.file_exists(self.path):
            with open(self.path, 'r') as file:
                csv = file.read()
            lines = csv.splitlines()
            for i in range(number_experiments):
                if i < len(lines):
                    self.status.append(lines[i])
                else:
                    self.status.append(Status.queued)
        else:
            for i in range(number_experiments):
                self.status.append(Status.queued)

    def save(self):
        with open(self.path, 'w') as file:
            for status in self.status:
                file.write(status + '\n')

    def size(self):
        return len(self.status)

    def get_status(self, index):
        return self.status[index]

    def set_status(self, index, status):
        self.status[index] = status
