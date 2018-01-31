from util import file_util


class CsvFile:

    def __init__(self, path):
        self.path = path
        self.values = dict()
        if file_util.file_exists(path):
            with open(path, 'r') as file:
                csv = file.read()
            column_names = csv[:csv.find('\n')].split(',')
            lines = csv[csv.find('\n') + 1:].splitlines()
            for line in lines:
                if line.strip() != '':
                    values = line.split(',')
                    row = dict()
                    row_key = values[0]
                    for i in range(1, len(column_names)):
                        row[column_names[i]] = values[i]
                    self.add_row(row_key, row)

    def add_row(self, row_key, values):
        if row_key not in self.values.keys():
            self.values[row_key] = dict()
        self.values[row_key].update(values)

    def save(self):
        column_names = set()
        for row_key in self.values.keys():
            row = self.values[row_key]
            for key in row.keys():
                column_names.add(key)
        column_names = sorted(column_names)
        csv = 'method_name'
        for column_name in column_names:
            csv += ',' + column_name
        csv += '\n'
        for row_key in self.values.keys():
            row = self.values[row_key]
            csv_row = row_key
            for column_name in column_names:
                if column_name in row:
                    csv_row += ',' + str(row[column_name])
                else:
                    csv_row += ','
            csv += csv_row + '\n'
        with open(self.path, 'w') as file:
            file.write(csv)
