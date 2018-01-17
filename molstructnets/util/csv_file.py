from util import file_util


class CsvFile:

    def __init__(self, path):
        self.path = path
        self.values = list()
        if file_util.file_exists(path):
            with open(path, 'r') as file:
                csv = file.read()
            column_names = csv[:csv.find('\n')].split(',')
            lines = csv[csv.find('\n') + 1:].splitlines()
            for line in lines:
                if line.strip() != '':
                    values = line.split(',')
                    row = dict()
                    for i in range(len(column_names)):
                        row[column_names[i]] = values[i]
                    self.values.append(row)

    def add_row(self, row):
        self.values.append(row)

    def save(self):
        column_names = set()
        for row in self.values:
            for key in row.keys():
                column_names.add(key)
        column_names = sorted(column_names)
        csv = ''
        for column_name in column_names:
            csv += column_name + ','
        csv = csv[:-1] + '\n'
        for row in self.values:
            csv_row = ''
            for column_name in column_names:
                if column_name in row:
                    csv_row += str(row[column_name]) + ','
                else:
                    csv_row += ','
            csv += csv_row[:-1] + '\n'
        with open(self.path, 'w') as file:
            file.write(csv)
