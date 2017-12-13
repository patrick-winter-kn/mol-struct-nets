import tkinter
from tkinter import ttk, messagebox
import sys


class ParameterEditor(ttk.Frame):

    def __init__(self, parent, save, parameter_list, parameter=None):
        self.top = tkinter.Toplevel(parent)
        self.top.title('Edit Parameter')
        ttk.Frame.__init__(self, self.top)
        self.save = save
        self.parameter = parameter
        self.parameter_list = parameter_list
        self.last_parameter = None
        self.init_parameter()
        self.add_widgets()
        self.pack(fill=tkinter.BOTH, expand=True)
        self.top.wait_visibility()
        self.top.grab_set()
        self.top.transient(parent)

    def init_parameter(self):
        if self.parameter is None:
            self.parameter = (self.parameter_list[0]['id'], None)

    def add_widgets(self):
        # Key
        key_frame = ttk.Frame(self)
        key_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        key_label = ttk.Label(key_frame, text='Parameter')
        key_label.pack(side=tkinter.LEFT)
        key_value = tkinter.StringVar()
        key_options = list()
        for param in self.parameter_list:
            key_options.append(param['name'])
            if param['id'] == self.parameter[0]:
                key_value.set(param['name'])
        key_option = ttk.OptionMenu(key_frame, key_value, None, *key_options)
        key_option.pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        key_value.trace('w', self.key_changed)
        # Text
        text_frame = ttk.Frame(self)
        text_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        text_label = ttk.Label(text_frame, text='Value')
        text_label.pack(side=tkinter.LEFT)
        text_value = tkinter.StringVar()
        text = ttk.Entry(text_frame, textvariable=text_value)
        text.pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        # Options
        options_frame = ttk.Frame(self)
        options_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        options_label = ttk.Label(options_frame, text='Value')
        options_label.pack(side=tkinter.LEFT)
        options_value = tkinter.StringVar()
        options = ttk.OptionMenu(options_frame, options_value)
        options.pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        # Binary
        binary_frame = ttk.Frame(self)
        binary_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        binary_label = ttk.Label(binary_frame, text='Value')
        binary_label.pack(side=tkinter.LEFT)
        binary_value = tkinter.BooleanVar()
        binary_true = ttk.Radiobutton(binary_frame, text='True', variable=binary_value, value=True)
        binary_false = ttk.Radiobutton(binary_frame, text='False', variable=binary_value, value=False)
        binary_true.pack(side=tkinter.LEFT)
        binary_false.pack(side=tkinter.LEFT)
        # Spinner
        spinner_frame = ttk.Frame(self)
        spinner_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        spinner_label = ttk.Label(spinner_frame, text='Value')
        spinner_label.pack(side=tkinter.LEFT)
        spinner_value = tkinter.StringVar()
        spinner = tkinter.Spinbox(spinner_frame, textvariable=spinner_value)
        spinner.pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        # Save / discard
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        save_changes = ttk.Button(bottom_frame, text='Save', command=self.save_changes)
        save_changes.pack(side=tkinter.RIGHT)
        discard_changes = ttk.Button(bottom_frame, text='Discard', command=self.discard_changes)
        discard_changes.pack(side=tkinter.RIGHT)
        # Parameter info
        info_frame = ttk.Frame(self)
        info_frame.pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)
        info = tkinter.Text(info_frame, wrap=tkinter.WORD)
        # TODO way to big
        # TODO overlaps previous element
        info.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        self.key_value = key_value
        self.text_frame = text_frame
        self.options_frame = options_frame
        self.binary_frame = binary_frame
        self.spinner_frame = spinner_frame
        self.options = options
        self.options_value = options_value
        self.spinner = spinner
        self.text_value = text_value
        self.binary_value = binary_value
        self.spinner_value = spinner_value
        self.info = info
        self.master.minsize(width=300, height=150)
        self.update_value_fields()

    def save_changes(self):
        valid, message = self.update_parameter()
        if valid:
            self.save(self.parameter)
            self.top.destroy()
        else:
            messagebox.showerror('Invalid Parameter', message)

    def discard_changes(self):
        self.top.destroy()

    def key_changed(self, *args):
        self.update_value_fields()

    def update_value_fields(self):
        param = self.get_selected_parameter()
        if self.last_parameter != param['id']:
            self.last_parameter = param['id']
            self.text_frame.pack_forget()
            self.options_frame.pack_forget()
            self.binary_frame.pack_forget()
            self.spinner_frame.pack_forget()
            self.info.config(state=tkinter.NORMAL)
            self.info.delete(1.0, tkinter.END)
            if 'description' in param:
                self.info.insert(tkinter.INSERT, param['description'])
            self.info.config(state=tkinter.DISABLED)
            if param['type'] == str:
                if 'options' in param:
                    self.options_frame.pack(side=tkinter.TOP, fill=tkinter.X)
                    self.options['menu'].delete(0, tkinter.END)
                    for choice in param['options']:
                        self.options['menu'].add_command(label=choice, command=tkinter._setit(self.options_value, choice))
                    if self.parameter[1] is not None:
                        self.options_value.set(self.parameter[1])
                    elif 'default' in param:
                        self.options_value.set(param['default'])
                    else:
                        self.options_value.set(param['options'][0])
                else:
                    self.text_frame.pack(side=tkinter.TOP, fill=tkinter.X)
                    if self.parameter[1] is not None:
                        self.text_value.set(self.parameter[1])
                    elif 'default' in param and param['default'] is not None:
                        self.text_value.set(param['default'])
                    else:
                        self.text_value.set('')
            elif param['type'] == bool:
                self.binary_frame.pack(side=tkinter.TOP, fill=tkinter.X)
                if self.parameter[1] is not None:
                    self.binary_value.set(self.parameter[1])
                elif 'default' in param:
                    self.binary_value.set(param['default'])
                else:
                    self.binary_value.set(True)
            elif param['type'] == int or param['type'] == float:
                self.spinner_frame.pack(side=tkinter.TOP, fill=tkinter.X)
                if param['type'] == int:
                    minimum = -sys.maxsize - 1
                    maximum = sys.maxsize
                else:
                    minimum = float('-inf')
                    maximum = float('inf')
                if 'min' in param:
                    minimum = param['min']
                if 'max' in param:
                    maximum = param['max']
                self.spinner.config(from_=minimum, to=maximum)
                if self.parameter[1] is not None:
                    self.spinner_value.set(self.parameter[1])
                elif 'default' in param and param['default'] is not None:
                    self.spinner_value.set(str(param['default']))
                else:
                    self.spinner_value.set(str(max(minimum, 1)))

    def get_selected_key(self):
        return self.key_value.get()

    def get_selected_parameter(self):
        key = self.get_selected_key()
        for param in self.parameter_list:
            if param['name'] == key:
                return param

    def get_value(self):
        param = self.get_selected_parameter()
        if param['type'] == str:
            if 'options' in param:
                return self.options_value.get()
            else:
                return self.text_value.get()
        elif param['type'] == bool:
            return self.binary_value.get()
        elif param['type'] == int:
            try:
                return int(self.spinner_value.get())
            except:
                return None
        elif param['type'] == float:
            try:
                return float(self.spinner_value.get())
            except:
                return None

    def update_parameter(self):
        param = self.get_selected_parameter()
        value = self.get_value()
        valid = True
        message = None
        if value is None:
            valid = False
            message = 'The value is not set'
        elif 'min' in param and value < param['min']:
            valid = False
            message = str(value) + ' is smaller than the allowed minimum of ' + str(param['min'])
        elif 'max' in param and value > param['max']:
            valid = False
            message = str(value) + ' is bigger than the allowed maximum of ' + str(param['max'])
        elif 'options' in param and value not in param['options']:
            valid = False
            message = value + ' is not among the allowed options'
        if valid:
            self.parameter = (param['id'], value)
        return valid, message
