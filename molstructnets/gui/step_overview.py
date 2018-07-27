import tkinter
from tkinter import messagebox
from tkinter import ttk

from gui import parameter_editor
from steps import steps_repository


class StepOverview(ttk.Frame):

    def __init__(self, parent, save, step=None, index=None):
        self.top = tkinter.Toplevel(parent)
        self.top.title('Edit Step')
        ttk.Frame.__init__(self, self.top)
        self.save = save
        self.step = step
        self.index = index
        self.init_step()
        self.type_name_to_id = dict()
        for step_repo in steps_repository.instance.get_steps():
            self.type_name_to_id[step_repo.get_name()] = step_repo.get_id()
        self.add_widgets()
        self.pack(fill=tkinter.BOTH, expand=True)
        self.top.wait_visibility()
        self.top.grab_set()
        self.top.transient(parent)

    def init_step(self):
        if self.step is None:
            step_repo = steps_repository.instance.get_steps()[0]
            type_ = step_repo.get_id()
            id_ = step_repo.get_implementations()[0].get_id()
            self.step = {'id': id_, 'type': type_}
        if 'parameters' not in self.step:
            self.step['parameters'] = dict()

    def add_widgets(self):
        # Type
        type_frame = ttk.Frame(self)
        type_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        type_label = ttk.Label(type_frame, text='Type')
        type_label.pack(side=tkinter.LEFT)
        type_value = tkinter.StringVar()
        type_options = steps_repository.instance.get_step_names()
        type_option = ttk.OptionMenu(type_frame, type_value, None, *type_options)
        type_value.set(steps_repository.instance.get_step_name(self.step['type']))
        type_option.pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        type_value.trace('w', self.type_changed)
        # ID
        id_frame = ttk.Frame(self)
        id_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        id_label = ttk.Label(id_frame, text='Implementation')
        id_label.pack(side=tkinter.LEFT)
        id_value = tkinter.StringVar()
        step_implementations = steps_repository.instance.get_step_implementations(self.step['type'])
        id_options = list()
        self.impl_name_to_id = dict()
        for step_impl in step_implementations:
            id_options.append(step_impl.get_name())
            self.impl_name_to_id[step_impl.get_name()] = step_impl.get_id()
        id_option = ttk.OptionMenu(id_frame, id_value, None, *id_options)
        id_value.set(steps_repository.instance.get_step_implementation(self.step['type'], self.step['id']).get_name())
        id_option.pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        id_value.trace('w', self.implementation_changed)
        # Parameter list
        parameters_frame = ttk.Frame(self)
        parameters_frame.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        parameters = tkinter.Listbox(parameters_frame)
        parameters.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)
        scrollbar = ttk.Scrollbar(parameters_frame)
        scrollbar.pack(side=tkinter.LEFT, fill=tkinter.Y)
        parameters.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=parameters.yview)
        # Step buttons
        buttons_frame = ttk.Frame(parameters_frame)
        buttons_frame.pack(side=tkinter.TOP)
        add_step = ttk.Button(buttons_frame, text='Add', command=self.add_parameter)
        add_step.pack(side=tkinter.TOP, fill=tkinter.X)
        edit_step = ttk.Button(buttons_frame, text='Edit', command=self.edit_parameter)
        edit_step.pack(side=tkinter.TOP, fill=tkinter.X)
        remove_step = ttk.Button(buttons_frame, text='Remove', command=self.remove_parameter)
        remove_step.pack(side=tkinter.TOP, fill=tkinter.X)
        # Save / discard
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        save_changes = ttk.Button(bottom_frame, text='Save', command=self.save_changes)
        save_changes.pack(side=tkinter.RIGHT)
        discard_changes = ttk.Button(bottom_frame, text='Discard', command=self.discard_changes)
        discard_changes.pack(side=tkinter.RIGHT)
        self.master.minsize(width=400, height=200)
        self.type_option = type_option
        self.id_option = id_option
        self.id_value = id_value
        self.type_value = type_value
        self.parameters_list = parameters
        self.parameter_ids = []
        self.rebuild_parameter_list()
        parameters.bind('<Double-Button-1>', self.parameter_double_click)

    def type_changed(self, *args):
        self.update_implementation_list()

    def update_implementation_list(self):
        type_ = self.get_type_id()
        if self.step['type'] != type_:
            self.step['type'] = type_
            step_implementations = steps_repository.instance.get_step_implementations(type_)
            id_options = list()
            self.impl_name_to_id = dict()
            for step_impl in step_implementations:
                id_options.append(step_impl.get_name())
                self.impl_name_to_id[step_impl.get_name()] = step_impl.get_id()
            self.id_value.set(id_options[0])
            self.id_option['menu'].delete(0, tkinter.END)
            for choice in id_options:
                self.id_option['menu'].add_command(label=choice, command=tkinter._setit(self.id_value, choice))

    def implementation_changed(self, *args):
        id_ = self.get_impl_id()
        if self.step['id'] != id_:
            self.step['id'] = id_
            self.step['parameters'] = dict()
            self.rebuild_parameter_list()

    def get_type_id(self):
        return self.type_name_to_id[self.type_value.get()]

    def get_impl_id(self):
        return self.impl_name_to_id[self.id_value.get()]

    def save_changes(self):
        if self.check_mandatory_parameters():
            if len(self.step['parameters']) == 0:
                del self.step['parameters']
            self.save(self.step, self.index)
            self.top.destroy()

    def discard_changes(self):
        self.top.destroy()

    def parameter_double_click(self, event):
        self.edit_parameter()

    def get_selected_parameter_id(self):
        index = self.parameters_list.curselection()[0]
        return self.parameter_ids[index]

    def add_parameter(self):
        impl = steps_repository.instance.get_step_implementation(self.step['type'], self.step['id'])
        parameter_list = []
        for parameter in impl.get_parameters():
            if parameter['id'] not in self.parameter_ids:
                parameter_list.append(parameter)
        if len(parameter_list) > 0:
            parameter_editor.ParameterEditor(self.top, self.save_parameter, parameter_list)
        else:
            messagebox.showerror('No More Parameters', 'Can\'t add anymore parameters')

    def edit_parameter(self):
        id_ = self.get_selected_parameter_id()
        impl = steps_repository.instance.get_step_implementation(self.step['type'], self.step['id'])
        parameter_list = []
        for parameter in impl.get_parameters():
            if parameter['id'] not in self.parameter_ids or parameter['id'] == id_:
                parameter_list.append(parameter)
        parameter_editor.ParameterEditor(self.top, self.save_parameter, parameter_list,
                                         (id_, self.step['parameters'][id_]))

    def remove_parameter(self):
        id = self.get_selected_parameter_id()
        del self.step['parameters'][id]
        self.rebuild_parameter_list()

    def save_parameter(self, parameter):
        key, value = parameter
        self.step['parameters'][key] = value
        self.rebuild_parameter_list()

    def rebuild_parameter_list(self):
        self.parameters_list.delete(0, tkinter.END)
        impl = steps_repository.instance.get_step_implementation(self.step['type'], self.step['id'])
        self.parameter_ids = list()
        i = 1
        for parameter in impl.get_parameters():
            if parameter['id'] in self.step['parameters']:
                self.parameter_ids.append(parameter['id'])
                parameter_name = parameter['name'] + ': ' + str(self.step['parameters'][parameter['id']])
                self.parameters_list.insert(i, parameter_name)
                i += 1

    def check_mandatory_parameters(self):
        impl = steps_repository.instance.get_step_implementation(self.step['type'], self.step['id'])
        mandatory_list = list()
        for parameter in impl.get_parameters():
            if 'default' not in parameter and parameter['id'] not in self.step['parameters']:
                mandatory_list.append('• ' + parameter['name'])
        if len(mandatory_list) == 0:
            return True
        else:
            error_message = 'The following mandatory parameters are missing:\n' + '\n'.join(mandatory_list)
            messagebox.showerror('Missing Parameters', error_message)
