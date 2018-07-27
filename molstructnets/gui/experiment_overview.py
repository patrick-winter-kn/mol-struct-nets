import sys
import tkinter
from tkinter import ttk

from gui import step_overview
from steps import steps_repository
from util import misc


class ExperimentOverview(ttk.Frame):

    def __init__(self, parent, experiment):
        self.top = tkinter.Toplevel(parent)
        self.top.title('Edit Experiment')
        ttk.Frame.__init__(self, self.top)
        self.experiment = experiment
        self.seed_spinner = None
        self.steps_list = None
        self.add_widgets()
        self.pack(fill=tkinter.BOTH, expand=True)

    def add_widgets(self):
        # Location
        location_frame = ttk.Frame(self)
        location_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        location_label = ttk.Label(location_frame, text=self.experiment.get_file_path())
        location_label.pack(side=tkinter.LEFT)
        # Seed
        seed_frame = ttk.Frame(self)
        seed_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        seed_label = ttk.Label(seed_frame, text='seed')
        seed_label.pack(side=tkinter.LEFT)
        seed_value = tkinter.StringVar()
        seed_spinner = tkinter.Spinbox(seed_frame, from_=-sys.maxsize - 1, to=sys.maxsize, textvariable=seed_value)
        seed = misc.to_int(self.experiment.get_seed())
        if seed is None:
            seed = ''
        seed_value.set(seed)
        seed_spinner.pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        # Steps
        steps_frame = ttk.Frame(self)
        steps_frame.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        steps = tkinter.Listbox(steps_frame)
        steps.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(steps_frame)
        scrollbar.pack(side=tkinter.LEFT, fill=tkinter.Y)
        steps.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=steps.yview)
        # Step buttons
        buttons_frame = ttk.Frame(steps_frame)
        buttons_frame.pack(side=tkinter.TOP)
        add_step = ttk.Button(buttons_frame, text='Add', command=self.add_step)
        add_step.pack(side=tkinter.TOP, fill=tkinter.X)
        edit_step = ttk.Button(buttons_frame, text='Edit', command=self.edit_step)
        edit_step.pack(side=tkinter.TOP, fill=tkinter.X)
        remove_step = ttk.Button(buttons_frame, text='Remove', command=self.remove_step)
        remove_step.pack(side=tkinter.TOP, fill=tkinter.X)
        up_step = ttk.Button(buttons_frame, text='Up', command=self.move_step_up)
        up_step.pack(side=tkinter.TOP, fill=tkinter.X)
        down_step = ttk.Button(buttons_frame, text='Down', command=self.move_step_down)
        down_step.pack(side=tkinter.TOP, fill=tkinter.X)
        # Save / discard
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        save_changes = ttk.Button(bottom_frame, text='Save', command=self.save_changes)
        save_changes.pack(side=tkinter.RIGHT)
        discard_changes = ttk.Button(bottom_frame, text='Discard', command=self.discard_changes)
        discard_changes.pack(side=tkinter.RIGHT)
        self.master.minsize(width=500, height=300)
        self.seed_spinner = seed_spinner
        self.steps_list = steps
        self.rebuild_step_list()
        steps.bind('<Double-Button-1>', self.step_double_click)

    def get_selected_step_index(self):
        return self.steps_list.curselection()[0]

    def get_seed(self):
        return misc.to_int(self.seed_spinner.get())

    def rebuild_step_list(self):
        self.steps_list.delete(0, tkinter.END)
        for i in range(self.experiment.number_steps()):
            step_id = self.experiment.get_step(i)['id']
            step_type = self.experiment.get_step(i)['type']
            type_name = steps_repository.instance.get_step_name(step_type)
            implementation_name = steps_repository.instance.get_step_implementation(step_type, step_id).get_name()
            self.steps_list.insert(i + 1, type_name + ' - ' + implementation_name)

    def save_changes(self):
        self.experiment.set_random_seed(self.get_seed())
        self.experiment.save()
        self.top.quit()

    def discard_changes(self):
        self.top.quit()

    def add_step(self):
        step_overview.StepOverview(self.top, self.save_step)

    def edit_step(self):
        index = self.get_selected_step_index()
        step = self.experiment.get_step(index)
        step_overview.StepOverview(self.top, self.save_step, step, index)

    def save_step(self, step, index):
        if index is None:
            self.experiment.add_step(step)
        else:
            self.experiment.set_step(step, index)
        self.rebuild_step_list()

    def step_double_click(self, event):
        self.edit_step()

    def remove_step(self):
        index = self.get_selected_step_index()
        if self.experiment.remove_step(index):
            self.rebuild_step_list()

    def move_step_up(self):
        index_1 = self.get_selected_step_index()
        index_2 = index_1 - 1
        if self.experiment.swap_steps(index_1, index_2):
            self.rebuild_step_list()
            self.steps_list.select_set(index_2)

    def move_step_down(self):
        index_1 = self.get_selected_step_index()
        index_2 = index_1 + 1
        if self.experiment.swap_steps(index_1, index_2):
            self.rebuild_step_list()
            self.steps_list.select_set(index_2)
