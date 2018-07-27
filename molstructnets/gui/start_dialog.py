import tkinter
from tkinter import ttk, filedialog


class StartDialog(ttk.Frame):

    def __init__(self, parent, callback):
        self.callback = callback
        self.top = tkinter.Toplevel(parent)
        self.top.title('Select Experiment')
        ttk.Frame.__init__(self, self.top)
        self.add_widgets()
        self.pack(fill=tkinter.BOTH, expand=True)

    def add_widgets(self):
        button_frame = ttk.Frame(self)
        button_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        new = ttk.Button(button_frame, text='New', command=self.new_file)
        new.pack(side=tkinter.LEFT)
        open_ = ttk.Button(button_frame, text='Open', command=self.open_file)
        open_.pack(side=tkinter.LEFT)

    def new_file(self):
        self.top.destroy()
        file_path = filedialog.asksaveasfilename(initialdir="~", title="Select experiment file",
                                                 filetypes=(("experiment files", "*.msne"),))
        self.callback(file_path)

    def open_file(self):
        self.top.destroy()
        file_path = filedialog.askopenfilename(initialdir="~", title="Select experiment file",
                                               filetypes=(("experiment files", "*.msne"),))
        self.callback(file_path)
