try:
    from util import initialization
    import tkinter
    from util import logger
    from experiments import experiment
    from tkinter import ttk
    from gui import experiment_overview, start_dialog
    import sys


    def close_application():
        root.quit()


    def edit_experiment(file_path):
        if not isinstance(file_path, str):
            logger.log('No experiment file selected')
            exit(0)
        experiment_ = experiment.Experiment(file_path)
        experiment_window = experiment_overview.ExperimentOverview(root, experiment_)
        experiment_window.winfo_toplevel().protocol('WM_DELETE_WINDOW', close_application)


    root = tkinter.Tk()
    root.style = ttk.Style()
    root.style.theme_use('clam')
    root.withdraw()
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        edit_experiment(file_path)
        root.mainloop()
    else:
        dialog = start_dialog.StartDialog(root, edit_experiment)
        dialog.winfo_toplevel().protocol('WM_DELETE_WINDOW', close_application)
        root.mainloop()
except Exception as e:
    from tkinter import messagebox
    messagebox.showerror('Error', str(e))
    raise e
