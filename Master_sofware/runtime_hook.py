import os
import sys

def _append_run_path():
    if getattr(sys, 'frozen', False):
        pathlist = []

        # MEIPASS is a temporary folder where PyInstaller stores the running executable
        pathlist.append(sys._MEIPASS)

        # Append original PYTHONPATH
        if 'PYTHONPATH' in os.environ:
            pathlist = os.environ['PYTHONPATH'].split(os.pathsep) + pathlist

        # Set a new PYTHONPATH
        os.environ['PYTHONPATH'] = os.pathsep.join(pathlist)

_append_run_path()