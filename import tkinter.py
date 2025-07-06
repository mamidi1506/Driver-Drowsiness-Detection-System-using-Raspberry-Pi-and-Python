import tkinter
tkinter._test()
try:
    import tkinter as tk
    print("Tkinter is installed.")
except ImportError:
    print("Tkinter is not installed.")
