from importlib import import_module as _im
import sys, pathlib, types
m=types.ModuleType("common_hier")
m.__dict__.update(_im("atus_analysis.scripts.common_hier").__dict__)
sys.modules["common_hier"]=m