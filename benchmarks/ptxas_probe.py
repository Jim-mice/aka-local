import importlib.util

path = r"<PROJECT_ROOT>\ops\swiglu_forward_v2b\kernel.py"
spec = importlib.util.spec_from_file_location("aka_local_swiglu_ptxas", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("PTXAS_BUILD_OK")
