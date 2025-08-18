import sys
import os
import importlib.util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# 动态导入 sa_gpr_kernels.py
spec = importlib.util.find_spec('sa_gpr_kernels')
if spec is None:
	raise ImportError('Cannot find sa_gpr_kernels module')
sa_gpr_kernels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa_gpr_kernels)
sagpr_kernel = sa_gpr_kernels.sagpr_kernel

# sa_gpr_kernels.py -lval 0 -f coords_1000.xyz -sg 0.3 -lc 6 -rc 4.0 -cw 1.0 -cen O
sagpr_kernel(
	lval=0,                # -lval 0
	ftrs='SA-GPR/example/water_monomer/coords_1000.xyz',# -f coords_1000.xyz
	sg=0.3,                # -sg 0.3
	lc=6,                  # -lc 6
	rcut=4.0,              # -rc 4.0
	cweight=1.0,           # -cw 1.0
	centers=['O']          # -cen O
)