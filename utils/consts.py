EXECUTABLE_PATH = "/scratch/nb3891/Script/MLIR_RL_2/MLScheduler/build/bin/CustomAutoSchedulerML"
NEPTUNE_PROJECT_NAME = "nassimam/mlir-rl"
NEPTUNE_API_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYWQ1NmJiZC05YmMyLTRkYzMtYTQ2ZC1kNWZjMGI0NmY1ZDYifQ=="
OMP_BUILD_PATH = "/data/na3758/llvm-project/build"
MLIR_BUILD_PATH = "/data/na3758/llvm-project/build-mlir"
TMP_FILES_PATH = "tmp/"


MAX_NUM_STORES_LOADS = 5 # the maximum number of loads in the nested loops
MAX_NUM_LOOPS = 7 # the max number of nested loops
MAX_NUM_LOAD_STORE_DIM = 7 # the max number of dimensions in load/store buffers 

NUM_TILE_SIZES = 4

NUM_TRANSFORMATIONS = 5

INTERCHANGE_ACTIONS = []
for c in [1, 2, 3]:
    for i in range(MAX_NUM_LOOPS-c):
        params = list(range(MAX_NUM_LOOPS))
        params[i], params[i+c] = params[i+c], params[i]
        INTERCHANGE_ACTIONS.append(tuple(params))