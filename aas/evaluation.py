import os
import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager
from typing import Union, Optional
import multiprocessing
import multiprocessing.managers
from aas import config as cfg
from aas.state import OperationState
from aas.transforms import apply_transformation_with_timeout
from aas.observation.benchmark import BenchmarkFeatures
import json


# ================================== Evaluation Functions (Python Bindings) ==================================

# TODO: Adapt this function to be able to run code without benchmark name
def evaluate_code_with_bindings(code: str, function_name: str) -> tuple[Optional[float], Union[Exception, bool]]:
    """Lowers and runs the given MLIR code using Python bindings, then returns the execution time and assertion
    result (if the executed code returns the correct result).

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    pass_pipeline = """builtin.module(
        loop-invariant-code-motion,
        canonicalize,
        convert-vector-to-scf,
        convert-linalg-to-loops,
        buffer-deallocation-pipeline,
        scf-forall-to-parallel,
        convert-scf-to-openmp,
        expand-strided-metadata,
        finalize-memref-to-llvm,
        convert-scf-to-cf,
        lower-affine,

        convert-openmp-to-llvm,
        convert-vector-to-llvm,
        convert-math-to-llvm,
        convert-func-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-cf-to-llvm,

        reconcile-unrealized-casts,
        canonicalize,
        cse
    )"""

    with Context():
        module = Module.parse(code)
        pm = PassManager.parse(pass_pipeline)
        pm.run(module.operation)
    execution_engine = ExecutionEngine(
        module,
        shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
    )

    full_function_name = os.path.join(
        cfg.benchmarks_folder_path,
        function_name + ".mlir"
    )
    with open(full_function_name, "r") as f:
        original_code = f.read()

    np_file: np.lib.npyio.NpzFile = np.load(full_function_name + ".npz")
    expected: np.ndarray = np.load(full_function_name + ".npy")

    args_names: list[str] = sorted(
        np_file.files,
        key=lambda s: original_code.index(s)
    )
    args_map: dict[str, np.ndarray] = {arr: np_file[arr] for arr in args_names}
    args = []
    for arg_name in args_names:
        args.append(ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(args_map[arg_name])
        )))

    delta_arg = (ctypes.c_int64 * 1)(0)
    args.append(delta_arg)

    try:
        execution_engine.invoke("main", *args)
        execution_engine.invoke("main", *args)
    except Exception as e:
        np_file.close()
        return None, e
    actual = args_map[args_names[-1]]
    if expected.dtype == np.complex128:
        actual = actual.view(np.complex128).squeeze(len(actual.shape) - 1)
    assertion = np.allclose(actual, expected)

    np_file.close()
    return delta_arg[0], assertion


def evaluate_code_with_bindings_wrapper(code: str, function_name: str, exec_times: multiprocessing.managers.ListProxy, assertions: multiprocessing.managers.ListProxy):
    """Wrapper function for evaluate_code_with_bindings to be used in multiprocessing.

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.
        exec_times (multiprocessing.managers.ListProxy): A list to store the execution times.
        assertions (multiprocessing.managers.ListProxy): A list to store the assertion results
    """
    exec_time, assertion = evaluate_code_with_bindings(code, function_name)
    exec_times.append(exec_time)
    assertions.append(assertion)


def evaluate_code_with_bindings_and_timeout(code: str, function_name: str, timeout: Optional[float] = None):
    """Evaluates the given MLIR code using Python bindings with a timeout.

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.
        timeout (Optional[float]): The timeout in seconds.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    manager = multiprocessing.Manager()
    exec_times = manager.list()
    assertions = manager.list()
    process = multiprocessing.Process(target=evaluate_code_with_bindings_wrapper, args=(code, function_name, exec_times, assertions))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # The function is still running, terminate the process
        process.terminate()
        process.join()
        process.close()

        return None, False
    else:
        # The function completed within the timeout
        process.close()
        return exec_times[0], assertions[0]


# ================================== Evaluation Functions (MLIR CPU Runner) ==================================

def evaluate_code_with_cmd(code: str, tmp_file_path: str):
    """Lowers and runs the given MLIR code using MLIR opt and MLIR CPU Runner, then returns the execution time and assertion.

    Args:
        code (str): The MLIR code to run.
        tmp_file_path (str): The temporary file path to write the MLIR code.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    command_1 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt -loop-invariant-code-motion -canonicalize -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' -convert-vector-to-scf -convert-linalg-to-loops -buffer-deallocation-pipeline -scf-forall-to-parallel -convert-scf-to-openmp -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize -cse"
    command_2 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs={os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_c_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libomp.so"

    os.environ["OMP_NUM_THREADS"] = "8"

    with open(tmp_file_path, "w") as file:
        file.write(code)

    out = os.popen(f"""{command_1} {tmp_file_path} | {command_2} /dev/stdin""").read()

    if out:
        return int(out.strip().split('\n')[-1]), True
    else:
        return None, False


def evaluate_code_with_cmd_wrapper(code: str, tmp_file_path: str, exec_times: multiprocessing.managers.ListProxy, assertions: multiprocessing.managers.ListProxy):
    """Wrapper function for evaluate_code_with_cmd to be used in multiprocessing.

    Args:
        code (str): The MLIR code to run.
        tmp_file_path (str): The temporary file path to write the MLIR code.
        exec_times (multiprocessing.managers.ListProxy): A list to store the execution times.
        assertions (multiprocessing.managers.ListProxy): A list to store the assertion results
    """
    exec_time, assertion = evaluate_code_with_cmd(code, tmp_file_path)
    exec_times.append(exec_time)
    assertions.append(assertion)


def evaluate_code_with_cmd_and_timeout(code: str, tmp_file_path: str, timeout: Optional[float] = None):
    """Evaluates the given MLIR code using MLIR opt and MLIR CPU Runner with a timeout.

    Args:
        code (str): The MLIR code to run.
        tmp_file_path (str): The temporary file path to write the MLIR code.
        timeout (Optional[float]): The timeout in seconds.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    manager = multiprocessing.Manager()
    exec_times = manager.list()
    assertions = manager.list()
    process = multiprocessing.Process(target=evaluate_code_with_cmd_wrapper, args=(code, tmp_file_path, exec_times, assertions))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # The function is still running, terminate the process
        process.terminate()
        process.join()
        process.close()

        return None, False
    else:
        # The function completed within the timeout
        process.close()
        return exec_times[0], assertions[0]


# ================================== Evaluation Functions (Both) ==================================

def evaluate_code_with_timeout(state: OperationState, tmp_file_path: str, timeout: Optional[float] = None):
    """Evaluates the given MLIR code using Python bindings or MLIR opt and MLIR CPU Runner with a timeout.

    Args:
        state (OperationState): The state to run the Alpha AutoScheduler on.
        tmp_file_path (str): The temporary file path to write the MLIR code.
        timeout (Optional[float]): The timeout in seconds.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
        str: the transformed code.
    """
    # Get the code
    code = state.bench_features.code
    # Get the full schedule
    if cfg.optimization_mode == 'all':
        full_schedule = []
        for op_tag in state.bench_features.operation_tags:
            if op_tag == state.operation_tag:
                full_schedule.append(state.transformation_history)
            else:
                full_schedule.append([])
    else:
        full_schedule = [state.transformation_history]
    # Transform the code
    for action in state.transformation_history:
        # If code is not None or empty, apply the transformation
        if code:
            code = apply_transformation_with_timeout(
                state=state,
                code=code,
                tmp_file_path=tmp_file_path,
                action=action,
                timeout=timeout,
                use_vectorizer=cfg.use_vectorizer
            )
        else:
            # If code is None or empty, return execution error
            return None, False, code
    # Check last transformation code
    if not code:
        return None, False, code

    # Check execution database for the execution time of the given state
    if cfg.exec_db_path:
        try:
            with open(cfg.exec_db_path, "r") as f:
                exec_db = json.load(f)
            bench_db = exec_db.get(state.bench_features.bench_name)
            if bench_db:
                exec_time = bench_db.get(BenchmarkFeatures.any_schedule_to_str(full_schedule))
                if exec_time:
                    return exec_time, True, code
        except Exception:
            pass

    # Otherwise execute the code manually using Python bindings if enabled
    if cfg.use_bindings:
        exec_time, assertion = evaluate_code_with_bindings_and_timeout(code, state.bench_features.bench_name, timeout=timeout)
    else:
        exec_time, assertion = evaluate_code_with_cmd_and_timeout(code, tmp_file_path, timeout=timeout)
    # Store the execution time in the execution database
    if (exec_time is not None) and assertion and cfg.exec_db_path:
        try:
            with open(cfg.exec_db_path, "r") as f:
                exec_db = json.load(f)
            bench_db = exec_db.get(state.bench_features.bench_name)
            if not bench_db:
                bench_db = {}
                exec_db[state.bench_features.bench_name] = bench_db
            exec_db[state.bench_features.bench_name][BenchmarkFeatures.any_schedule_to_str(full_schedule)] = exec_time
            with open(cfg.exec_db_path, "w") as f:
                json.dump(exec_db, f, indent=2)
        except Exception:
            pass

    # Return the execution time and assertion result and transformed code
    return exec_time, assertion, code


def evaluate_benchmark_code_with_timeout(states: list[OperationState], tmp_file_path: str, timeout: Optional[float] = None):
    """Evaluates the given MLIR code using Python bindings or MLIR opt and MLIR CPU Runner with a timeout.

    Args:
        states (list[OperationState]): The states to run the Alpha AutoScheduler on.
        tmp_file_path (str): The temporary file path to write the MLIR code.
        timeout (Optional[float]): The timeout in seconds.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
        str: the transformed code.
    """
    if not states:
        return None, False, None
    # Get the code
    bench_features = states[0].bench_features
    code = bench_features.code
    # Get the full schedule
    full_schedule = []
    # Transform the code
    for state in states:
        for action in state.transformation_history:
            # If code is not None or empty, apply the transformation
            if code:
                code = apply_transformation_with_timeout(
                    state=state,
                    code=code,
                    tmp_file_path=tmp_file_path,
                    action=action,
                    timeout=timeout,
                    use_vectorizer=cfg.use_vectorizer
                )
            else:
                # If code is None or empty, return execution error
                return None, False, code
        # Update the full schedule
        full_schedule.insert(0, state.transformation_history)
    # Check last transformation code
    if not code:
        return None, False, code

    # Check execution database for the execution time of the given state
    if cfg.exec_db_path:
        try:
            with open(cfg.exec_db_path, "r") as f:
                exec_db = json.load(f)
            bench_db = exec_db.get(bench_features.bench_name)
            if bench_db:
                exec_time = bench_db.get(BenchmarkFeatures.any_schedule_to_str(full_schedule))
                if exec_time:
                    return exec_time, True, code
        except Exception:
            pass
    # Otherwise execute the code manually using Python bindings if enabled
    if cfg.use_bindings:
        exec_time, assertion = evaluate_code_with_bindings_and_timeout(code, bench_features.bench_name, timeout=timeout)
    else:
        exec_time, assertion = evaluate_code_with_cmd_and_timeout(code, tmp_file_path, timeout=timeout)
    # Store the execution time in the execution database
    if (exec_time is not None) and assertion and cfg.exec_db_path:
        try:
            with open(cfg.exec_db_path, "r") as f:
                exec_db = json.load(f)
            bench_db = exec_db.get(bench_features.bench_name)
            if not bench_db:
                bench_db = {}
                exec_db[bench_features.bench_name] = bench_db
            exec_db[bench_features.bench_name][BenchmarkFeatures.any_schedule_to_str(full_schedule)] = exec_time
            with open(cfg.exec_db_path, "w") as f:
                json.dump(exec_db, f, indent=2)
        except Exception:
            pass

    # Return the execution time and assertion result and transformed code
    return exec_time, assertion, code


def get_cached_exec_time(state: OperationState):
    """Get the cached execution time of the given state.

    Args:
        state (OperationState): The state to get the execution time of.

    Returns:
        Optional[float]: the cached execution time in seconds.
    """
    # Get the full schedule
    if cfg.optimization_mode == 'all':
        full_schedule = []
        for op_tag in state.bench_features.operation_tags:
            if op_tag == state.operation_tag:
                full_schedule.append(state.transformation_history)
            else:
                full_schedule.append([])
    else:
        full_schedule = [state.transformation_history]
    # Check execution database for the execution time of the given state
    try:
        if cfg.exec_db_path:
            with open(cfg.exec_db_path, "r") as f:
                exec_db = json.load(f)
            bench_db = exec_db.get(state.bench_features.bench_name)
            if bench_db:
                exec_time = bench_db.get(BenchmarkFeatures.any_schedule_to_str(full_schedule))
                if exec_time:
                    return exec_time
    except Exception:
        pass
    return None
