import logging
import os
import shlex
import uuid
from concurrent.futures import Future
from functools import partial
from typing import Callable, Collection, Dict, Optional, Tuple, Union

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import (
    LauncherConfig,
    MPIConfig,
    PolicyConfig,
    SystemConfig,
    aurora_config,
)
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.orchestrator import ClusterClient

from colmena.models import Result
from colmena.models.methods import ColmenaMethod, ExecutableMethod
from colmena.queue.base import ColmenaQueues

from .base import FutureBasedTaskServer, convert_to_colmena_method
from .parsl import _execute_postprocess, _execute_preprocess

logger = logging.getLogger(__name__)


def _preprocess_callback(
    preprocess_future: Future,
    serialized_inputs: str,
    result: Result,
    task_server: "EnsembleTaskServer",
    topic: str,
    client: ClusterClient,
    execute_task: Task,
    post_process_task: Task,
):
    """Perform the next steps in an executable workflow

    If preprocessing was unsuccessful, send failed result back to client.
    If successful, submit the execute task to EnsembleLauncher and attach a callback
    to that function which will submit the postprocess task.

    Args:
        preprocess_future: Future provided when submitting pre-process task
        serialized_inputs: Original inputs, still in serialized form.
            We deserialize the inputs in `_execute_preprocess` and do not pass the input data back to this callback function to minimize data sent.
        result: Result object to be gradually updated
        task_server: Connection to the EnsembleTaskServer. Used to send results back to client
        topic: Topic of the task
        client: ClusterClient used to submit tasks
        execute_task: Task description for the executable step
        post_process_task: Task description for post-processing
    """

    # If execution was unsuccessful, send the result back to the client
    if preprocess_future.exception() is not None:
        return task_server.perform_callback(preprocess_future, result, topic)

    # If successful, unpack the outputs
    result, temp_dir, (exec_args, exec_stdin) = preprocess_future.result()

    # If unsuccessful, send the results back to the client
    if result.success is not None and not result.success:
        logger.info("Result failed during preprocessing. Sending back to client.")

        # Send the serialized inputs back to the client if they are expected
        if result.keep_inputs:
            result.inputs = serialized_inputs

        # Store the time it took to run the preprocessing
        result.time.running = result.time.additional.get("exec_preprocess", 0)
        return task_server.queues.send_result(result)

    # If successful, submit the execute step
    logger.info(
        f"Preprocessing was successful for {result.method} task. Submitting to execute"
    )

    ## Update execute task
    execute_task.executable = execute_task.executable + " " + " ".join(shlex.quote(str(a)) for a in exec_args)
    execute_task.stdout_file = str(temp_dir / "colmena.stdout")
    execute_task.stderr_file = str(temp_dir / "colmena.stderr")
    exec_future: Future = client.submit(execute_task)

    # Submit post-process to collect the results of the exec_function
    post_process_task.args = (
        0,  # exit_code: EL raises on non-zero, so reaching here means success
        result,
        temp_dir,
        serialized_inputs if result.keep_inputs else None,
    )
    post_future: Future = client.submit(post_process_task, dependencies=[exec_future])

    # Once that function completes, you are ready to submit the task back to the client
    def _send_back(future: Future):
        # Send the results back to the client, if desired
        #  This is only used if the task fails. (Otherwise, the copy of "result" held in the future is used)
        if result.keep_inputs:
            result.inputs = serialized_inputs
        return task_server.perform_callback(future, result, topic)

    post_future.add_done_callback(_send_back)


class EnsembleTaskServer(FutureBasedTaskServer):
    def __init__(
        self,
        queues: ColmenaQueues,
        methods: Collection[Union[Callable, ColmenaMethod]],
        system_config: SystemConfig | None = None,
        child_executor_name: str = "async_mpi",
        task_executor_name: list[str] | str = ["async_loky", "async_mpi"],
        gpu_selector: str = "ZE_AFFINITY_MASK",
        log_dir: str = "logs",
        mpi_flavour: str = "test",
    ):
        self._methods: Dict[str, Tuple[ColmenaMethod, str]] = {}
        for method in methods:
            method = convert_to_colmena_method(method)
            if isinstance(method, ExecutableMethod):
                self._methods[method.name] = (method, "exec")
            else:
                self._methods[method.name] = (method, "basic")
        super().__init__(queues=queues, method_names=list(self._methods.keys()))
        if system_config is None:
            system_config = aurora_config

        self._system_config = system_config

        ckpt_dir = os.path.join(os.getcwd(), f"ckpt_dir_{str(uuid.uuid4())}")
        os.makedirs(ckpt_dir, exist_ok=True)

        self._launcher_config = LauncherConfig(
            checkpoint_dir=ckpt_dir,
            child_executor_name=child_executor_name,
            task_executor_name=task_executor_name,
            return_stdout=True,
            master_logs=True,
            children_scheduler_policy="fixed_leafs_children_policy",
            policy_config=PolicyConfig(nlevels=1, leaf_nodes=len(get_nodes())),
            cluster=True,
            gpu_selector=gpu_selector,
            log_dir=log_dir,
            mpi_config=MPIConfig(mpi_flavour=mpi_flavour),
        )

        self._el: EnsembleLauncher = None
        self._client: ClusterClient = None

    def _setup(self):
        self._el = EnsembleLauncher(
            ensemble_file={},
            system_config=self._system_config,
            launcher_config=self._launcher_config,
        )
        self._el.start(wait_time=5)
        self._client = ClusterClient(
            checkpoint_dir=self._launcher_config.checkpoint_dir
        )
        self._client.start()

    def _cleanup(self):
        self._client.teardown()
        self._el.stop()

    def _build_el_task(self, task: Result) -> Task | Tuple[Task, Task, Task]:

        function, func_type = self._methods[task.method]

        if func_type == "exec":
            function: ExecutableMethod
            ##preprocess task
            pre_task = Task(
                task_id=f"pre_{function.name}",
                nnodes=1,
                ppn=1,
                executable=_execute_preprocess,
                args=(function, task),
                executor_name="async_loky",
            )

            ## Actual executable
            exec_task = Task(
                task_id=function.name,
                nnodes=task.resources.node_count,
                ppn=task.resources.cpu_processes,
                ngpus_per_process=task.resources.gpus_per_process,
                executable=" ".join(function.executable),
            )

            ## Post task
            post_task = Task(
                task_id=f"post_{function.name}",
                nnodes=1,
                ppn=1,
                executable=partial(_execute_postprocess, function),
                executor_name="async_loky",
            )
            return (pre_task, exec_task, post_task)
        elif func_type == "basic":
            return Task(
                task_id=task.task_id,
                nnodes=task.resources.node_count,
                ppn=task.resources.cpu_processes,
                ngpus_per_process=task.resources.gpus_per_process,
                executable=function,
                args=(task,),
                executor_name="async_mpi"
                if task.resources.node_count > 1
                else "async_loky",
            )
        else:
            raise ValueError("Unknown function type")

    def _submit(self, task: Result, topic: str):
        # Submit the application
        task.mark_start_task_submission()
        el_task = self._build_el_task(task=task)
        if isinstance(el_task, tuple):
            pre_task = el_task[0]
            exec_task = el_task[1]
            post_task = el_task[2]
            future = self._client.submit(pre_task)
            future.add_done_callback(
                lambda x: _preprocess_callback(
                    x,
                    task.inputs,
                    task,
                    self,
                    topic,
                    self._client,
                    exec_task,
                    post_task,
                )
            )
            return None  # `None` prevents the Task Server from adding its own callback
        else:
            return self._client.submit(el_task)
