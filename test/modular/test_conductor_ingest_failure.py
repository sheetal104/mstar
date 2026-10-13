"""A request whose ingest raises in the conductor fails alone: the client hears
why, the preprocess worker drops its inputs, and the loop's other messages run."""
import types
from collections import deque

from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import CurrentForwardConductorMetadata, PartitionDefinition
from mstar.model.base import ForwardPassArgs
from mstar.utils.ipc_format import NewRequestConductor, ReadsDone, WorkerMessageType

PREPROCESS = "api_server_preprocess_worker"


class _Model:
    """Reads its kwargs the way a model hook can trip on them."""

    def get_max_output_tokens(self, **model_kwargs):
        return min(model_kwargs.get("max_output_tokens", 8), 8)  # None: TypeError

    def get_partitions(self):
        return [PartitionDefinition(name="default", graph_walks={"prefill"}, initial_walk="prefill")]

    def get_partition_topology(self):
        return types.SimpleNamespace(connections=[])

    def get_initial_forward_pass_args(self, **kwargs):
        meta = CurrentForwardConductorMetadata(graph_walk="prefill", is_prefill=True)
        return ForwardPassArgs(full_metadata=meta, inputs=[], unpersist_tensors=[])

    def get_request_resource_configs(self, partition_fwd_args, model_kwargs):
        if "top_k" in model_kwargs:
            int(model_kwargs["top_k"])  # "many": ValueError
        return {}

    def prefix_key_streams(self):
        return {}


def _conductor(workers=("w0",), fail_new_request_to=None):
    c = Conductor.__new__(Conductor)
    c.sent = []

    def send(entity_id, msg):
        if entity_id == fail_new_request_to and msg.message_type == WorkerMessageType.NEW_REQUEST:
            raise OverflowError("int too big to convert")  # as the wire refuses one
        c.sent.append((entity_id, msg))

    c.communicator = types.SimpleNamespace(send=send)
    c.model = _Model()
    c.requests = {}
    c.draining = {}
    c.waiting_queue = []
    c.max_concurrent_requests = 1
    c.enable_nvtx = False
    c._drain_ttl_s = 120.0
    c._early_reads_done = {}
    c._early_abort_requests = set()
    c._draining_deadlines = deque()
    c.worker_graphs = {0: types.SimpleNamespace(graph_walks={"prefill"})}
    c._all_worker_graph_ids_to_graph_walks = {0: {"prefill"}}
    c._assign_worker_graphs_to_workers = lambda: {0: list(workers)}
    c._build_request_sharding_config = lambda worker_graph_to_workers: None
    return c


def _request(rid, **model_kwargs):
    return NewRequestConductor(
        request_id=rid, initial_signals={}, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs=model_kwargs,
    )


def _failures(c):
    return [m.body for e, m in c.sent if e == "api_server" and m.message_type == "request_failed"]


def _targets(c, message_type):
    return {e for e, m in c.sent if getattr(m, "message_type", None) == message_type}


def test_a_hook_raising_before_registration_fails_the_request_as_a_400():
    c = _conductor()
    c._ingest_request(_request("r1", max_output_tokens=None))

    assert "r1" not in c.requests
    [failure] = _failures(c)
    assert failure.request_id == "r1" and failure.status == 400
    assert "TypeError" in failure.error_message
    # only the preprocess worker holds it; no worker heard of it
    assert _targets(c, WorkerMessageType.REMOVE_REQUEST) == {PREPROCESS}
    assert _targets(c, WorkerMessageType.NEW_REQUEST) == set()


def test_a_hook_raising_after_registration_frees_its_slot():
    c = _conductor()
    c._ingest_request(_request("r1", top_k="many"))

    assert "r1" not in c.requests and not c.draining
    [failure] = _failures(c)
    assert failure.status == 400 and "ValueError" in failure.error_message
    assert _targets(c, WorkerMessageType.REMOVE_REQUEST) == {PREPROCESS}
    assert _targets(c, WorkerMessageType.DRAIN_REQUEST) == set()


def test_a_request_the_wire_refuses_before_any_worker_fails_without_a_drain():
    c = _conductor(fail_new_request_to="w0")
    c._ingest_request(_request("r1"))

    assert "r1" not in c.requests and not c.draining
    [failure] = _failures(c)
    assert failure.status == 500 and "OverflowError" in failure.error_message
    assert _targets(c, WorkerMessageType.REMOVE_REQUEST) == {PREPROCESS}


def test_a_failure_after_a_worker_has_it_drains_before_the_client_hears():
    c = _conductor(workers=("w0", "w1"), fail_new_request_to="w1")
    c._ingest_request(_request("r1"))

    assert _targets(c, WorkerMessageType.NEW_REQUEST) == {"w0"}
    assert _targets(c, WorkerMessageType.DRAIN_REQUEST) == {"w0", "w1", PREPROCESS}
    assert _failures(c) == []
    c._try_admit_waiting = lambda: None
    for entity in ("w0", "w1", PREPROCESS):
        c._handle_reads_done(ReadsDone(request_id="r1", entity_id=entity))
    [failure] = _failures(c)
    assert "OverflowError" in failure.error_message
    assert "r1" not in c.requests


def test_a_bad_queued_request_leaves_the_queue_to_the_next():
    c = _conductor()
    c.waiting_queue = [_request("bad", top_k="many"), _request("good")]
    c._try_admit_waiting()

    assert [f.request_id for f in _failures(c)] == ["bad"]
    assert list(c.requests) == ["good"] and c.waiting_queue == []
    assert _targets(c, WorkerMessageType.NEW_REQUEST) == {"w0"}
