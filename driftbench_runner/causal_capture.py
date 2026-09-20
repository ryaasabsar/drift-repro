"""Instrumentation installed inside the real vLLM worker, only for diagnostics."""
from functools import partial
import inspect
from pathlib import Path
import re
import sys

from .common import digest, read_json, write_json
from .causal_ops import Archive, MUTATED, OUTPUT_ONLY, SUPPORTED, require, runtime_environment


def parameter_hashes(model):
    import torch
    with torch.inference_mode():
        return {name: {'shape': list(t.shape), 'dtype': str(t.dtype),
                       'sha256': digest(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())}
                for name, t in model.named_parameters()}


def output_descriptor(t):
    return {'output_buffer': {'shape': list(t.shape), 'stride': list(t.stride()),
                              'dtype': str(t.dtype).removeprefix('torch.')}}


def make_mode(recorder):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class OperationMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            custom = not name.startswith(('aten.', 'prim.'))
            if recorder.busy or (name not in SUPPORTED and not custom):
                return func(*args, **kwargs)
            recorder.busy = True
            try:
                # Mutable custom kernels not in the registry may depend on
                # hidden vLLM state; preserve observations but do not export them.
                event = {'kind': 'operation', 'name': name, 'module': recorder.target,
                         'replayable': name in SUPPORTED, 'schema': str(func._schema)}
                aliases = {}
                for i, a in enumerate(args):
                    if isinstance(a, torch.Tensor) and i not in OUTPUT_ONLY.get(name, ()):
                        aliases.setdefault(a.untyped_storage().data_ptr(), []).append(i)
                event['aliased_inputs'] = [v for v in aliases.values() if len(v) > 1]
                if event['aliased_inputs']:
                    event['replayable'] = False
                    event['unsupported_reason'] = 'Aliased input views require an explicit storage-preserving adapter'
                try:
                    event['args'] = {'tuple': [output_descriptor(a) if i in OUTPUT_ONLY.get(name, ())
                                              else recorder.archive.encode(a) for i, a in enumerate(args)]}
                    event['kwargs'] = recorder.archive.encode(kwargs)
                except ValueError as exc:
                    if 'limit exceeded' in str(exc):
                        raise
                    event.update(replayable=False, unsupported_reason=str(exc))
                actual = func(*args, **kwargs)
                observed = tuple(args[i] for i in MUTATED[name]) if name in MUTATED else actual
                try:
                    event['outputs'] = recorder.archive.encode(observed)
                except ValueError as exc:
                    if 'limit exceeded' in str(exc):
                        raise
                    event.update(replayable=False, unsupported_reason=str(exc))
                recorder.add(event)
                return actual
            finally:
                recorder.busy = False

    return OperationMode()


class Recorder:
    def __init__(self, model, target, max_bytes):
        self.model, self.target, self.max_bytes = model, target, max_bytes
        self.active, self.busy = False, False
        self.events, self.handles = [], []
        self.module_calls = {}
        self.pending_adapters = {}
        self.mode = None
        self.original_logits = model.compute_logits
        modules = dict(model.named_modules())
        if target:
            require(target in modules, f'Unknown module {target}; inspect modules.json from a boundary capture')
        selected = [name for name in modules if name and (
            (target and (name == target or name.startswith(target + '.'))) or
            (not target and (re.search(r'(?:^|\.)layers\.\d+$', name) or
                             name.endswith(('.embed_tokens', '.norm')))))]
        require(selected, 'No supported layer boundaries found; provide an exact --module')
        self.selected = selected
        for name in selected:
            module = modules[name]
            self.handles.append(module.register_forward_pre_hook(partial(self.before, name), with_kwargs=True))
            self.handles.append(module.register_forward_hook(partial(self.after, name), with_kwargs=True))

        def logits(*args, **kwargs):
            result = self.original_logits(*args, **kwargs)
            if self.active:
                self.add({'kind': 'logits', 'name': 'compute_logits',
                          'inputs': self.encode((args, kwargs)), 'outputs': self.encode(result)})
            return result

        model.compute_logits = logits

    def encode(self, obj):
        old = self.busy
        self.busy = True
        try:
            return self.archive.encode(obj)
        finally:
            self.busy = old

    def add(self, event):
        require(len(self.events) < 10000, 'Trace event limit exceeded; narrow the target')
        self.events.append({'event': len(self.events), **event})

    def before(self, name, module, args, kwargs):
        if not self.active:
            return
        call = self.module_calls.get(name, 0)
        self.module_calls[name] = call + 1
        self.add({'kind': 'module_input', 'name': name, 'call': call,
                  'class': type(module).__qualname__, 'inputs': self.encode((args, kwargs))})
        adapter = None
        if type(module).__name__ == 'ChunkGatedDeltaRule' and self.target:
            method = module._forward_method.__name__
            adapter = {'name': f'vllm.chunk_gated_delta_rule.{method}',
                       'args': self.encode(args), 'kwargs': self.encode(kwargs)}
        elif type(module).__name__ == 'RMSNormGated' and self.target:
            x = args[0] if args else kwargs['x']
            z = args[1] if len(args) > 1 else kwargs.get('z')
            adapter = {'name': 'vllm.rms_norm_gated.forward_cuda',
                       'args': self.encode((x, module.weight, module.bias, z, module.eps,
                                            module.group_size, module.norm_before_gate, module.activation)),
                       'kwargs': self.encode({})}
        if adapter:
            adapter.update(kind='operation', module=name, replayable=adapter['name'] in SUPPORTED,
                           schema='Explicit pure-module adapter; captures parameters and initial state')
            self.pending_adapters[name] = adapter
        if name == self.target and adapter is None:
            require(self.mode is None, 'Recursive target modules are unsupported')
            self.mode = make_mode(self)
            self.mode.__enter__()

    def after(self, name, module, args, kwargs, result):
        if not self.active:
            return
        if name == self.target and self.mode is not None:
            self.mode.__exit__(None, None, None)
            self.mode = None
        if name in self.pending_adapters:
            adapter = self.pending_adapters.pop(name)
            self.add({**adapter, 'outputs': self.encode(result)})
        self.add({'kind': 'module_output', 'name': name, 'call': self.module_calls[name] - 1,
                  'class': type(module).__qualname__, 'outputs': self.encode(result)})


def install(model, directory, target, max_bytes):
    require(not hasattr(model, '_drift_recorder'), 'Capture already installed')
    require(hasattr(model, 'compute_logits'), 'Model does not expose compute_logits')
    model._drift_recorder = Recorder(model, target, max_bytes)
    modules = []
    for name, module in model.named_modules():
        cls = type(module)
        try:
            source = inspect.getsourcefile(cls)
            sha = digest(Path(source).read_bytes()) if source else None
        except (TypeError, OSError):
            source, sha = None, None
        modules.append({'name': name, 'class': f'{cls.__module__}.{cls.__qualname__}',
                        'source': source, 'source_sha256': sha})
    write_json(Path(directory) / 'modules.json', modules)
    weights = parameter_hashes(model)
    write_json(Path(directory) / 'weights-before.json', weights)
    return {'selected_modules': model._drift_recorder.selected, 'weights_sha256': digest(weights)}


def begin(model, directory, protocol):
    recorder = model._drift_recorder
    require(not recorder.active, 'Previous capture still active')
    recorder.archive = Archive(recorder.max_bytes)
    recorder.events, recorder.module_calls = [], {}
    recorder.directory, recorder.protocol = Path(directory), protocol
    recorder.directory.mkdir(parents=True, exist_ok=False)
    recorder.active = True
    return True


def finish(model):
    import torch
    recorder = model._drift_recorder
    recorder.active = False
    require(recorder.mode is None, 'Operation context did not close')
    torch.cuda.synchronize()
    require(any(e['kind'] == 'logits' for e in recorder.events), 'No raw logits captured')
    require(any(e['kind'] == 'module_output' for e in recorder.events), 'No layer outputs captured')
    sha = recorder.archive.save(recorder.directory / 'tensors.npz')
    trace = {'schema_version': 1, 'status': 'complete', 'protocol': recorder.protocol,
             'execution': 'vLLM worker, serial eager, full-prefix one-token prefill; instrumented',
             'target_module': recorder.target, 'events': recorder.events,
             'environment': runtime_environment(), 'tensors_sha256': sha}
    trace['fingerprint'] = digest(trace)
    write_json(recorder.directory / 'trace.json', trace)
    size = recorder.archive.size
    recorder.archive = None
    recorder.events = []
    return {'trace': str(recorder.directory), 'bytes': size, 'fingerprint': trace['fingerprint']}


def uninstall(model, directory):
    recorder = model._drift_recorder
    require(not recorder.active, 'Capture is active')
    for h in recorder.handles:
        h.remove()
    model.compute_logits = recorder.original_logits
    del model._drift_recorder
    weights = parameter_hashes(model)
    write_json(Path(directory) / 'weights-after.json', weights)
    return digest(weights)


class CausalWorkerExtension:
    """String RPC methods avoid enabling vLLM's pickle serialization fallback."""
    def drift_capture_install(self, directory, target, max_bytes):
        return install(self.get_model(), directory, target, max_bytes)

    def drift_capture_begin(self, directory, protocol):
        return begin(self.get_model(), directory, protocol)

    def drift_capture_finish(self):
        return finish(self.get_model())

    def drift_capture_uninstall(self, directory):
        return uninstall(self.get_model(), directory)


def worker(plan_path):
    """Executed by the serving Python in its own process (spawn-safe main)."""
    import os
    from .common import local_environment
    from .credentials import load_credentials
    from .serving import configure_serving_environment
    plan = read_json(plan_path)
    config, output = plan['config'], Path(plan['output'])
    local_environment()
    configure_serving_environment(config)
    load_credentials()
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    kwargs = dict(plan['engine'])
    engine = LLM(model=config['model'], revision=config['revision'],
                 tokenizer_revision=config['revision'], seed=config['seed'],
                 trust_remote_code=False,
                 worker_extension_cls='driftbench_runner.causal_capture.CausalWorkerExtension', **kwargs)
    sampling = SamplingParams(temperature=0., top_p=1., top_k=-1, max_tokens=1,
                              seed=config['seed'], detokenize=False, logprobs=10,
                              repetition_penalty=1., presence_penalty=0., frequency_penalty=0.)
    info = engine.collective_rpc('drift_capture_install', args=(str(output), plan['module'], plan['max_bytes']))
    require(len(info) == 1, 'Tensor/pipeline parallel capture is not supported')
    records = []

    def generate(case):
        result = engine.generate([TokensPrompt(prompt_token_ids=case['input_ids'])], sampling, use_tqdm=False)[0]
        require(result.prompt_token_ids == case['input_ids'], 'vLLM changed the frozen input IDs')
        value = result.outputs[0]
        return {'token_ids': value.token_ids, 'top_logprobs':
                [{str(k): v.logprob for k, v in step.items()} for step in value.logprobs or []]}

    for i, case in enumerate(plan['cases']):
        before = generate(case)  # Recorder inactive: detect an instrumentation effect.
        for repeat in range(plan['repeats']):
            directory = output / f'case-{i:03d}' / f'capture-{repeat + 1:03d}'
            protocol = {'model': config['model'], 'revision': config['revision'],
                        'dtype': kwargs['dtype'], 'seed': config['seed'], 'engine': kwargs,
                        'case': case, 'sampling': {'temperature': 0., 'max_tokens': 1},
                        'weights_sha256': info[0]['weights_sha256'],
                        'capture_index': repeat + 1, 'prefix_order_index': i,
                        'implementation_sha256': digest([digest(Path(__file__).read_bytes()),
                                                         digest(Path(__file__).with_name('causal_ops.py').read_bytes())])}
            engine.collective_rpc('drift_capture_begin', args=(str(directory), protocol))
            response = generate(case)
            details = engine.collective_rpc('drift_capture_finish')
            records.append({'case': i, 'repeat': repeat + 1, 'response': response,
                            'uninstrumented_before': before, 'capture': details[0]})
            print(f'Captured {case["prompt_id"]} repeat {repeat + 1}', flush=True)
        after = generate(case)
        for r in records:
            if r['case'] == i:
                r['uninstrumented_after'] = after
                r['observer_check_equal'] = r['response'] == before == after
    weight_after = engine.collective_rpc('drift_capture_uninstall', args=(str(output),))[0]
    write_json(output / 'capture-run.json', {'schema_version': 1, 'status': 'complete',
               'records': records, 'weights_before': info[0]['weights_sha256'],
               'weights_after': weight_after, 'weights_unchanged': weight_after == info[0]['weights_sha256'],
               'environment': runtime_environment(), 'engine': kwargs,
               'limits': ['Hooks and host copies can alter execution; compare uninstrumented controls.',
                          'Eager full-prefix diagnostic, not original compiled/incremental decoding.',
                          'Operation trace covers registered ATen/custom calls; opaque Triton calls may be invisible.']})


if __name__ == '__main__':
    worker(sys.argv[1])
