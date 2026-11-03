"""Output-patch augmentation of a ReLU artifact (G3b `compose_output`).

A patch artifact reads parent inputs and/or parent outputs and replaces named
parent outputs. The result is one hidden-ReLU/final-linear artifact
(allocation mode `augmented`): parent hidden layers are copied verbatim, the
parent's final linear layer becomes a signed boundary layer relu(+-(Wh+b)),
bound inputs and passthrough outputs travel on exact signed relays, and the
patch's layers read the reconstructed values. Every parent and patch tensor is
an exact (possibly negated) slice of the composed weights; `verify_placement`
checks this. The composition is labelled augmentation and claims nothing about
contracts, which are checked separately (see wspl_transaction).
"""
from fractions import Fraction as Q
import hashlib
from .wspl_artifact import ReLUArtifact, canonical_bytes

OPERATION = 'compose_output'
COMPOSER = 'wspl.compose_output.v0'


def _artifact(value):
    return value if isinstance(value, ReLUArtifact) else ReLUArtifact(value)


def _layers(document):
    return [([[Q(w) for w in row] for row in layer['weights']], [Q(b) for b in layer['bias']])
            for layer in document['layers']]


def composition_record(parent, patch, bindings, replace):
    parent, patch = _artifact(parent), _artifact(patch)
    return {'format': 'wspl.composition.v0', 'operation': OPERATION,
            'parent_sha256': parent.sha256, 'patch_sha256': patch.sha256,
            'bindings': dict(sorted(bindings.items())), 'replace': dict(sorted(replace.items()))}


def _validate(p, q, bindings, replace):
    p_in = {x['name']: (i, x['type']) for i, x in enumerate(p['inputs'])}
    p_out = {x['name']: (i, x['type']) for i, x in enumerate(p['outputs'])}
    q_in = {x['name']: x['type'] for x in q['inputs']}
    q_out = {x['name']: x['type'] for x in q['outputs']}
    if not isinstance(bindings, dict) or set(bindings) != set(q_in):
        raise ValueError('bindings must name every patch input exactly once')
    if len(set(bindings.values())) != len(bindings):
        raise ValueError('each parent input/output may feed at most one patch input')
    sources = {}
    for name, source in bindings.items():
        kind, _, key = source.partition('.') if isinstance(source, str) else ('', '', '')
        table = {'input': p_in, 'parent': p_out}.get(kind)
        if table is None or key not in table:
            raise ValueError(f'binding {name!r}: source must be input.<parent input> or parent.<parent output>, got {source!r}')
        index, ty = table[key]
        if ty['unit'] != q_in[name]['unit']:
            raise ValueError(f'binding {name!r}: unit {ty["unit"]!r} differs from patch input unit {q_in[name]["unit"]!r}')
        if kind == 'input' and not (Q(q_in[name]['lower']) <= Q(ty['lower']) and Q(ty['upper']) <= Q(q_in[name]['upper'])):
            raise ValueError(f'binding {name!r}: parent input domain is not inside the patch input domain')
        sources[name] = (kind, index)
    if not isinstance(replace, dict) or set(replace) != set(q_out):
        raise ValueError('replace must map every patch output to a parent output; new outputs are not supported')
    if len(set(replace.values())) != len(replace):
        raise ValueError('a parent output may be replaced at most once')
    targets = {}
    for name, target in replace.items():
        if target not in p_out:
            raise ValueError(f'replace {name!r}: unknown parent output {target!r}')
        ty = p_out[target][1]
        if (ty['kind'], ty['unit']) != (q_out[name]['kind'], q_out[name]['unit']):
            raise ValueError(f'replace {name!r}: kind/unit differ from parent output {target!r}')
        targets[p_out[target][0]] = name
    return sources, targets


def compose_output(parent, patch, *, bindings, replace):
    """Return the augmented artifact computing parent outputs with `replace` applied."""
    parent, patch = _artifact(parent), _artifact(patch)
    p, q = parent.to_dict(), patch.to_dict()
    sources, targets = _validate(p, q, bindings, replace)
    q_index = {x['name']: i for i, x in enumerate(q['outputs'])}
    replaced = {o: q_index[name] for o, name in targets.items()}
    p_layers, q_layers = _layers(p), _layers(q)
    p_hidden, (p_final_w, p_final_b) = p_layers[:-1], p_layers[-1]
    q_hidden, (q_final_w, q_final_b) = q_layers[:-1], q_layers[-1]
    n_in, n_out = len(p['inputs']), len(p['outputs'])
    carried_inputs = sorted({index for kind, index in sources.values() if kind == 'input'})
    boundary = sorted({index for kind, index in sources.values() if kind == 'parent'} | (set(range(n_out)) - set(replaced)))

    hidden, tags = [], []

    def signed(rows, row_tags, terms, bias, tag, args):
        """Append relu(+value) and relu(-value) rows; return the signal reconstructing value."""
        rows.append((dict(terms), bias))
        row_tags.append([tag + '+'] + args)
        rows.append(({c: -v for c, v in terms.items()}, -bias))
        row_tags.append([tag + '-'] + args)
        return {len(rows) - 2: Q(1), len(rows) - 1: Q(-1)}

    # Signals: signed values as linear combinations of the previous layer's columns.
    inputs = {j: {j: Q(1)} for j in carried_inputs}
    parent_cols = list(range(n_in))
    for layer_id, (weights, biases) in enumerate(p_hidden):
        rows, row_tags = [], []
        for r, (row, bias) in enumerate(zip(weights, biases)):
            rows.append(({parent_cols[k]: w for k, w in enumerate(row) if w}, bias))
            row_tags.append(['parent', layer_id, r])
        carried = {}
        for j, terms in inputs.items():
            carried[j] = signed(rows, row_tags, terms, Q(0), 'relay', ['input', j])
        hidden.append(rows)
        tags.append(row_tags)
        parent_cols, inputs = list(range(len(weights))), carried

    rows, row_tags = [], []
    outputs = {}
    for o in boundary:
        outputs[o] = signed(rows, row_tags, {parent_cols[k]: w for k, w in enumerate(p_final_w[o]) if w},
                            p_final_b[o], 'boundary', [o])
    carried = {}
    for j, terms in inputs.items():
        carried[j] = signed(rows, row_tags, terms, Q(0), 'relay', ['input', j])
    hidden.append(rows)
    tags.append(row_tags)
    inputs = carried

    q_names = [x['name'] for x in q['inputs']]
    patch_inputs = [inputs[index] if kind == 'input' else outputs[index]
                    for kind, index in (sources[name] for name in q_names)]

    def read_patch_inputs(row):
        terms = {}
        for weight, signal in zip(row, patch_inputs):
            for column, sign in signal.items():
                if weight:
                    terms[column] = terms.get(column, Q(0)) + weight * sign
        return terms

    passthrough = {o: outputs[o] for o in range(n_out) if o not in replaced}
    patch_cols = None
    for layer_id, (weights, biases) in enumerate(q_hidden):
        rows, row_tags = [], []
        for r, (row, bias) in enumerate(zip(weights, biases)):
            terms = read_patch_inputs(row) if layer_id == 0 else {patch_cols[k]: w for k, w in enumerate(row) if w}
            rows.append((terms, bias))
            row_tags.append(['patch', layer_id, r])
        carried = {}
        for o, terms in passthrough.items():
            carried[o] = signed(rows, row_tags, terms, Q(0), 'passthrough', [o])
        hidden.append(rows)
        tags.append(row_tags)
        patch_cols, passthrough = list(range(len(weights))), carried

    final, final_tags = [], []
    for o in range(n_out):
        if o in replaced:
            row, bias = q_final_w[replaced[o]], q_final_b[replaced[o]]
            terms = read_patch_inputs(row) if not q_hidden else {patch_cols[k]: w for k, w in enumerate(row) if w}
            final.append((terms, bias))
            final_tags.append(['patch_final', replaced[o]])
        else:
            final.append((passthrough[o], Q(0)))
            final_tags.append(['passthrough', o])

    widths = [n_in] + [len(rows) for rows in hidden] + [n_out]
    dense = []
    for level, rows in enumerate(hidden + [final]):
        dense.append({'activation': 'relu' if level < len(hidden) else 'linear',
                      'weights': [[str(terms.get(c, Q(0))) for c in range(widths[level])] for terms, _ in rows],
                      'bias': [str(bias) for _, bias in rows]})
    parameters = sum(b * (a + 1) for a, b in zip(widths, widths[1:]))
    record = composition_record(parent, patch, bindings, replace)
    parent_alloc, patch_alloc = p['allocation'], q['allocation']
    document = {
        'format': 'wspl.relu_mlp.v0', 'semantics': 'rational_real_graph', 'compiler': COMPOSER,
        'source_sha256': hashlib.sha256(canonical_bytes(record)).hexdigest(),
        'inputs': p['inputs'], 'outputs': p['outputs'], 'layers': dense,
        'allocation': {'mode': 'augmented', 'widths': widths, 'parameters': parameters},
        'source_map': {'composition': {
            **record, 'label': 'output-patch augmentation',
            'hidden_row_tags': tags, 'final_row_tags': final_tags,
            'patch_output_types': {x['name']: x['type'] for x in q['outputs']},
            'parent_allocation': parent_alloc, 'patch_allocation': patch_alloc,
            'added_depth': len(widths) - len(parent_alloc['widths']),
            'added_parameters': parameters - parent_alloc['parameters']}}}
    return ReLUArtifact(document)


def verify_placement(edited, parent, patch):
    """Check that every parent and patch tensor is an exact slice of `edited`.

    Returns the number of checked rows; raises ValueError on the first mismatch.
    """
    edited, parent, patch = _artifact(edited), _artifact(parent), _artifact(patch)
    e = edited.to_dict()
    composition = e['source_map'].get('composition') if isinstance(e['source_map'], dict) else None
    if e['allocation']['mode'] != 'augmented' or not composition:
        raise ValueError('not an output-patch augmentation artifact')
    if composition['parent_sha256'] != parent.sha256 or composition['patch_sha256'] != patch.sha256:
        raise ValueError('composition record names different parent/patch artifacts')
    p, q = parent.to_dict(), patch.to_dict()
    sources, _ = _validate(p, q, composition['bindings'], composition['replace'])
    p_layers, q_layers = _layers(p), _layers(q)
    e_layers = _layers(e)
    all_tags = composition['hidden_row_tags'] + [composition['final_row_tags']]
    if len(all_tags) != len(e_layers):
        raise ValueError('row tags do not match composed depth')
    q_names = [x['name'] for x in q['inputs']]
    checked = 0

    def expect(level, r, expected_terms, expected_bias, what):
        weights, biases = e_layers[level]
        row = weights[r]
        actual = {c: w for c, w in enumerate(row) if w}
        if actual != {c: w for c, w in expected_terms.items() if w} or biases[r] != expected_bias:
            raise ValueError(f'composed layer {level} row {r} is not the exact slice for {what}')

    def signal_columns(level, kind, index):
        """{column: sign} reconstructing a parent input or output at composed `level`."""
        if kind == 'input' and level == 0:
            return {index: Q(1)}
        prev = all_tags[level - 1]
        label = 'relay' if kind == 'input' else ('boundary' if prev and any(t[0].startswith('boundary') for t in prev) else 'passthrough')
        key = ['input', index] if kind == 'input' else [index]
        pos = [c for c, t in enumerate(prev) if t[0] == label + '+' and t[1:] == key]
        neg = [c for c, t in enumerate(prev) if t[0] == label + '-' and t[1:] == key]
        if len(pos) != 1 or len(neg) != 1:
            raise ValueError(f'no unique signed relay for {kind} {index} at layer {level}')
        return {pos[0]: Q(1), neg[0]: Q(-1)}

    def parent_hidden_cols(level, layer_id):
        if level == 0:
            return list(range(len(p['inputs'])))
        return [c for c, t in enumerate(all_tags[level - 1]) if t[0] == 'parent' and t[1] == layer_id - 1]

    for level, row_tags in enumerate(all_tags):
        for r, tag in enumerate(row_tags):
            kind = tag[0]
            if kind == 'parent':
                layer_id, row = tag[1], tag[2]
                cols = parent_hidden_cols(level, layer_id)
                w, b = p_layers[layer_id][0][row], p_layers[layer_id][1][row]
                expect(level, r, {cols[k]: v for k, v in enumerate(w)}, b, f'parent layer {layer_id} row {row}')
            elif kind in ('boundary+', 'boundary-'):
                sign = Q(1) if kind.endswith('+') else Q(-1)
                o = tag[1]
                cols = parent_hidden_cols(level, len(p_layers) - 1)
                w, b = p_layers[-1][0][o], p_layers[-1][1][o]
                expect(level, r, {cols[k]: sign * v for k, v in enumerate(w)}, sign * b, f'parent final row {o} ({kind})')
            elif kind in ('relay+', 'relay-', 'passthrough+', 'passthrough-'):
                sign = Q(1) if kind.endswith('+') else Q(-1)
                source = ('input', tag[2]) if kind.startswith('relay') else ('parent', tag[1])
                cols = signal_columns(level, *source)
                expect(level, r, {c: sign * v for c, v in cols.items()}, Q(0), f'{kind} {tag[1:]}')
            elif kind == 'passthrough':
                expect(level, r, signal_columns(level, 'parent', tag[1]), Q(0), f'final passthrough of output {tag[1]}')
            elif kind in ('patch', 'patch_final'):
                layer_id, row = (tag[1], tag[2]) if kind == 'patch' else (len(q_layers) - 1, tag[1])
                w, b = q_layers[layer_id][0][row], q_layers[layer_id][1][row]
                if layer_id == 0:
                    terms = {}
                    for weight, name in zip(w, q_names):
                        for c, sign in signal_columns(level, *sources[name]).items():
                            terms[c] = terms.get(c, Q(0)) + weight * sign
                else:
                    cols = [c for c, t in enumerate(all_tags[level - 1]) if t[0] == 'patch' and t[1] == layer_id - 1]
                    terms = {cols[k]: v for k, v in enumerate(w)}
                expect(level, r, terms, b, f'patch layer {layer_id} row {row}')
            else:
                raise ValueError(f'unknown row tag {tag!r}')
            checked += 1
    return checked


def interface_obligations(parent, patch, bindings):
    """I/O contracts on the parent: every parent output feeding a patch input stays
    inside that patch input's declared bounds (declared output types are not guarantees)."""
    parent, patch = _artifact(parent), _artifact(patch)
    q_in = {x['name']: x['type'] for x in patch.to_dict()['inputs']}
    contracts = []
    for name, source in sorted(bindings.items()):
        if not source.startswith('parent.'):
            continue
        port, ty = 'output.' + source.split('.', 1)[1], q_in[name]
        side = lambda terms, bias: {'terms': terms, 'bias': bias, 'unit': ty['unit']}
        contracts.append({'name': f'interface_{name}', 'assumptions': {'op': 'true'}, 'ensures': {'op': 'and', 'args': [
            {'op': 'ge', 'left': side({port: '1'}, '0'), 'right': side({}, ty['lower'])},
            {'op': 'le', 'left': side({port: '1'}, '0'), 'right': side({}, ty['upper'])}]}})
    return contracts
