from collections import defaultdict
import stim
import numpy as np


def xor_event_nodes(events: dict, event_name: str, tag: str | None = None) -> dict:
    """
    Compute per-sample XOR across outcome of a given event using ±1 encoding:
        False -> +1
        True  -> -1

    XOR becomes multiplication.
    """
    if event_name not in events:
        for k in events.keys():
            if event_name in k:
                event_name = k
                break
    if event_name not in events:
        raise ValueError(f"Event '{event_name}' not found")

    event_nodes = events[event_name]
    # Select nodes by tag in name
    selected = [
        values
        for node_name, values in event_nodes.items()
        if tag is None or tag in node_name
    ]

    if not selected:
        return {event_name + (tag if tag else ""): []}

    result = []

    for sample_values in zip(*selected):
        prod = 1
        for v in sample_values:
            # False → +1, True → -1
            prod *= -1 if v else 1
        result.append(prod)

    return {event_name + (tag if tag else ""): result}


def concat_events_per_sample(event_results: dict) -> dict:
    """
    Convert:
        {event_name: [values per sample]}
    into:
        {sample_index: [values across events]}
    """

    if not event_results:
        return {}
    event_names = list(event_results.keys())
    num_samples = len(next(iter(event_results.values())))

    result = {}

    for i in range(num_samples):
        result[f"sample{i}"] = [event_results[event][i] for event in event_names]

    return result


def run(context, program, num_samples=1):
    """Run a program and return the outcomes of the recorded events in a structured way.

    Return:
        {event_name: {node_name: [values per sample]}}
    """
    samples = compile_and_sample(context, program, num_samples=num_samples)
    node_latest_outcomes = defaultdict(dict)
    outcomes = defaultdict(dict)
    global_idx = 0
    for r in context.record.events:
        if (
            r.size != 0
        ):  # Some observables are made of already recoreded results and have size 0, we skip those for now. This will be used later to build the detector error model.

            for n in r.measured_nodes:
                node_latest_outcomes[n.tag] = global_idx
                outcomes[r.tag][n.tag] = global_idx

                if "bridge" in n.tag:
                    outcomes["bridge"][n.tag] = global_idx
                global_idx += 1
        else:
            for n in r.measured_nodes:
                outcomes[r.tag][n.tag] = node_latest_outcomes[n.tag]

    out_array = np.array(samples)
    sample_value = outcomes.copy()
    for k, v in outcomes.items():
        for n, idx in v.items():
            sample_value[k][n] = out_array[:, idx]
    return sample_value


def compile_and_sample(ctx, program, num_samples=10):
    """Compile a program and sample it a specified number of times.

    Parameters
    ----------
    ctx : Context
    program : List[instruction]
    num_samples : int, optional
        number of samples to generate, by default 10

    Returns
    -------
    np.ndarray
        array of samples outcome. Each column corresponds to a recorded event node, and each row to a sample.
    """
    compilers = []

    for p in program:
        compilers.extend(p.build_compiler_instructions())

    stim_instructions = []
    meas_tag = []
    # print("Available record tags:")
    for c in compilers:
        stim_instructions.append(f"# Compiler for {c.tag}")
        si, tag = c.compile(ctx.memory)
        if tag:
            if not isinstance(tag, list):
                tag = [tag]

            for t in tag:
                ctx.record.add_event(t)
                # print(f"  {t.tag}")
        # print(f"Compiler {c.__class__.__name__} produced {len(si)} stim instructions")
        # print(si)

        stim_instructions.extend(si)
    for si in stim_instructions:
        print(si)
    circ = stim.Circuit("\n".join(stim_instructions))

    sampler = circ.compile_sampler()

    samples = sampler.sample(num_samples)

    return samples
