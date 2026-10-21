"""Opponent inference settings and call order; no tensor or CUDA dependencies."""

OPPONENT_INFERENCE_MODES = ('stream_parallel', 'sequential', 'simple', 'stream_parallel_isolated', 'single_stream_deferred')


def validate_opponent_inference(mode):
    if mode not in OPPONENT_INFERENCE_MODES:
        raise ValueError(f'unknown opponent_inference: {mode!r}')
    return mode


def read_opponent_inference(runtime):
    return validate_opponent_inference(runtime.get('opponent_inference', 'stream_parallel'))


def enqueue_opponent_actions(opponents, pool, by_name, mode):
    """warm の全相手を旧順で投入する。例外時は呼出元が全対象を回収する。"""
    if mode not in ('stream_parallel_isolated', 'single_stream_deferred'):
        raise ValueError('この方式は一括投入の対象外')
    if getattr(opponents, '_action_ticket', None) is not None:
        raise ValueError('相手が未回収')
    ticket = dict(mode=mode, pool=pool, clock=tuple(pool.meta[:, 0]),
                  names=tuple((n, tuple(rs)) for n,rs in by_name.items()), complete=False)
    opponents._action_ticket = ticket
    streams = [opponents[n]._stream.cuda_stream for n in by_name]
    if mode == 'stream_parallel_isolated' and len(set(streams)) != len(streams):
        raise ValueError('isolated streams must be distinct')
    if mode == 'single_stream_deferred' and len(set(streams)) > 1:
        raise ValueError('deferred opponents must share one stream')
    for name,rows in by_name.items():
        opponents[name].enqueue(pool,rows)
    ticket['complete'] = True
    return ticket


def collect_opponent_actions(ticket, opponents, pool, by_name, actions, mode):
    if (ticket is not getattr(opponents, '_action_ticket', None) or ticket is None
            or not ticket['complete'] or pool is not ticket['pool'] or mode != ticket['mode']
            or tuple(pool.meta[:, 0]) != ticket['clock']
            or tuple((n, tuple(rs)) for n,rs in by_name.items()) != ticket['names']):
        raise ValueError('相手の回収対象が投入時と不一致')
    if mode == 'single_stream_deferred' and by_name:
        opponents[next(reversed(by_name))]._output_ready.synchronize()
        for name in by_name: opponents[name]._deferred_ready = True
    for name,rows in by_name.items():
        for r,a in zip(rows,opponents[name].collect(pool,rows)): actions[r] = a
    opponents._action_ticket = None


def fill_opponent_actions(opponents, pool, by_name, actions, mode='stream_parallel'):
    """Fill only opponent rows, retaining opponent and row insertion order."""
    validate_opponent_inference(mode)
    if mode in ('stream_parallel_isolated','single_stream_deferred'):
        # CPU 専用 sampler は CUDA の準備・event を持たない。
        for name,rows in by_name.items():
            if hasattr(opponents[name],'prepare'): opponents[name].prepare(pool,rows)
        streams = [opponents[name]._stream.cuda_stream for name in by_name if hasattr(opponents[name],'_stream')]
        if mode == 'stream_parallel_isolated':
            if len(set(streams)) != len(streams): raise ValueError('isolated streams must be distinct')
        else:
            if len(set(streams)) > 1: raise ValueError('deferred opponents must share one stream')
    if mode in ('stream_parallel','stream_parallel_isolated','single_stream_deferred'):
        for name,rows in by_name.items(): opponents[name].enqueue(pool,rows)
        if mode == 'single_stream_deferred' and streams:
            # 最後の相手の event は、それ以前の全相手の D2H 完了も表す。
            opponents[next(reversed(by_name))]._output_ready.synchronize()
            for name in by_name: opponents[name]._deferred_ready = True
        for name,rows in by_name.items():
            for r,a in zip(rows,opponents[name].collect(pool,rows)): actions[r]=a
    elif mode == 'sequential':
        for name,rows in by_name.items():
            opponents[name].enqueue(pool,rows)
            for r,a in zip(rows,opponents[name].collect(pool,rows)): actions[r]=a
    else:
        for name,rows in by_name.items():
            for r,a in zip(rows,opponents[name].act(pool,rows)): actions[r]=a
