"""Exercise the local HTTP service and real toolkit/model calls (no mocks)."""
import json
import argparse
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:4173"
TIMEOUT = 600


def call(path, payload=None, headers=None):
    request = Request(BASE + path,
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers=headers or {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def experiment(method, **kwargs):
    deadline = time.monotonic() + TIMEOUT
    while call('/api/health')[1]['busy']:
        assert time.monotonic() < deadline, "Existing experiment timed out"
        time.sleep(.5)
    payload = {"method": method, "query": "Explain photosynthesis in one sentence.",
               "target": "Plants use sunlight", "steps": 3, **kwargs}
    status, job = call('/api/runs', payload)
    assert status == 202, (status, job)
    # Single-model execution must reject overlapping jobs.
    assert call('/api/runs', payload)[0] == 409
    while time.monotonic() < deadline:
        status, state = call('/api/runs/' + job['id'])
        assert status == 200
        if state['status'] != 'running':
            break
        time.sleep(.5)
    assert state['status'] == 'completed', state
    result = state['result']
    assert result['method'] == method and result['query'] == payload['query']
    assert result['baseline_response'] and result['response'] and result['messages']
    assert not {'source', 'backend', 'input_ids', 'config'} & result.keys()
    if method in ('gcg', 'autodan', 'cold_attack'):
        assert len(result['trace']) >= 2, result
        assert isinstance(result['final_loss'], float)
    print(json.dumps({"method": method, "mode": kwargs.get('mode'),
                      "seconds": result['elapsed'], "trace_points": len(result['trace']),
                      "loss": [result.get('initial_loss'), result.get('final_loss')]}, ensure_ascii=False), flush=True)
    return result


def main():
    global BASE, TIMEOUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--timeout', type=int, default=TIMEOUT)
    parser.add_argument('--query', default='Explain photosynthesis in one sentence.')
    parser.add_argument('--target', default='Plants use sunlight')
    parser.add_argument('--steps', type=int, choices=(3, 6, 10), default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    BASE, TIMEOUT = args.base.rstrip('/'), args.timeout
    health = call('/api/health')[1]
    assert health['ready'], health
    for path in ('/assets/sources.json', '/assets/textbook-toolkit-source.zip',
                 '/assets/examples.json', '/server.py', '/private/legacy-assets/sources.json',
                 '/private/service-logs/stdout.log', '/private/service-logs/stderr.log',
                 '/service/com.textbook-toolkit.lab.plist',
                 '/../server.py', '/%2e%2e/server.py'):
        assert call(path)[0] == 404, path
    for payload in ({}, {"method": "gcg", "query": ""},
                    {"method": "gcg", "query": "x", "steps": True},
                    {"method": "gcg", "query": "x", "steps": 999},
                    {"method": "gcg", "query": "x", "seed": "{invalid}"}):
        assert call('/api/runs', payload)[0] == 400, payload
    assert call('/api/runs', {"method": "gcg", "query": "x"},
                {"Content-Type": "application/json", "Origin": "https://example.com"})[0] == 403
    print('File exposure, validation, and origin checks passed.', flush=True)
    methods = ('template', 'role_play', 'task_disguise', 'rule_rewrite', 'context',
               'base64', 'caesar', 'red_teaming', 'gcg', 'autodan', 'cold_attack')
    options = dict(query=args.query, target=args.target, steps=args.steps)
    results = [experiment(method, **options) for method in methods]
    for mode in ('control', 'paraphrase'):
        results.append(experiment('cold_attack', mode=mode, **options))
    changed = experiment('template', query='Explain how rainbows form in one sentence.')
    assert changed['baseline_response'] != results[0]['baseline_response']
    assert changed['messages'] != results[0]['messages']
    results.append(changed)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump({'health': health, 'results': results}, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print('All 11 methods, 3 COLD modes, and changed-input inference passed.', flush=True)


if __name__ == '__main__':
    main()
