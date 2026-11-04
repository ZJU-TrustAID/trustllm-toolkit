"""Exercise a running real-model API and verify returned detector results.

Run from the repository root: python scripts/smoke_api.py --base-url http://127.0.0.1:8000
The server chooses its model. This script never substitutes fixture answers.
"""

import argparse
import json
import os
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--watermark', choices=['KGW', 'Unigram'], default='KGW')
    args = parser.parse_args()
    headers = {'Content-Type': 'application/json'}
    if os.getenv('MARKSEC_API_TOKEN'):
        headers['Authorization'] = 'Bearer ' + os.environ['MARKSEC_API_TOKEN']

    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(args.base_url.rstrip('/') + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)

    assert request('/health')['mode'] == 'live'
    prompt = 'Explain why passwords should be unique for each account.'
    result = request('/v1/run', {
        'prompt': prompt, 'watermark': args.watermark, 'attack': 'LLMP',
        'max_new_tokens': 32, 'seed': 42,
    })
    assert result['mode'] == 'live' and result['prompt'] == prompt
    assert result['provenance']['backend'] == 'transformers'
    assert len(result['variants']) == 2
    for variant in result['variants']:
        assert variant['text']
        assert ''.join(token['text'] for token in variant['tokens']) == variant['text']
        detected = request('/v1/detect', {'text': variant['text'], 'watermark': args.watermark})
        assert detected['variant']['detection'] == variant['detection']
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
