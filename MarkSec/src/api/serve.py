"""Serve the toolkit API for the TrustLLM hub chapter workbench.

The hub renders this chapter's ``server.config`` template into a JSON file
(model entry, device, loopback port) and passes its path via ``--config``.
This launcher translates it into the ``MARKSEC_*`` environment variables read
by :class:`src.api.engine.Settings.from_env`, then runs the same app as
``python -m src.api``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description='Serve the MarkSec toolkit API for the TrustLLM hub.')
    parser.add_argument('--config', required=True, help='hub 渲染的前端服务配置（JSON）')
    args = parser.parse_args()
    load_dotenv()
    rendered = json.loads(Path(args.config).read_text(encoding='utf-8'))
    # Hub workbench settings win over inherited environment variables.
    if rendered.get('model'):
        os.environ['MARKSEC_MODEL_ID'] = str(rendered['model'])
    if rendered.get('device'):
        os.environ['MARKSEC_DEVICE'] = str(rendered['device'])
    web = rendered.get('web') or {}
    uvicorn.run('src.api.app:app', host=str(web.get('host') or '127.0.0.1'),
                port=int(web.get('port') or 8000), workers=1)


if __name__ == '__main__':
    main()
