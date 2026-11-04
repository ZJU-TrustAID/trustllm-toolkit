import argparse

import uvicorn
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description='Run the MarkSec toolkit HTTP backend.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    load_dotenv()
    uvicorn.run('src.api.app:app', host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
