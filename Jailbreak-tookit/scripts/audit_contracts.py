"""Print current public contracts. Historical findings remain in docs/."""

import json

from textbook_toolkit import jailbreak


def main():
    print(
        json.dumps(
            {name: jailbreak.describe(name) for name in jailbreak.list_methods()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
