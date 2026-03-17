"""
This is used by participants to serve their policy.
"""

from remote_evaluation.server import PolicyServer
from remote_evaluation.policy import DummyPolicy
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a policy for the CVPR challenge.")

    parser.add_argument(
        "--host",
        default="141.212.115.116",
        help="Host/IP to bind the policy server (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the policy server (default: %(default)s).",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    policy = DummyPolicy()
    server = PolicyServer(policy, host=args.host, port=args.port)
    server.serve_forever()

if __name__ == "__main__":
    main()