"""
This is used by participants to serve their policy. 

Participants may need to modify this file to adapt to their own policy. for example, loading multiple model ckpts.
"""

from challenge_inteface.server import PolicyServer
from challenge_inteface.policy import DummyPolicy
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a policy for the CVPR challenge.")

    parser.add_argument(
        "--host",
        default="0.0.0.0",
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