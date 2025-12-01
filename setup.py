import argparse
import datetime
import sys
from datetime import date
from pathlib import Path

from setuptools import find_packages, setup

# update this version when a new official pypi release is made
__version__ = "0.1.0"


def get_package_version():
    return __version__


def get_nightly_version():
    today = date.today()
    now = datetime.datetime.now()
    timing = f"{now.hour:02d}{now.minute:02d}"
    return f"{today.year}.{today.month}.{today.day}.{timing}"


def get_python_version():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def get_dependencies():
    install_requires = [
        "mani_skill>=3.0.0b21",
    ]
    return install_requires


def parse_args(argv):
    parser = argparse.ArgumentParser(description="HistoryBench setup.py configuration")
    parser.add_argument(
        "--package_name",
        type=str,
        default="historybench",
        choices=["historybench", "historybench-nightly"],
        help="the name of this output wheel. Should be either 'historybench' or 'historybench_nightly'",
    )
    return parser.parse_known_args(argv)


def main(argv):

    args, unknown = parse_args(argv)
    name = args.package_name

    version = get_package_version()

    sys.argv = [sys.argv[0]] + unknown
    print(sys.argv)
    setup(
        name=name,
        version=version,
        description="HistoryBench",
        python_requires=">=3.9",
        setup_requires=["setuptools>=62.3.0"],
        install_requires=get_dependencies(),
        packages=find_packages(include=["historybench*"]),
    )


if __name__ == "__main__":
    main(sys.argv[1:])