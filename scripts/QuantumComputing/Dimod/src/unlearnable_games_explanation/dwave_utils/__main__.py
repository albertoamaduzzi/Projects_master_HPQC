"""python -m dwave_utils <command> [options]: the D-Wave commands of main_simulation.py.

Commands: feasibility, noise-fit, budget, calibrate, run, report, compare, shim, coupling-scan.
Options are those of `python main_simulation.py <command> --help`.
"""
import sys

from . import COMMANDS


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        try:
            print(__doc__)
        except BrokenPipeError:  # pragma: no cover  (output piped into head)
            pass
        if argv and argv[0] not in ("-h", "--help"):
            sys.exit(f"unknown command {argv[0]!r}; choose one of {', '.join(COMMANDS)}")
        return
    import main_simulation as ms
    ms.main(argv)


if __name__ == "__main__":
    main()
