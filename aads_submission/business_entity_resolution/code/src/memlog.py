import sys
import resource


def log_memory(tag: str) -> None:
    """
    Prints the process's peak resident-set size (RSS) so far. This exists
    because the real dataset (~12.5M rows train, ~11.7M test) is too large to
    run locally on a 16GB machine -- static reasoning about where memory
    peaks might be isn't a substitute for an actual measurement, so this
    logs real numbers at each checkpoint the first time this runs on
    SageMaker, instead of guessing.

    ru_maxrss is already a HIGH-WATER MARK for the whole process, not a
    point-in-time reading, so calling this at several checkpoints still
    reports "the worst it's been so far," which is what matters for an OOM
    risk assessment.

    Units differ by platform: Linux (SageMaker's containers) reports KB,
    macOS reports bytes -- handled here so this gives a sane number in both
    a local smoke test and a real SageMaker run.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 1024 if sys.platform != 'darwin' else peak / (1024 * 1024)
    print(f"  [mem] peak RSS so far ({tag}): {peak_mb:.1f} MB")
