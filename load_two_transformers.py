import importlib
import sys
from pathlib import Path


def load_fork(fork_root: str, alias: str = "transformers_fork"):
    """Load a forked copy of the 'transformers' package *alongside* the baseline
    one already installed with pip.

    Parameters
    ----------
    fork_root : str | Path
        Path that contains the fork's 'transformers/' package directory.
    alias : str
        The key under which the fork will be exposed in sys.modules.

    Returns
    -------
    (baseline_module, fork_module)

    """
    fork_root = Path(fork_root).expanduser().resolve()

    # Hugging Face repo layout puts the package in src/transformers.
    candidate_src = fork_root / "src"
    if (candidate_src / "transformers").is_dir():
        fork_root = candidate_src
    elif not (fork_root / "transformers").is_dir():
        raise RuntimeError(
            f"Cannot find 'transformers' package inside {fork_root}. "
            "Did you pass the wrong path? Expected either <repo>/src "
            "or a folder containing transformers/__init__.py."
        )

    # 1️⃣ import baseline first so we keep a handle to it
    import transformers as baseline

    # 2️⃣ unload the baseline package (and all its submodules)
    original_modules = {}
    for name, mod in sys.modules.items():
        if name == "transformers" or name.startswith("transformers."):
            original_modules[name] = mod
    for name in original_modules:
        del sys.modules[name]

    # 3️⃣ temporarily put fork first on sys.path, import it, then pop the path
    sys.path.insert(0, str(fork_root))
    try:
        fork = importlib.import_module("transformers")
    finally:
        sys.path.remove(str(fork_root))

    # 4️⃣ restore the baseline package so that other code that imports
    # transformers gets the baseline
    sys.modules.update(original_modules)

    # 5️⃣ expose the fork under its alias so you can `import transformers_fork`
    sys.modules[alias] = fork
    return baseline, fork


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "transformers-fork"
    base, fork = load_fork(path)

    print("baseline:", base.__version__, "at", base.__file__)
    print("fork:    ", fork.__version__, "at", fork.__file__)
