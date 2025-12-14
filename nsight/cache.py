import atexit
import glob
import hashlib
import os
from pathlib import Path

import pandas as pd


def _cache_dir_path() -> Path:
    """
    Return the cache directory for generated files in global ncu
    profile cache manager.

    Returns:
        Path to the cache directory.
    """
    # Get cache directory
    _cache_dir = os.environ.get("NSIGHT_PYTHON_CACHE_DIR")
    if _cache_dir is None:
        _cache_dir = Path.home() / ".nsight-python"
    else:
        _cache_dir = Path(_cache_dir)

    # Get absolute path
    _cache_dir = _cache_dir.absolute()

    # Create directory if it doesn't exist
    if not _cache_dir.exists():
        _cache_dir.mkdir(parents=True, exist_ok=True)

    return _cache_dir


class GlobalNCUProfileCache:
    """
    A global ncu profile cache manager with singleton pattern.
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        cache_dir = _cache_dir_path()

        # If instance already exists
        if cls._instance is not None:
            # Check if trying to change directory
            if cls._instance._active_cache_dir != cache_dir:
                raise RuntimeError(
                    f"Cache directory already set to {cls._instance._active_cache_dir}, "
                    f"cannot change to {cache_dir}"
                )
            return cls._instance

        # Validate directory
        if not cache_dir.is_dir():
            raise ValueError(
                f"Cache directory {cache_dir} does not exist or is not a directory"
            )

        # Create new instance
        instance = super().__new__(cls)
        cls._instance = instance

        return instance

    def __init__(self):
        # Prevent reinitialization
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        # Global shared profile ID
        self._global_ncu_profile_id = 0
        # Currently active cache directory
        self._active_cache_dir = _cache_dir_path()

        # Register cleanup function
        atexit.register(self._cleanup_cache_files)

    # ===== ID Management Methods =====

    def _increment_profile_id(self) -> int:
        """Increment and return a new profile ID."""
        self._global_ncu_profile_id += 1
        return self._global_ncu_profile_id

    # ===== Cache Management Methods =====

    def _cache_file_path(self, func_name: str) -> Path:
        """
        Construct cache file path for a given function name and profile ID.

        Args:
            func_name: Function name.

        Returns:
            Cache file path.
        """
        # Get current global profile id
        profile_id = self._global_ncu_profile_id

        # Generate cache key
        cache_key = hashlib.md5(f"{func_name}_{profile_id}".encode()).hexdigest()[:8]

        # Create cache filename
        cache_filename = f"{func_name}_pid{profile_id}_{cache_key}.pkl"
        cache_path = self._active_cache_dir / cache_filename

        return cache_path

    def save_profile_result(
        self,
        func_name: str,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> None:
        """
        Save profiling results to disk in the active cache directory.

        Args:
            func_name: Function name.
            df: DataFrame containing profiling results.
            verbose: Whether to print verbose messages.

        Raises:
            RuntimeError: If there is an error saving the cache.
        """
        if "NSPY_NCU_PROFILE" in os.environ:
            raise RuntimeError("Cache saving can only be used in the main process")

        cache_path = self._cache_file_path(func_name)

        try:
            # Save DataFrame to disk
            df.to_pickle(cache_path)

            if verbose:
                print(
                    f"[NSIGHT-PYTHON] Saved profile results for {func_name} "
                    f"(Global NCU ID: {self._global_ncu_profile_id}) to {cache_path}"
                )

            # Increment profile ID for next save
            self._increment_profile_id()

        except Exception as e:
            raise RuntimeError(f"Failed to save cache for {func_name}: {e}")

    def load_profile_result(
        self, func_name: str, verbose: bool = False
    ) -> pd.DataFrame:
        """
        Load profiling results from the active cache directory.

        Args:
            func_name: Function name.
            verbose: Whether to print verbose messages.

        Returns:
            DataFrame containing profiling results.

        Raises:
            RuntimeError: If not in ncu subprocess or there is an error loading the cache.
            FileNotFoundError: If the cache does not exist.
        """
        if "NSPY_NCU_PROFILE" not in os.environ:
            raise RuntimeError(
                "Cache loading can only be used in ncu profile subprocess"
            )

        cache_path = self._cache_file_path(func_name)

        # Try to load from the file
        if cache_path.exists():
            try:
                df = pd.read_pickle(cache_path)

                if verbose:
                    print(
                        f"[NSIGHT-PYTHON] Loaded cached results for {func_name} "
                        f"(Global NCU ID: {self._global_ncu_profile_id}) from {cache_path}"
                    )

                # Increment profile ID for next save
                self._increment_profile_id()

                return df
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load cache for {func_name} from {cache_path}: {e}"
                ) from e
        else:
            raise FileNotFoundError(f"Cache for {func_name} does not exist")

    # ===== Delete Cache Methods =====

    def _cleanup_cache_files(self) -> None:
        """
        Delete all cache files in the active cache directory.
        """
        cache_pattern = str(self._active_cache_dir / "*.pkl")
        for file in glob.glob(cache_pattern):
            try:
                os.remove(file)
            except Exception as e:
                raise RuntimeError(f"Failed to remove cache file {file}: {e}")
