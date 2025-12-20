# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import functools
import hashlib
import os
from pathlib import Path
from typing import ClassVar

import pandas as pd


class GlobalNCUProfileCache:
    """
    A global singleton manager for NCU profile caching with process isolation.

    This class provides a singleton implementation for managing cached profiling
    results from NVIDIA Nsight Compute. The cache stores profiling results as
    pandas DataFrames in pickle files, allowing for persistence across multiple
    profiling sessions or subprocesses.

    Key Features:
    - Main process isolation: Each main profiling process uses its own cache
      directory (identified by the main process ID), preventing cross-process
      interference
    - Environment variable NSPY_NCU_MAIN_PROCESS passes the main process ID
    - Singleton pattern: Ensures only one cache instance per process
    - Environment variable NSIGHT_PYTHON_CACHE_DIR controls cache location
    - Automatic cache file cleanup on program termination (process-specific)
    - MD5-based cache key generation for unique file naming

    Process Safety:
        The cache is designed to be process-safe, meaning multiple processes
        can run concurrently without interfering with each other's cache files.
        Each process creates files in its own subdirectory and only cleans up
        its own files on exit.

    Thread Safety Considerations:
        The current implementation does not require thread safety for its
        intended use case. In typical usage scenarios, cache operations are
        called sequentially from a single thread within each process. If
        multi-threaded usage becomes necessary in the future, appropriate
        synchronization mechanisms can be added.

    The cache is designed to work in a specific workflow:
    1. Main process profiles functions and saves results to cache
    2. Subprocess (ncu profile) loads cached results for analysis
    3. Each process automatically cleans up only its own cache files on exit

    Example usage::

        cache = GlobalNCUProfileCache()
        # Save profiling results from main process
        cache.save_profile_result("my_function", dataframe)
        # Later, in ncu subprocess:
        results = cache.load_profile_result("my_function")

    Note:
        This class requires proper environment setup:
        - NSIGHT_PYTHON_CACHE_DIR: Optional, specifies custom cache directory
        - NSPY_NCU_PROFILE: Must be set in ncu subprocess for loading
        - NSPY_NCU_MAIN_PID: Must be set in ncu subprocess for identifing main process

    Raises:
        RuntimeError: If trying to change cache directory after initialization,
            or if cache operations are called from wrong process context.
        ValueError: If specified cache directory doesn't exist.
        FileNotFoundError: If trying to load non-existent cache.
    """

    # Singleton instance (per process)
    _instance: ClassVar[GlobalNCUProfileCache | None] = None

    def __new__(cls) -> GlobalNCUProfileCache:
        """
        Create or retrieve the singleton instance.

        This method implements the singleton pattern. It ensures that only
        one instance of GlobalNCUProfileCache exists per process. The cache
        directory is determined at first instantiation and cannot be changed
        subsequently.
        """
        cache_dir = cls._cache_dir_path()

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

        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the singleton instance with process isolation.

        Each process gets:
        - Its own main process ID for isolation
        - Process-specific subdirectory for cache files
        - Independent profile ID counter
        - Cleanup handler for its own files only

        Raises:
            RuntimeError: If NSPY_NCU_MAIN_PID environment variable is not
                set or is not a valid integer.
        """
        # Prevent reinitialization
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        # Get current main process ID for isolation
        nspy_ncu_main_pid = os.environ.get("NSPY_NCU_MAIN_PID")
        if nspy_ncu_main_pid is None:
            raise RuntimeError(
                "NSPY_NCU_MAIN_PID environment variable is not set. "
                "This is required for cache process isolation."
            )

        try:
            self._process_id = int(nspy_ncu_main_pid)
        except ValueError as e:
            raise RuntimeError(
                f"Invalid NSPY_NCU_MAIN_PID value: '{nspy_ncu_main_pid}'. "
                "Must be a valid integer."
            ) from e

        # Global shared profile ID
        self._global_ncu_profile_id = 0

        # Currently active cache directory
        self._active_cache_dir = self._cache_dir_path()

        # Process-specific subdirectory
        self._process_dir = self._active_cache_dir / f"pid_{self._process_id}"
        self._process_dir.mkdir(exist_ok=True)

        # Register cleanup function
        atexit.register(self._cleanup_process_cache_files)

    @classmethod
    @functools.lru_cache(maxsize=1)
    def _cache_dir_path(cls) -> Path:
        """
        Determine and return the cache directory path.

        This method uses LRU caching to ensure the directory path is computed
        only once. The cache directory is determined by:
        1. NSIGHT_PYTHON_CACHE_DIR environment variable (if set)
        2. Default: ~/.nsight-python

        If the directory doesn't exist, it will be created automatically.

        Returns:
            Path: Absolute path to the cache directory.

        Note:
            The @functools.lru_cache decorator ensures this method's result
            is cached, preventing repeated filesystem operations.
        """
        # Get cache directory
        _env_cache_dir = os.environ.get("NSIGHT_PYTHON_CACHE_DIR")
        if _env_cache_dir is None:
            _cache_dir = Path.home() / Path(".nsight-python")
        else:
            _cache_dir = Path(_env_cache_dir)

        # Get absolute path
        _cache_dir = _cache_dir.absolute()

        # Create directory if it doesn't exist
        if not _cache_dir.exists():
            _cache_dir.mkdir(parents=True, exist_ok=True)

        return _cache_dir

    def _increment_profile_id(self) -> int:
        """
        Increment and return the global profile ID.

        This method atomically increments the profile ID counter and returns
        the new value. Each profile operation gets a unique ID, ensuring
        cache file uniqueness.

        Returns:
            int: The new profile ID after incrementing.
        """
        self._global_ncu_profile_id += 1
        return self._global_ncu_profile_id

    def _construct_cache_file_path(self, func_name: str) -> Path:
        """
        Construct the cache file path for a given function.

        The cache filename is generated using:
        - Function name
        - Current profile ID (process-specific)
        - Process ID for isolation
        - MD5 hash of "funcname_profileid_pid" (first 8 characters)

        This ensures unique filenames across processes.

        Args:
            func_name: Name of the profiled function.

        Returns:
            Path: Full path to the cache file in process-specific directory.
        """
        # Get current global profile id
        profile_id = self._global_ncu_profile_id

        # Generate cache key
        cache_key = hashlib.md5(f"{func_name}_{profile_id}".encode()).hexdigest()[:8]

        # Create cache filename
        cache_filename = f"{func_name}_{profile_id}_{cache_key}.pkl"
        cache_path = self._process_dir / cache_filename

        return cache_path

    def save_profile_result(
        self,
        func_name: str,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> None:
        """
        Save profiling results to the cache.

        This method should only be called from the main profiling process.
        It saves a DataFrame to a pickle file and increments the profile ID
        for the next operation.

        Args:
            func_name: Name of the profiled function.
            df: DataFrame containing profiling results to cache.
            verbose: If True, print status messages to stdout.

        Raises:
            RuntimeError: If called from an ncu subprocess, or if file
                operations fail.
        """
        if "NSPY_NCU_PROFILE" in os.environ:
            raise RuntimeError("Cache saving can only be used in the main process")

        cache_path = self._construct_cache_file_path(func_name)

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
        Load profiling results from the cache.

        This method should only be called from an ncu profile subprocess.
        It loads a previously saved DataFrame from a pickle file and
        increments the profile ID for consistency.

        Args:
            func_name: Name of the function whose results to load.
            verbose: If True, print status messages to stdout.

        Returns:
            pd.DataFrame: The loaded profiling results.

        Raises:
            RuntimeError: If called from the main process, or if file
                operations fail.
            FileNotFoundError: If no cache file exists for the given
                function name and current profile ID.
        """
        if "NSPY_NCU_PROFILE" not in os.environ:
            raise RuntimeError(
                "Cache loading can only be used in ncu profile subprocess"
            )

        cache_path = self._construct_cache_file_path(func_name)

        # Try to load from the file
        if cache_path.exists():
            try:
                df = pd.read_pickle(cache_path)

                if verbose:
                    print(
                        f"[NSIGHT-PYTHON] Loaded cached results for {func_name} "
                        f"(Global NCU ID: {self._global_ncu_profile_id}) from {cache_path}"
                    )

                # Increment profile ID for next load
                self._increment_profile_id()

                return df
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load cache for {func_name} from {cache_path}: {e}"
                ) from e
        else:
            raise FileNotFoundError(f"Cache for {func_name} does not exist")

    def _cleanup_process_cache_files(self) -> None:
        """
        Clean up all cache files for this process by removing its directory.

        This method is registered with atexit and automatically removes the
        entire process-specific cache directory when the process terminates.
        Since each process has its own isolated directory, this operation
        does not affect other processes.

        Note:
            This is a best-effort cleanup. If the program crashes or is
            terminated abruptly, the directory may remain on disk.
        """
        try:
            if self._process_dir.exists():
                # Remove directory and all its contents
                import shutil

                shutil.rmtree(self._process_dir, ignore_errors=True)
        except Exception as e:
            # Log but don't raise - atexit handlers shouldn't raise exceptions
            print(
                f"[NSIGHT-PYTHON] Warning: Failed to remove cache directory {self._process_dir}: {e}"
            )
