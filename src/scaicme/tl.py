import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple, Union, overload

from anndata import AnnData

from .strategies.base import BaseLabelingStrategy, LabelingResult

## Asynchronous interface for single strategy execution
## ====================================================


async def label_async(
    adata: AnnData, strategy: BaseLabelingStrategy, key_added: str | None = None
) -> Dict[str, LabelingResult]:
    """
    Asynchronously apply a single labeling strategy to the AnnData object.

    This function is atomic and non-blocking. It offloads the computation to a
    thread executor and writes the results back to the AnnData object on the
    main thread.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix to label.
    strategy : BaseLabelingStrategy
        A single labeling strategy to apply.
    key_added : str | None, optional
        Key under which to add the labeling results in `adata.obs`. If None, an
        auto-generated key based on strategy name will be used.

    Returns
    -------
    dict[str, LabelingResult]
        A dictionary mapping the key under which results were stored to the
        LabelingResult object containing the labeling outcomes.

    Raises
    ------
    TypeError
        If `strategy` is a list, tuple, or dict. Use asyncio.gather() to run
        multiple strategies in parallel.

    Notes
    -----
    This function is designed to be used within an async context. Results are
    written to the AnnData object in-place on the main thread after computation
    completes on a background thread executor.

    Examples
    --------
    >>> import asyncio
    >>> from anndata import AnnData
    >>> adata = AnnData(...)
    >>> strategy = BaseLabelingStrategy(...)
    >>> result = await label_async(adata, strategy, key_added="labels")
    >>> # result is a dict: {"labels": LabelingResult}
    """

    if isinstance(strategy, (list, tuple, dict)):
        raise TypeError(
            "label_async only accepts a single strategy. "
            "To run a batch, use: await asyncio.gather(*[label_async(adata, s) for s in strategies])"
        )

    loop = asyncio.get_running_loop()

    # 1. Run (Off-Main Thread)
    # We pass the bound method directly to the executor.
    result = await loop.run_in_executor(None, strategy.execute_on, adata)

    # 2. Write (Main Thread)
    return {result.write_in(key=key_added): result}


## Synchronous interface for single and batch strategy execution
## =============================================================


@overload
def label(
    adata: AnnData,
    strategies: BaseLabelingStrategy,
    key_added: str | None = None,
    n_jobs: int = 4,
) -> Dict[str, LabelingResult]: ...


@overload
def label(
    adata: AnnData, strategies: Sequence[BaseLabelingStrategy], n_jobs: int = 4
) -> Dict[str, LabelingResult]: ...


@overload
def label(
    adata: AnnData, strategies: Dict[str, BaseLabelingStrategy], n_jobs: int = 4
) -> Dict[str, LabelingResult]: ...


def label(
    adata: AnnData,
    strategies: Union[
        BaseLabelingStrategy, Sequence[BaseLabelingStrategy], Dict[str, BaseLabelingStrategy]
    ],
    key_added: str | None = None,
    n_jobs: int = 4,
) -> Dict[str, LabelingResult]:
    """
    Apply one or more labeling strategies to the AnnData object.

    Handles single-strategy execution, list-based batch execution (auto-named),
    and dictionary-based batch execution (custom-named).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    strategies : BaseLabelingStrategy, Sequence, or Dict
        - Single Strategy: Executed sequentially; key is taken from `key_added` or auto-generated.
        - Sequence[Strategy]: Executed in parallel; keys are auto-generated based on strategy name.
        - Dict[str, Strategy]: Executed in parallel; keys are taken directly from the dictionary.
    key_added : str | None, optional
        Key for `adata.obs`. Used ONLY if `strategies` is a single object.
    n_jobs : int, default 4
        Number of threads to use when running a batch.

    Returns
    -------
    Dict[str, LabelingResult]
        A dictionary mapping the keys under which results were saved in the AnnData object to their corresponding LabelingResult objects.

    Examples
    --------
    #### 1. Single Strategy (Sequential)
    >>> strat = AdaptiveThresholding(markers=my_markers, quota=50)
    >>> results = icme.tl.label(adata, strategies=strat, key_added="seeds_fixed")
    >>> # Returns: {"seeds_fixed": LabelingResult}

    #### 2. List Batch (Parallel, Auto-named)
    >>> # Useful for parameter sweeps where names don't matter yet
    >>> strategies = [AdaptiveThresholding(markers, quota=q) for q in [10, 50, 100]]
    >>> results = icme.tl.label(adata, strategies=strategies)
    >>> # Returns: {'_icme_label_adaptive': LabelingResult, '_icme_label_adaptive_1': LabelingResult, '_icme_label_adaptive_2': LabelingResult}

    #### 3. Dictionary Batch (Parallel, Custom-named)
    >>> # Useful for defining semantic variations
    >>> batch = {
    ...     "seeds_strict": AdaptiveThresholding(markers, quota=10),
    ...     "seeds_loose": AdaptiveThresholding(markers, quota=100)
    ... }
    >>> results = icme.tl.label(adata, strategies=batch)
    >>> # Returns: {'seeds_strict': LabelingResult, 'seeds_loose': LabelingResult}
    """

    # Batch processing for lists and dicts, in parallel using ThreadPoolExecutor
    if isinstance(strategies, (list, tuple, dict)):
        tasks: List[Tuple[BaseLabelingStrategy, str | None]] = []

        if isinstance(strategies, dict):
            # Dict: Key explicitly provided by user
            tasks = [(s, k) for k, s in strategies.items()]
        else:
            # List: Key is None (triggers auto-generation in result.write_in)
            tasks = [(s, None) for s in strategies]

        outputs: Dict[str, LabelingResult] = {}
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit for execution: $ (S, K) \mapsto (S, K, S(data)) $
            futures_to_metadata = {
                executor.submit(s.execute_on, adata): (s, target_key) for s, target_key in tasks
            }

            # Process in order of completion.
            # This allows us to write results back to the AnnData object as soon as they are ready,
            # without waiting for the entire batch to finish.
            for future in as_completed(futures_to_metadata.keys()):
                s, target_key = futures_to_metadata[future]
                try:
                    result = future.result()
                    # Safe sequential write on main thread
                    # Uses target_key if provided (Dict), else None (List -> Auto)
                    outputs[result.write_in(key=target_key)] = result
                    print(f"Strategy '{s.name}' completed successfully.")
                except Exception as e:
                    print(f"Strategy '{s.name}' failed: {e}")
                    print(f"Strategy details: {s}")
                    # In batch mode, we log failures but continue processing others

        # Mapping of keys to results for the entire batch
        return outputs

    # Single strategy execution
    result = strategies.execute_on(adata)
    return {result.write_in(key=key_added): result}
