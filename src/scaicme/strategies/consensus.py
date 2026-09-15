from collections import Counter
from typing import List

from anndata import AnnData

from .base import BaseLabelingStrategy, LabelingResult


class ConsensusVoting(BaseLabelingStrategy):
    """
    Aggregates multiple label vectors to generate high-confidence consensus seeds.

    This strategy compares the predictions of independent weak-labeling algorithms.
    It assigns a definitive label to a cell only if a specified fraction of the
    valid voters agree. Votes for the `unknown_label` are ignored (i.e., they
    do not count against the majority fraction).

    Parameters
    ----------
    keys : List[str]
        A list of column names in `adata.obs` containing the labels to aggregate.
    majority_fraction : float | None, default 0.66
        The agreement fraction required to assign a consensus label.
        - $0.51$ = Simple majority
        - $0.66$ = Supermajority (e.g., 2 out of 3)
        - $1.00$ = Unanimous agreement required
        - ``None`` = Plurality: the most common valid vote always wins.
    fraction_of : {"valid", "all"}, default "valid"
        Denominator of the agreement fraction: the number of valid (non-unknown) votes
        for the cell, or the total number of voters in `keys`.
    unknown_label : str, default 'unknown'
        The string used to denote an unlabeled or abstained cell.
    """

    def __init__(
        self,
        keys: List[str],
        majority_fraction: float | None = 0.66,
        fraction_of: str = "valid",
        unknown_label: str = "unknown",
        **kwargs,
    ):
        if not keys:
            raise ValueError("Must provide at least one key for consensus voting.")
        if fraction_of not in ("valid", "all"):
            raise ValueError("fraction_of must be 'valid' or 'all'.")

        self.keys = keys
        self.majority_fraction = majority_fraction
        self.fraction_of = fraction_of
        self.unknown_label = unknown_label

    @property
    def name(self) -> str:
        return "consensus_seeds"

    def execute_on(self, adata: AnnData) -> LabelingResult:
        # 1. Validate inputs
        missing_keys = [k for k in self.keys if k not in adata.obs.columns]
        if missing_keys:
            raise ValueError(f"The following keys were not found in adata.obs: {missing_keys}")

        # Extract the voting block
        votes_df = adata.obs[self.keys].astype(str)
        n_voters = len(self.keys)

        # 2. Voting Logic
        # For ~100k cells and ~4 voters, apply with a row-wise parser is highly efficient.
        def get_consensus(row):
            # Filter out abstentions ("unknown")
            valid_votes = [v for v in row if v != self.unknown_label]

            # If all strategies abstained, the consensus is unknown
            if not valid_votes:
                return self.unknown_label, 0.0, 0

            # Count the votes
            counts = Counter(valid_votes)
            top_label, top_count = counts.most_common(1)[0]

            # Check if the winner meets the required supermajority
            denominator = len(valid_votes) if self.fraction_of == "valid" else n_voters
            fraction = top_count / denominator
            if self.majority_fraction is None or fraction >= self.majority_fraction:
                return top_label, fraction, len(valid_votes)

            return self.unknown_label, fraction, len(valid_votes)

        # Apply row-wise
        results = votes_df.apply(get_consensus, axis=1, result_type="expand")

        # 3. Parse outputs
        final_labels = results[0]
        agreement_frac = results[1]
        valid_voters_count = results[2]

        is_confident = final_labels != self.unknown_label

        # 4. Return Rich DTO
        return LabelingResult(
            adata=adata,
            strategy=self,
            labels=final_labels,
            obs={
                "agreement_fraction": agreement_frac,
                "valid_voters": valid_voters_count,
                "is_confident": is_confident,
            },
            uns={
                "input_keys": self.keys,
                "fraction_assigned": float(is_confident.mean()),
                "majority_threshold": self.majority_fraction,
                "fraction_of": self.fraction_of,
            },
        )
