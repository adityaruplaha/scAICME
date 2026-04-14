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
    majority_fraction : float, default 0.66
        The fraction of valid (known) votes required to assign a consensus label.
        - $0.51$ = Simple majority
        - $0.66$ = Supermajority (e.g., 2 out of 3)
        - $1.00$ = Unanimous agreement required
    unknown_label : str, default 'unknown'
        The string used to denote an unlabeled or abstained cell.
    """

    def __init__(
        self,
        keys: List[str],
        majority_fraction: float = 0.66,
        unknown_label: str = "unknown",
        **kwargs,
    ):
        if not keys:
            raise ValueError("Must provide at least one key for consensus voting.")

        self.keys = keys
        self.majority_fraction = majority_fraction
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
            fraction = top_count / len(valid_votes)
            if fraction >= self.majority_fraction:
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
            },
        )
