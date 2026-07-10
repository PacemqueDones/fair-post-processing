import numpy as np
import torch


class PairSampler:
    """Seleciona pares não ordenados de indivíduos.

    Se o número total de pares for menor ou igual a
    max_pairs, todos os pares são retornados.

    Caso contrário, max_pairs pares distintos são
    amostrados uniformemente.
    """

    def __init__(
        self,
        max_pairs=None,
        random_state=31,
    ):
        if (
            max_pairs is not None
            and max_pairs < 1
        ):
            raise ValueError(
                "max_pairs deve ser None ou "
                "um inteiro maior ou igual a 1."
            )

        self.max_pairs = (
            None
            if max_pairs is None
            else int(max_pairs)
        )

        self.random_state = random_state

    @staticmethod
    def num_total_pairs(num_samples):
        return (
            num_samples
            * (num_samples - 1)
            // 2
        )

    def _all_pairs(self, num_samples):
        source, target = np.triu_indices(
            num_samples,
            k=1,
        )

        return np.stack(
            [source, target],
            axis=0,
        )

    def _sample_pairs(
        self,
        num_samples,
        num_pairs,
        random_state,
    ):
        rng = np.random.default_rng(
            random_state
        )

        selected_pairs = np.empty(
            0,
            dtype=np.int64,
        )

        while selected_pairs.size < num_pairs:
            remaining = (
                num_pairs
                - selected_pairs.size
            )

            batch_size = max(
                remaining * 2,
                10_000,
            )

            first = rng.integers(
                0,
                num_samples,
                size=batch_size,
                dtype=np.int64,
            )

            second = rng.integers(
                0,
                num_samples,
                size=batch_size,
                dtype=np.int64,
            )

            valid = first != second

            first = first[valid]
            second = second[valid]

            source = np.minimum(
                first,
                second,
            )

            target = np.maximum(
                first,
                second,
            )

            encoded_pairs = (
                source * num_samples
                + target
            )

            selected_pairs = np.unique(
                np.concatenate(
                    [
                        selected_pairs,
                        encoded_pairs,
                    ]
                )
            )

            if selected_pairs.size > num_pairs:
                selected_pairs = rng.choice(
                    selected_pairs,
                    size=num_pairs,
                    replace=False,
                )

        source = (
            selected_pairs
            // num_samples
        )

        target = (
            selected_pairs
            % num_samples
        )

        return np.stack(
            [source, target],
            axis=0,
        )

    def sample(
        self,
        num_samples,
        random_state=None,
    ):
        if num_samples < 2:
            raise ValueError(
                "São necessárias pelo menos "
                "duas amostras."
            )

        total_pairs = self.num_total_pairs(
            num_samples
        )

        if self.max_pairs is None:
            num_pairs = total_pairs
        else:
            num_pairs = min(
                total_pairs,
                self.max_pairs,
            )

        if num_pairs == total_pairs:
            pair_index = self._all_pairs(
                num_samples
            )
        else:
            seed = (
                self.random_state
                if random_state is None
                else random_state
            )

            pair_index = self._sample_pairs(
                num_samples=num_samples,
                num_pairs=num_pairs,
                random_state=seed,
            )

        return torch.as_tensor(
            pair_index,
            dtype=torch.long,
        )