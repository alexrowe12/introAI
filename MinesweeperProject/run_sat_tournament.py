from __future__ import annotations

from functools import partial

from minesweeper_tournament import (
    LocalInferenceAgent,
    RandomAgent,
    RowMajorAgent,
    TournamentConfig,
    TournamentRunner,
)
from sat_agent import StrategicSATAgent


def main() -> None:
    runner = TournamentRunner(
        agent_factories={
            "sat": partial(StrategicSATAgent, seed=7, name="sat"),
            "local": partial(LocalInferenceAgent, seed=3, name="local"),
            "random": partial(RandomAgent, seed=1, name="random"),
            "row_major": partial(RowMajorAgent, name="row_major"),
        },
        config=TournamentConfig(
            board_size=5,
            num_mines=5,
            num_rounds=100,
            turn_timeout_seconds=0.5,
            random_seed=123,
        ),
    )
    result = runner.run()
    print(result.scores)


if __name__ == "__main__":
    main()
