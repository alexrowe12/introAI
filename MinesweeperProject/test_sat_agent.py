from __future__ import annotations

import unittest

from minesweeper_tournament import Action, Observation, RoundContext
from sat_agent import StrategicSATAgent, _PYSAT_AVAILABLE


def _observation(board, legal_positions):
    return Observation(
        round_index=1,
        turn_index=0,
        board=tuple(tuple(row) for row in board),
        legal_actions=tuple(Action(row=row, col=col) for row, col in sorted(legal_positions)),
        scores={"sat": 0},
        active_agents=("sat",),
        disqualified_agents=(),
    )


class StrategicSATAgentTests(unittest.TestCase):
    def test_returns_forced_safe_move(self):
        agent = StrategicSATAgent(seed=1)
        agent.new_round(RoundContext(1, 2, 1, ("sat",), 1))
        observation = _observation(
            (
                ("1", "?", "?"),
                (" ", "1", "?"),
            ),
            {(0, 1), (0, 2), (1, 2)},
        )

        action = agent.choose_action(observation)

        self.assertIn((action.row, action.col), {(0, 2), (1, 2)})

    def test_avoids_forced_mine(self):
        agent = StrategicSATAgent(seed=1)
        agent.new_round(RoundContext(1, 2, 1, ("sat",), 1))
        observation = _observation(
            (
                ("1", "?", "?"),
                (" ", "1", "?"),
            ),
            {(0, 1), (0, 2), (1, 2)},
        )

        action = agent.choose_action(observation)

        self.assertNotEqual(action, Action(row=0, col=1))
        self.assertIn((action.row, action.col), {(0, 2), (1, 2)})

    @unittest.skipUnless(_PYSAT_AVAILABLE, "PySAT is required for global SAT inference")
    def test_uses_global_mine_count_to_find_safe_move(self):
        agent = StrategicSATAgent(seed=1)
        agent.new_round(RoundContext(1, 3, 1, ("sat",), 1))
        observation = _observation(
            (
                ("1", "?", "?"),
                ("1", "?", "?"),
                (" ", " ", "?"),
            ),
            {(0, 1), (0, 2), (1, 1), (1, 2), (2, 2)},
        )

        action = agent.choose_action(observation)

        self.assertIn((action.row, action.col), {(0, 2), (1, 2), (2, 2)})

    def test_empty_board_returns_legal_action(self):
        agent = StrategicSATAgent(seed=1)
        agent.new_round(RoundContext(1, 3, 1, ("sat",), 1))
        legal = {(row, col) for row in range(3) for col in range(3)}
        observation = _observation(
            (
                ("?", "?", "?"),
                ("?", "?", "?"),
                ("?", "?", "?"),
            ),
            legal,
        )

        action = agent.choose_action(observation)

        self.assertIn((action.row, action.col), legal)


if __name__ == "__main__":
    unittest.main()
