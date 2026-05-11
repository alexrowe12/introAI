from __future__ import annotations

import random
from dataclasses import dataclass

from minesweeper_tournament import Action, MinesweeperAgent, Observation, RoundContext

try:  # PySAT is the intended path, but the fallback keeps the agent importable.
    from pysat.card import CardEnc, EncType
    from pysat.formula import IDPool
    from pysat.solvers import Solver

    _PYSAT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when python-sat is absent
    CardEnc = EncType = IDPool = Solver = None
    _PYSAT_AVAILABLE = False


Position = tuple[int, int]


@dataclass
class _SatModel:
    clauses: list[list[int]]
    var_for_position: dict[Position, int]
    position_for_var: dict[int, Position]
    constrained_positions: set[Position]
    known_mines: set[Position]
    consistent: bool


class StrategicSATAgent(MinesweeperAgent):
    """
    Minesweeper tournament agent using SAT inference.

    The agent proves safe tiles by checking whether the opposite assumption is
    unsatisfiable. If no safe tile is forced, it estimates mine probabilities
    from bounded SAT model enumeration and chooses the lowest-risk move.
    """

    def __init__(
        self,
        seed: int | None = None,
        name: str | None = None,
        max_models: int = 256,
    ):
        super().__init__(name=name)
        self._rng = random.Random(seed)
        self._seed = seed
        self._max_models = max_models
        self._known_mines: set[Position] = set()
        self._board_size: int | None = None
        self._num_mines: int | None = None

    def new_round(self, context: RoundContext) -> None:
        self._known_mines = set()
        self._board_size = context.board_size
        self._num_mines = context.num_mines
        self._rng.seed(None if self._seed is None else self._seed + context.round_index)

    def choose_action(self, observation: Observation) -> Action:
        legal_positions = {(action.row, action.col) for action in observation.legal_actions}
        if not legal_positions:
            return Action(row=0, col=0)

        self._known_mines = {pos for pos in self._known_mines if pos in legal_positions}

        if _PYSAT_AVAILABLE:
            safe_positions, risk_by_position = self._sat_inference(observation, legal_positions)
            safe_positions = safe_positions - self._known_mines
            if safe_positions:
                row, col = self._choose_safe_position(
                    observation.board,
                    safe_positions,
                    legal_positions,
                    risk_by_position,
                )
                return Action(row=row, col=col)

            guessable = legal_positions - self._known_mines
            if guessable:
                row, col = self._choose_guess_position(
                    observation.board,
                    guessable,
                    legal_positions,
                    risk_by_position,
                )
                return Action(row=row, col=col)

        safe_positions = self._local_safe_moves(observation.board, legal_positions)
        safe_positions = safe_positions - self._known_mines
        if safe_positions:
            row, col = self._choose_safe_position(
                observation.board,
                safe_positions,
                legal_positions,
                {},
            )
            return Action(row=row, col=col)

        fallback = sorted(legal_positions - self._known_mines)
        if not fallback:
            fallback = sorted(legal_positions)
        row, col = self._choose_guess_position(
            observation.board,
            set(fallback),
            legal_positions,
            {},
        )
        return Action(row=row, col=col)

    def _sat_inference(
        self,
        observation: Observation,
        legal_positions: set[Position],
    ) -> tuple[set[Position], dict[Position, float]]:
        model = self._build_sat_model(observation.board, legal_positions)
        if not model.consistent:
            self._known_mines = set()
            model = self._build_sat_model(observation.board, legal_positions)
            if not model.consistent:
                return set(), {}

        solver = self._new_solver(model.clauses)
        if solver is None:
            return set(), {}
        try:
            if not solver.solve():
                return set(), {}
        finally:
            solver.delete()

        safe_positions: set[Position] = set()
        newly_forced_mines: set[Position] = set()
        candidate_positions = sorted(
            pos for pos in model.var_for_position if pos not in model.known_mines
        )

        for pos in candidate_positions:
            var = model.var_for_position[pos]
            mine_possible = self._is_satisfiable(model.clauses, [var])
            safe_possible = self._is_satisfiable(model.clauses, [-var])
            if not mine_possible and safe_possible:
                safe_positions.add(pos)
            elif mine_possible and not safe_possible:
                newly_forced_mines.add(pos)

        if newly_forced_mines:
            self._known_mines.update(newly_forced_mines)

        risk_by_position = self._estimate_risk(model)
        for pos in newly_forced_mines:
            risk_by_position[pos] = 1.0
        for pos in safe_positions:
            risk_by_position[pos] = 0.0
        return safe_positions, risk_by_position

    def _build_sat_model(
        self,
        board: tuple[tuple[str, ...], ...],
        legal_positions: set[Position],
    ) -> _SatModel:
        vpool = IDPool()
        clauses: list[list[int]] = []
        var_for_position = {pos: vpool.id(pos) for pos in sorted(legal_positions)}
        position_for_var = {var: pos for pos, var in var_for_position.items()}

        board_hidden = {
            (row, col)
            for row in range(len(board))
            for col in range(len(board[row]))
            if board[row][col] == "?"
        }
        detonated_mines = board_hidden - legal_positions
        known_mines = (self._known_mines & legal_positions) | detonated_mines

        for pos in sorted(self._known_mines & legal_positions):
            clauses.append([var_for_position[pos]])

        constrained_positions: set[Position] = set()
        consistent = True
        for row in range(len(board)):
            for col in range(len(board[row])):
                cell = board[row][col]
                if not cell.isdigit():
                    continue

                clue = int(cell)
                neighboring_known_mines = 0
                neighboring_unknown_vars: list[int] = []
                for neighbor in self._neighbors(board, row, col):
                    if neighbor in known_mines:
                        neighboring_known_mines += 1
                    elif neighbor in legal_positions:
                        neighboring_unknown_vars.append(var_for_position[neighbor])
                        constrained_positions.add(neighbor)

                required = clue - neighboring_known_mines
                if not self._add_exactly(clauses, neighboring_unknown_vars, required, vpool):
                    consistent = False

        if self._num_mines is not None:
            remaining_mines = self._num_mines - len(known_mines)
            remaining_vars = [
                var_for_position[pos]
                for pos in sorted(legal_positions)
                if pos not in known_mines
            ]
            if not self._add_exactly(clauses, remaining_vars, remaining_mines, vpool):
                consistent = False

        return _SatModel(
            clauses=clauses,
            var_for_position=var_for_position,
            position_for_var=position_for_var,
            constrained_positions=constrained_positions,
            known_mines=set(known_mines),
            consistent=consistent,
        )

    def _add_exactly(
        self,
        clauses: list[list[int]],
        lits: list[int],
        required: int,
        vpool,
    ) -> bool:
        if required < 0 or required > len(lits):
            return False
        if required == 0:
            clauses.extend([[-lit] for lit in lits])
            return True
        if required == len(lits):
            clauses.extend([[lit] for lit in lits])
            return True
        cnf = CardEnc.equals(
            lits=lits,
            bound=required,
            vpool=vpool,
            encoding=EncType.seqcounter,
        )
        clauses.extend(cnf.clauses)
        return True

    def _new_solver(self, clauses: list[list[int]]):
        for solver_name in ("g3", "glucose3", "m22", "minisat22"):
            try:
                return Solver(name=solver_name, bootstrap_with=clauses)
            except Exception:
                continue
        return None

    def _is_satisfiable(self, clauses: list[list[int]], assumptions: list[int]) -> bool:
        solver = self._new_solver(clauses)
        if solver is None:
            return True
        try:
            return bool(solver.solve(assumptions=assumptions))
        finally:
            solver.delete()

    def _estimate_risk(self, model: _SatModel) -> dict[Position, float]:
        positions = sorted(
            pos for pos in model.var_for_position if pos not in model.known_mines
        )
        if not positions:
            return {}

        solver = self._new_solver(model.clauses)
        if solver is None:
            return {}

        true_counts = {pos: 0 for pos in positions}
        model_count = 0
        try:
            while model_count < self._max_models and solver.solve():
                assignment = set(lit for lit in solver.get_model() if lit > 0)
                block_clause: list[int] = []
                for pos in positions:
                    var = model.var_for_position[pos]
                    if var in assignment:
                        true_counts[pos] += 1
                        block_clause.append(-var)
                    else:
                        block_clause.append(var)
                solver.add_clause(block_clause)
                model_count += 1
        finally:
            solver.delete()

        if model_count == 0:
            return {}
        return {pos: true_counts[pos] / model_count for pos in positions}

    def _local_safe_moves(
        self,
        board: tuple[tuple[str, ...], ...],
        legal_positions: set[Position],
    ) -> set[Position]:
        safe_moves: set[Position] = set()
        board_hidden = {
            (row, col)
            for row in range(len(board))
            for col in range(len(board[row]))
            if board[row][col] == "?"
        }
        known_mines = self._known_mines | (board_hidden - legal_positions)
        changed = True
        while changed:
            changed = False
            for row in range(len(board)):
                for col in range(len(board[row])):
                    cell = board[row][col]
                    if not cell.isdigit():
                        continue

                    hidden_neighbors = []
                    known_mine_neighbors = 0
                    for neighbor in self._neighbors(board, row, col):
                        if neighbor in known_mines:
                            known_mine_neighbors += 1
                        elif neighbor in legal_positions:
                            hidden_neighbors.append(neighbor)

                    if not hidden_neighbors:
                        continue

                    clue = int(cell)
                    if clue == known_mine_neighbors:
                        safe_moves.update(hidden_neighbors)

                    remaining_mines = clue - known_mine_neighbors
                    if remaining_mines == len(hidden_neighbors):
                        for position in hidden_neighbors:
                            if position not in known_mines:
                                known_mines.add(position)
                                self._known_mines.add(position)
                                changed = True
                                safe_moves.discard(position)

        return {
            position
            for position in safe_moves
            if position in legal_positions and position not in self._known_mines
        }

    def _choose_safe_position(
        self,
        board: tuple[tuple[str, ...], ...],
        safe_positions: set[Position],
        legal_positions: set[Position],
        risk_by_position: dict[Position, float],
    ) -> Position:
        def score(pos: Position) -> tuple[float, int, int, int, int, int]:
            adjacent_mine_pressure = self._adjacent_mine_pressure(
                board,
                pos,
                risk_by_position,
            )
            adjacent_digits = self._adjacent_digit_count(board, pos)
            adjacent_blanks = self._adjacent_blank_count(board, pos)
            hidden_neighbors = self._adjacent_legal_count(board, pos, legal_positions)
            row, col = pos
            return (
                -adjacent_mine_pressure,
                -adjacent_digits,
                adjacent_blanks,
                hidden_neighbors,
                row,
                col,
            )

        return min(safe_positions, key=score)

    def _choose_guess_position(
        self,
        board: tuple[tuple[str, ...], ...],
        positions: set[Position],
        legal_positions: set[Position],
        risk_by_position: dict[Position, float],
    ) -> Position:
        default_risk = self._default_mine_probability(len(legal_positions))

        def score(pos: Position) -> tuple[float, int, float, int, int, int]:
            risk = risk_by_position.get(pos, default_risk)
            if pos not in risk_by_position and self._has_visible_constraint(board, pos):
                risk = min(risk, self._frontier_risk_estimate(board, pos, legal_positions))
            adjacent_mine_pressure = self._adjacent_mine_pressure(
                board,
                pos,
                risk_by_position,
            )
            hidden_neighbors = self._adjacent_legal_count(board, pos, legal_positions)
            row, col = pos
            return (
                risk,
                -self._adjacent_digit_count(board, pos),
                -adjacent_mine_pressure,
                -hidden_neighbors,
                row,
                col,
            )

        return min(positions, key=score)

    def _default_mine_probability(self, legal_count: int) -> float:
        if self._num_mines is None or legal_count <= 0:
            return 0.5
        remaining_known = len(self._known_mines)
        remaining_mines = max(0, self._num_mines - remaining_known)
        return min(1.0, remaining_mines / legal_count)

    def _frontier_risk_estimate(
        self,
        board: tuple[tuple[str, ...], ...],
        pos: Position,
        legal_positions: set[Position],
    ) -> float:
        estimates = []
        for neighbor_row, neighbor_col in self._neighbors(board, pos[0], pos[1]):
            cell = board[neighbor_row][neighbor_col]
            if not cell.isdigit():
                continue
            hidden = []
            known_mines = 0
            for clue_neighbor in self._neighbors(board, neighbor_row, neighbor_col):
                if clue_neighbor in self._known_mines:
                    known_mines += 1
                elif clue_neighbor in legal_positions:
                    hidden.append(clue_neighbor)
            if pos in hidden and hidden:
                estimates.append(max(0, int(cell) - known_mines) / len(hidden))
        if not estimates:
            return self._default_mine_probability(len(legal_positions))
        return max(estimates)

    def _adjacent_mine_pressure(
        self,
        board: tuple[tuple[str, ...], ...],
        pos: Position,
        risk_by_position: dict[Position, float],
    ) -> float:
        pressure = 0.0
        for neighbor in self._neighbors(board, pos[0], pos[1]):
            if neighbor in self._known_mines:
                pressure += 1.0
            else:
                pressure += risk_by_position.get(neighbor, 0.0)
        return pressure

    def _adjacent_digit_count(
        self,
        board: tuple[tuple[str, ...], ...],
        pos: Position,
    ) -> int:
        return sum(
            1
            for row, col in self._neighbors(board, pos[0], pos[1])
            if board[row][col].isdigit()
        )

    def _adjacent_blank_count(
        self,
        board: tuple[tuple[str, ...], ...],
        pos: Position,
    ) -> int:
        return sum(
            1
            for row, col in self._neighbors(board, pos[0], pos[1])
            if board[row][col] == " "
        )

    def _adjacent_legal_count(
        self,
        board: tuple[tuple[str, ...], ...],
        pos: Position,
        legal_positions: set[Position],
    ) -> int:
        return sum(
            1
            for neighbor in self._neighbors(board, pos[0], pos[1])
            if neighbor in legal_positions
        )

    def _has_visible_constraint(
        self,
        board: tuple[tuple[str, ...], ...],
        pos: Position,
    ) -> bool:
        return any(
            board[row][col].isdigit()
            for row, col in self._neighbors(board, pos[0], pos[1])
        )

    def _neighbors(
        self,
        board: tuple[tuple[str, ...], ...],
        row: int,
        col: int,
    ) -> list[Position]:
        neighbors = []
        for neighbor_row in range(max(0, row - 1), min(len(board), row + 2)):
            for neighbor_col in range(max(0, col - 1), min(len(board[row]), col + 2)):
                if neighbor_row != row or neighbor_col != col:
                    neighbors.append((neighbor_row, neighbor_col))
        return neighbors


class SATAgent(StrategicSATAgent):
    """Short alias for tournament registration."""


def make_sat_agent() -> StrategicSATAgent:
    return StrategicSATAgent(seed=7, name="sat")
