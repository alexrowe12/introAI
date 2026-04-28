#!/usr/bin/env python3
"""
Exact finite-deck Blackjack tournament agent.

This script preserves the older tournament_agent.py and implements a stronger
hit/stand/double decision engine for the no-5s finite-deck tournament. It uses exact
expectimax over the remaining deck, with memoization, instead of approximating
future draws with fixed probabilities.

Fast commands:
    n 7 9 6        start hand: player 7, player 9, dealer 6
    h T            record hit card
    s              stand
    x 6            double down: record one card, then stand
    e 6 T 8        end hand / record dealer cards
    o 2 A K        record observed cards from other hands
    u              undo last mutation
    d              show deck
    rules          show rules
    set dealer h17 set dealer hit-soft-17 mode
    set peek on    condition decisions on dealer having checked blackjack
    r              reset/shuffle deck
    q              quit
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import argparse
import math
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from blackjack_game import evaluate_hand


Deck = Tuple[int, ...]  # indexes 0..9 represent values A..10

HIT = "HIT"
STAND = "STAND"
DOUBLE = "DOUBLE"


@dataclass
class Rules:
    """Configurable tournament rule assumptions."""

    removed_value: int = 5
    dealer_hits_soft_17: bool = False
    blackjack_payout: float = 1.0
    dealer_peek: bool = False
    scoring_mode: str = "chip_ev"  # "chip_ev" or "flat_round"
    shuffle_threshold: int = 0
    double_after_hit: bool = False

    def reward_blackjack(self) -> float:
        if self.scoring_mode == "flat_round":
            return 1.0
        return self.blackjack_payout


@dataclass(frozen=True)
class Decision:
    action: str
    hit_ev: float
    stand_ev: float
    double_ev: float = -math.inf

    @property
    def edge(self) -> float:
        values = sorted([self.hit_ev, self.stand_ev, self.double_ev], reverse=True)
        return values[0] - values[1]


@dataclass
class Snapshot:
    rules: Rules
    deck: Deck
    player_cards: Tuple[int, ...]
    dealer_up: Optional[int]
    hand_active: bool
    hand_counted: bool
    doubled: bool
    hands_played: int
    wins: int
    losses: int
    pushes: int
    total_result: float
    observe_enabled: bool


def initial_deck(removed_value: int = 5) -> Deck:
    """Build value-count deck: A..9 have 4 cards, value 10 has 16 cards."""
    counts = [0] * 10
    for value in range(1, 10):
        counts[value - 1] = 0 if value == removed_value else 4
    counts[9] = 12 if removed_value == 10 else 16
    return tuple(counts)


def deck_total(deck: Deck) -> int:
    return sum(deck)


def decrement(deck: Deck, value: int) -> Deck:
    idx = value - 1
    if idx < 0 or idx >= 10 or deck[idx] <= 0:
        raise ValueError(f"Cannot remove unavailable card {format_card(value)}")
    counts = list(deck)
    counts[idx] -= 1
    return tuple(counts)


def add_to_hand(total: int, usable_ace: bool, value: int) -> Tuple[int, bool]:
    """Add one value-card to an evaluated hand state."""
    if value == 1:
        if total + 11 <= 21:
            total += 11
            usable_ace = True
        else:
            total += 1
    else:
        total += value

    if total > 21 and usable_ace:
        total -= 10
        usable_ace = False

    return total, usable_ace


def is_blackjack_pair(card1: int, card2: int) -> bool:
    return (card1 == 1 and card2 == 10) or (card1 == 10 and card2 == 1)


def parse_card(token: str) -> int:
    token = token.strip().upper()
    values = {
        "A": 1,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "T": 10,
        "J": 10,
        "Q": 10,
        "K": 10,
    }
    if token not in values:
        raise ValueError(f"Invalid card '{token}'. Use A, 2-10, T, J, Q, or K.")
    return values[token]


def parse_cards(text: str) -> List[int]:
    return [parse_card(part) for part in text.replace(",", " ").split()]


def format_card(value: Optional[int]) -> str:
    if value is None:
        return "-"
    if value == 1:
        return "A"
    if value == 10:
        return "T"
    return str(value)


def format_cards(cards: Iterable[int]) -> str:
    return " ".join(format_card(card) for card in cards)


class ExactFiniteDeckSolver:
    """Exact hit/stand finite-deck expectimax solver with memoization."""

    def __init__(self, rules: Rules):
        self.rules = rules
        self.fresh_deck = initial_deck(rules.removed_value)

    def clear(self) -> None:
        self._player_value.cache_clear()
        self._dealer_outcomes.cache_clear()

    def decision(
        self,
        player_total: int,
        usable_ace: bool,
        dealer_up: int,
        deck: Deck,
        player_natural: bool = False,
        can_double: bool = False,
    ) -> Decision:
        stand_ev = self.stand_ev(
            player_total, usable_ace, dealer_up, deck, player_natural
        )

        if player_total >= 21 or player_natural:
            return Decision(STAND, -math.inf, stand_ev, -math.inf)

        hit_ev = self.hit_ev(player_total, usable_ace, dealer_up, deck)
        double_ev = (
            self.double_ev(player_total, usable_ace, dealer_up, deck)
            if can_double
            else -math.inf
        )

        action_values = {
            HIT: hit_ev,
            STAND: stand_ev,
            DOUBLE: double_ev,
        }
        action = max(action_values, key=action_values.get)
        return Decision(action, hit_ev, stand_ev, double_ev)

    def best_ev(
        self,
        player_total: int,
        usable_ace: bool,
        dealer_up: int,
        deck: Deck,
        player_natural: bool = False,
        can_double: bool = False,
    ) -> float:
        return self._player_value(
            player_total,
            bool(usable_ace),
            dealer_up,
            self._draw_deck(deck),
            player_natural,
            can_double,
        )

    @lru_cache(maxsize=750_000)
    def _player_value(
        self,
        player_total: int,
        usable_ace: bool,
        dealer_up: int,
        deck: Deck,
        player_natural: bool,
        can_double: bool,
    ) -> float:
        if player_total > 21:
            return -1.0

        stand = self.stand_ev(
            player_total, usable_ace, dealer_up, deck, player_natural
        )
        if player_total >= 21 or player_natural:
            return stand

        hit = self.hit_ev(player_total, usable_ace, dealer_up, deck)
        if can_double:
            return max(hit, stand, self.double_ev(player_total, usable_ace, dealer_up, deck))
        return max(hit, stand)

    def hit_ev(
        self, player_total: int, usable_ace: bool, dealer_up: int, deck: Deck
    ) -> float:
        draw_deck = self._draw_deck(deck)
        total_cards = deck_total(draw_deck)
        ev = 0.0

        for value, count in self._available(draw_deck):
            prob = count / total_cards
            next_deck = decrement(draw_deck, value)
            next_total, next_usable = add_to_hand(player_total, usable_ace, value)
            if next_total > 21:
                ev += prob * -1.0
            else:
                ev += prob * self._player_value(
                    next_total, next_usable, dealer_up, next_deck, False, False
                )

        return ev

    def double_ev(
        self, player_total: int, usable_ace: bool, dealer_up: int, deck: Deck
    ) -> float:
        """Expected value of doubling: one card, then forced stand, double stakes."""
        draw_deck = self._draw_deck(deck)
        total_cards = deck_total(draw_deck)
        ev = 0.0

        for value, count in self._available(draw_deck):
            prob = count / total_cards
            next_deck = decrement(draw_deck, value)
            next_total, next_usable = add_to_hand(player_total, usable_ace, value)
            if next_total > 21:
                ev += prob * -2.0
            else:
                ev += prob * 2.0 * self.stand_ev(
                    next_total, next_usable, dealer_up, next_deck, False
                )

        return ev

    def stand_ev(
        self,
        player_total: int,
        usable_ace: bool,
        dealer_up: int,
        deck: Deck,
        player_natural: bool = False,
    ) -> float:
        del usable_ace  # total already encodes usable ace for comparison.

        if player_total > 21:
            return -1.0

        draw_deck = self._draw_deck(deck)
        allowed_holes = list(self._available(draw_deck))

        if self.rules.dealer_peek and not player_natural and dealer_up in (1, 10):
            allowed_holes = [
                (value, count)
                for value, count in allowed_holes
                if not is_blackjack_pair(dealer_up, value)
            ]

        total_holes = sum(count for _, count in allowed_holes)
        if total_holes <= 0:
            draw_deck = self.fresh_deck
            allowed_holes = list(self._available(draw_deck))
            total_holes = sum(count for _, count in allowed_holes)

        ev = 0.0
        for hole, count in allowed_holes:
            prob = count / total_holes
            after_hole = decrement(draw_deck, hole)
            dealer_total, dealer_usable = evaluate_hand([dealer_up, hole])
            dealer_natural = is_blackjack_pair(dealer_up, hole)

            if dealer_natural:
                reward = self._compare(
                    player_total, dealer_total, player_natural, dealer_natural
                )
                ev += prob * reward
                continue

            outcomes = self._dealer_outcomes(dealer_total, dealer_usable, after_hole)
            for dealer_final, outcome_prob in outcomes.items():
                ev += prob * outcome_prob * self._compare(
                    player_total, dealer_final, player_natural, False
                )

        return ev

    @lru_cache(maxsize=750_000)
    def _dealer_outcomes(
        self, dealer_total: int, dealer_usable: bool, deck: Deck
    ) -> Dict[int, float]:
        """Return dealer final total distribution. Key 22 means bust."""
        if self._dealer_should_stand(dealer_total, dealer_usable):
            return {22 if dealer_total > 21 else dealer_total: 1.0}

        draw_deck = self._draw_deck(deck)
        total_cards = deck_total(draw_deck)
        outcomes: Dict[int, float] = {}

        for value, count in self._available(draw_deck):
            prob = count / total_cards
            next_deck = decrement(draw_deck, value)
            next_total, next_usable = add_to_hand(dealer_total, dealer_usable, value)
            child = self._dealer_outcomes(next_total, next_usable, next_deck)
            for final_total, final_prob in child.items():
                outcomes[final_total] = outcomes.get(final_total, 0.0) + prob * final_prob

        return outcomes

    def _dealer_should_stand(self, total: int, usable_ace: bool) -> bool:
        if total > 17:
            return True
        if total < 17:
            return False
        return not (usable_ace and self.rules.dealer_hits_soft_17)

    def _compare(
        self,
        player_total: int,
        dealer_total: int,
        player_natural: bool,
        dealer_natural: bool,
    ) -> float:
        if player_natural and dealer_natural:
            return 0.0
        if player_natural:
            return self.rules.reward_blackjack()
        if dealer_natural:
            return -1.0
        if player_total > 21:
            return -1.0
        if dealer_total > 21:
            return 1.0
        if player_total > dealer_total:
            return 1.0
        if player_total < dealer_total:
            return -1.0
        return 0.0

    def _draw_deck(self, deck: Deck) -> Deck:
        """Use a fresh deck if a hypothetical line exhausts the deck mid-hand."""
        return deck if deck_total(deck) > 0 else self.fresh_deck

    @staticmethod
    def _available(deck: Deck) -> Iterable[Tuple[int, int]]:
        for idx, count in enumerate(deck):
            if count > 0:
                yield idx + 1, count

    def cache_info(self) -> str:
        return (
            f"player={self._player_value.cache_info()} "
            f"dealer={self._dealer_outcomes.cache_info()}"
        )


class ExactTournamentAgent:
    """Fast live CLI wrapper around ExactFiniteDeckSolver."""

    def __init__(self, rules: Optional[Rules] = None):
        self.rules = rules or Rules()
        self.solver = ExactFiniteDeckSolver(self.rules)
        self.deck: Deck = initial_deck(self.rules.removed_value)
        self.player_cards: List[int] = []
        self.dealer_up: Optional[int] = None
        self.hand_active = False
        self.hand_counted = False
        self.doubled = False
        self.observe_enabled = True

        self.hands_played = 0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.total_result = 0.0

        self.history: List[Snapshot] = []

    def snapshot(self) -> None:
        self.history.append(
            Snapshot(
                rules=Rules(
                    removed_value=self.rules.removed_value,
                    dealer_hits_soft_17=self.rules.dealer_hits_soft_17,
                    blackjack_payout=self.rules.blackjack_payout,
                    dealer_peek=self.rules.dealer_peek,
                    scoring_mode=self.rules.scoring_mode,
                    shuffle_threshold=self.rules.shuffle_threshold,
                    double_after_hit=self.rules.double_after_hit,
                ),
                deck=self.deck,
                player_cards=tuple(self.player_cards),
                dealer_up=self.dealer_up,
                hand_active=self.hand_active,
                hand_counted=self.hand_counted,
                doubled=self.doubled,
                hands_played=self.hands_played,
                wins=self.wins,
                losses=self.losses,
                pushes=self.pushes,
                total_result=self.total_result,
                observe_enabled=self.observe_enabled,
            )
        )
        if len(self.history) > 500:
            self.history.pop(0)

    def restore(self, snapshot: Snapshot) -> None:
        self.rules = snapshot.rules
        self.deck = snapshot.deck
        self.player_cards = list(snapshot.player_cards)
        self.dealer_up = snapshot.dealer_up
        self.hand_active = snapshot.hand_active
        self.hand_counted = snapshot.hand_counted
        self.doubled = snapshot.doubled
        self.hands_played = snapshot.hands_played
        self.wins = snapshot.wins
        self.losses = snapshot.losses
        self.pushes = snapshot.pushes
        self.total_result = snapshot.total_result
        self.observe_enabled = snapshot.observe_enabled
        self.solver = ExactFiniteDeckSolver(self.rules)

    def undo(self) -> str:
        if not self.history:
            return "Nothing to undo."
        snap = self.history.pop()
        self.restore(snap)
        return self.status("Undo complete.")

    def reset_deck(self) -> str:
        self.snapshot()
        self.deck = initial_deck(self.rules.removed_value)
        self.player_cards = []
        self.dealer_up = None
        self.hand_active = False
        self.hand_counted = False
        self.doubled = False
        self.solver.clear()
        return self.status("Deck reset/shuffled.")

    def start_hand(self, cards: List[int]) -> str:
        if len(cards) != 3:
            return "Usage: n <player1> <player2> <dealer_up>"
        self.snapshot()
        self.player_cards = []
        self.dealer_up = None
        self.hand_active = False
        self.hand_counted = False
        self.doubled = False

        ok, message = self._remove_cards(cards)
        if not ok:
            self.undo()
            return message

        self.player_cards = [cards[0], cards[1]]
        self.dealer_up = cards[2]
        self.hand_active = True
        return self.status()

    def hit(self, card: int) -> str:
        if not self.hand_active:
            return "No active hand. Use: n <p1> <p2> <dealer_up>"
        if self.doubled:
            return "Already doubled. Enter dealer cards with: e <cards>"

        self.snapshot()
        ok, message = self._remove_cards([card])
        if not ok:
            self.undo()
            return message

        self.player_cards.append(card)
        total, _ = evaluate_hand(self.player_cards)
        if total > 21:
            self.hand_active = False
            self._record_result("lose", -1.0)
            return self.status("BUST. Loss recorded. Use e <dealer cards> if you need to track revealed dealer cards.")

        return self.status()

    def double_down(self, card: int) -> str:
        if not self.hand_active:
            return "No active hand. Use: n <p1> <p2> <dealer_up>"
        if self.doubled:
            return "Already doubled. Enter dealer cards with: e <cards>"
        if not self._can_double():
            return "Double is only legal on the initial two-card hand."

        self.snapshot()
        ok, message = self._remove_cards([card])
        if not ok:
            self.undo()
            return message

        self.player_cards.append(card)
        self.doubled = True
        total, _ = evaluate_hand(self.player_cards)
        if total > 21:
            self.hand_active = False
            self._record_result("lose", -2.0)
            return self.status(
                "DOUBLE BUST. Double loss recorded. Use e <dealer cards> if you need to track revealed dealer cards."
            )

        return self.status("Doubled. Forced stand. Enter dealer cards with: e <cards>")

    def stand(self) -> str:
        if not self.player_cards:
            return "No current hand."
        total, _ = evaluate_hand(self.player_cards)
        if total > 21 or self.hand_counted:
            return "Current hand is already over."
        self.hand_active = True
        return self.status("Standing. Enter dealer cards with: e <cards>")

    def end_hand(self, cards: List[int]) -> str:
        if self.dealer_up is None and not cards:
            return "No dealer upcard known."

        self.snapshot()

        dealer_cards = list(cards)
        if self.dealer_up is not None:
            if not dealer_cards or dealer_cards[0] != self.dealer_up:
                dealer_cards = [self.dealer_up] + dealer_cards

        cards_to_remove: List[int] = []
        skipped_upcard = False
        for idx, card in enumerate(dealer_cards):
            if (
                idx == 0
                and self.dealer_up is not None
                and card == self.dealer_up
                and not skipped_upcard
            ):
                skipped_upcard = True
                continue
            cards_to_remove.append(card)

        ok, message = self._remove_cards(cards_to_remove)
        if not ok:
            self.undo()
            return message

        result_line = f"Recorded dealer cards: {format_cards(dealer_cards)}"
        if self.player_cards and not self.hand_counted:
            player_total, _ = evaluate_hand(self.player_cards)
            dealer_total, _ = evaluate_hand(dealer_cards)
            player_nat = len(self.player_cards) == 2 and player_total == 21
            dealer_nat = len(dealer_cards) == 2 and dealer_total == 21
            outcome, reward = self._actual_outcome(
                player_total, dealer_total, player_nat, dealer_nat
            )
            if self.doubled:
                reward *= 2.0
            self._record_result(outcome, reward)
            result_line = (
                f"Dealer: {format_cards(dealer_cards)} = {dealer_total}\n"
                f"Result: {outcome.upper()} ({reward:+.1f})"
            )

        self.player_cards = []
        self.dealer_up = None
        self.hand_active = False
        self.hand_counted = False
        self.doubled = False
        return self.status(result_line)

    def observe(self, cards: List[int]) -> str:
        if not cards:
            return "Usage: o <cards>"
        if not self.observe_enabled:
            return "Observe mode is off. Use: observe on"

        self.snapshot()
        ok, message = self._remove_cards(cards)
        if not ok:
            self.undo()
            return message
        return self.status(f"Observed: {format_cards(cards)}")

    def set_rule(self, parts: List[str]) -> str:
        if len(parts) < 2:
            return self.rules_text()

        key = parts[0].lower()
        value = parts[1].lower()
        self.snapshot()

        try:
            if key in ("dealer", "soft17"):
                if value not in ("s17", "h17"):
                    return "Use: set dealer s17|h17"
                self.rules.dealer_hits_soft_17 = value == "h17"
            elif key in ("peek", "dealer_peek"):
                self.rules.dealer_peek = value in ("on", "true", "yes", "1")
            elif key in ("bj", "blackjack"):
                self.rules.blackjack_payout = float(value)
            elif key in ("score", "scoring"):
                if value not in ("chip", "chip_ev", "flat", "flat_round"):
                    return "Use: set score chip|flat"
                self.rules.scoring_mode = "flat_round" if value.startswith("flat") else "chip_ev"
            elif key in ("shuffle", "threshold"):
                threshold = int(value)
                if threshold < 0:
                    return "Shuffle threshold must be non-negative."
                self.rules.shuffle_threshold = threshold
            elif key in ("double", "double_after_hit"):
                if value not in ("initial", "afterhit", "after_hit", "any"):
                    return "Use: set double initial|any"
                self.rules.double_after_hit = value in ("afterhit", "after_hit", "any")
            elif key in ("removed", "remove"):
                removed = parse_card(value)
                self.rules.removed_value = removed
                self.deck = initial_deck(removed)
                self.player_cards = []
                self.dealer_up = None
                self.hand_active = False
                self.hand_counted = False
                self.doubled = False
            else:
                return f"Unknown rule '{key}'."
        finally:
            self.solver = ExactFiniteDeckSolver(self.rules)

        return self.rules_text()

    def set_observe(self, enabled: bool) -> str:
        self.snapshot()
        self.observe_enabled = enabled
        return f"Observe mode: {'ON' if enabled else 'OFF'}"

    def status(self, prefix: str = "") -> str:
        lines: List[str] = []
        if prefix:
            lines.append(prefix)

        if self.player_cards:
            total, usable = evaluate_hand(self.player_cards)
            hand_kind = "soft" if usable else "hard"
            lines.append(f"Hand: {format_cards(self.player_cards)} = {total} ({hand_kind})")
        else:
            total, usable = 0, False
            lines.append("Hand: -")

        lines.append(f"Dealer: {format_card(self.dealer_up)}")
        lines.append(f"Deck: {deck_total(self.deck)} cards | {self.compact_deck()}")

        decision = self._decision_line(total, usable)
        if decision:
            lines.append(decision)

        if self.rules.shuffle_threshold and deck_total(self.deck) <= self.rules.shuffle_threshold:
            lines.append(
                f"Shuffle threshold reached ({self.rules.shuffle_threshold}). Use r when the real deck is shuffled."
            )

        if self.hands_played:
            lines.append(
                f"Session: {self.total_result:+.1f} over {self.hands_played} "
                f"(W {self.wins} / L {self.losses} / P {self.pushes})"
            )

        return "\n".join(lines)

    def _decision_line(self, total: int, usable: bool) -> str:
        if not self.player_cards or self.dealer_up is None or not self.hand_active:
            return ""
        if total > 21:
            return ">>> BUST"
        if self.doubled:
            return ">>> FORCED STAND after DOUBLE. Enter dealer cards with: e <cards>"

        player_natural = len(self.player_cards) == 2 and total == 21
        decision = self.solver.decision(
            total, usable, self.dealer_up, self.deck, player_natural, self._can_double()
        )

        if player_natural:
            return f">>> BLACKJACK / STAND | Stand EV {decision.stand_ev:+.4f}"

        if total == 21:
            return f">>> STAND (21) | Stand EV {decision.stand_ev:+.4f}"

        confidence = "strong"
        if decision.edge < 0.015:
            confidence = "razor-close"
        elif decision.edge < 0.05:
            confidence = "close"
        elif decision.edge < 0.12:
            confidence = "medium"

        values = f"H {decision.hit_ev:+.4f} / S {decision.stand_ev:+.4f}"
        if math.isfinite(decision.double_ev):
            values += f" / D {decision.double_ev:+.4f}"

        return f">>> {decision.action} ({confidence}) | {values} / edge {decision.edge:+.4f}"

    def compact_deck(self) -> str:
        labels = []
        for value, count in enumerate(self.deck, start=1):
            if count > 0 or value == self.rules.removed_value:
                labels.append(f"{format_card(value)}:{count}")
        return " ".join(labels)

    def deck_text(self) -> str:
        lines = [f"Deck: {deck_total(self.deck)} cards remaining"]
        for value, count in enumerate(self.deck, start=1):
            initial = initial_deck(self.rules.removed_value)[value - 1]
            lines.append(f"  {format_card(value):>2}: {count:2}/{initial:2}")
        return "\n".join(lines)

    def rules_text(self) -> str:
        dealer = "H17" if self.rules.dealer_hits_soft_17 else "S17"
        peek = "ON" if self.rules.dealer_peek else "OFF"
        observe = "ON" if self.observe_enabled else "OFF"
        score = self.rules.scoring_mode
        double_rule = "any hand" if self.rules.double_after_hit else "initial two cards only"
        return (
            "Rules:\n"
            f"  removed value: {format_card(self.rules.removed_value)}\n"
            f"  dealer: {dealer}\n"
            f"  dealer peek conditioning: {peek}\n"
            f"  blackjack payout: {self.rules.blackjack_payout:g}\n"
            f"  double down: allowed, {double_rule}\n"
            f"  scoring mode: {score}\n"
            f"  shuffle threshold: {self.rules.shuffle_threshold}\n"
            f"  observe mode: {observe}"
        )

    def stats_text(self) -> str:
        avg = self.total_result / self.hands_played if self.hands_played else 0.0
        return (
            f"Hands: {self.hands_played}\n"
            f"W/L/P: {self.wins}/{self.losses}/{self.pushes}\n"
            f"Total: {self.total_result:+.1f}\n"
            f"Average: {avg:+.3f}"
        )

    def help_text(self) -> str:
        return """
Fast commands:
  n 7 9 6        new hand: player 7, player 9, dealer 6
  h T            hit card
  x 6            double down: take one card, double stakes, forced stand
  s              stand
  e 6 T 8        end hand / record dealer cards
  e T 8          also accepted; dealer upcard is prepended
  o 2 A K        observed cards from other hands
  observe on/off factor or ignore observed cards
  d              deck
  rules          show rule assumptions
  set dealer s17|h17
  set peek on|off
  set bj 1.0
  set double initial|any
  set score chip|flat
  set shuffle N
  u              undo
  r              reset/shuffle deck
  stats          session stats
  cache          solver cache stats
  q              quit
""".strip()

    def _remove_cards(self, cards: List[int]) -> Tuple[bool, str]:
        trial = self.deck
        try:
            for card in cards:
                trial = decrement(trial, card)
        except ValueError as exc:
            return False, f"{exc}. Deck unchanged."
        self.deck = trial
        self.solver.clear()
        return True, ""

    def _actual_outcome(
        self,
        player_total: int,
        dealer_total: int,
        player_natural: bool,
        dealer_natural: bool,
    ) -> Tuple[str, float]:
        if player_natural and dealer_natural:
            return "push", 0.0
        if player_natural:
            return "blackjack", self.rules.reward_blackjack()
        if dealer_natural:
            return "lose", -1.0
        if player_total > 21:
            return "lose", -1.0
        if dealer_total > 21 or player_total > dealer_total:
            return "win", 1.0
        if player_total < dealer_total:
            return "lose", -1.0
        return "push", 0.0

    def _can_double(self) -> bool:
        if self.doubled or self.hand_counted:
            return False
        if len(self.player_cards) < 2:
            return False
        if self.rules.double_after_hit:
            return True
        return len(self.player_cards) == 2

    def _record_result(self, outcome: str, reward: float) -> None:
        if self.hand_counted:
            return
        self.hands_played += 1
        self.total_result += reward
        if outcome in ("win", "blackjack"):
            self.wins += 1
        elif outcome == "lose":
            self.losses += 1
        else:
            self.pushes += 1
        self.hand_counted = True

    def run(self) -> None:
        print("Exact finite-deck Blackjack agent")
        print(self.rules_text())
        print(self.help_text())
        print()
        print(self.status())

        while True:
            try:
                prompt = "hand> " if self.hand_active else "> "
                line = input(prompt).strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\nUse q to quit.")
                continue

            if not line:
                continue

            try:
                output = self.handle_command(line)
            except Exception as exc:  # Keep the live tool alive on game day.
                output = f"Error: {exc}"

            if output == "__quit__":
                break
            if output:
                print(output)

    def handle_command(self, line: str) -> str:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("q", "quit", "exit"):
            return "__quit__"
        if cmd in ("help", "?"):
            return self.help_text()
        if cmd in ("n", "new"):
            return self.start_hand(parse_cards(arg))
        if cmd in ("h", "hit"):
            cards = parse_cards(arg)
            if len(cards) != 1:
                return "Usage: h <card>"
            return self.hit(cards[0])
        if cmd in ("x", "double", "double_down"):
            cards = parse_cards(arg)
            if len(cards) != 1:
                return "Usage: x <card>"
            return self.double_down(cards[0])
        if cmd in ("s", "stand"):
            return self.stand()
        if cmd in ("e", "end"):
            return self.end_hand(parse_cards(arg))
        if cmd in ("o", "obs", "seen"):
            return self.observe(parse_cards(arg))
        if cmd == "observe":
            value = arg.strip().lower()
            if value not in ("on", "off"):
                return "Use: observe on|off"
            return self.set_observe(value == "on")
        if cmd in ("u", "undo"):
            return self.undo()
        if cmd in ("r", "reset", "shuffle"):
            return self.reset_deck()
        if cmd in ("d", "deck"):
            return self.deck_text()
        if cmd == "rules":
            return self.rules_text()
        if cmd == "set":
            return self.set_rule(arg.split())
        if cmd == "stats":
            return self.stats_text()
        if cmd == "cache":
            return self.solver.cache_info()
        if cmd == "status":
            return self.status()

        return f"Unknown command '{cmd}'. Use help."


def self_test() -> None:
    rules = Rules()
    agent = ExactTournamentAgent(rules)
    assert deck_total(agent.deck) == 48
    assert agent.deck[4] == 0

    out = agent.start_hand([10, 6, 10])
    assert ">>>" in out
    before = agent.deck
    out = agent.observe([5])
    assert "Cannot remove unavailable" in out
    assert agent.deck == before

    solver = agent.solver
    decision = solver.decision(16, False, 10, agent.deck, can_double=True)
    assert decision.action in (HIT, STAND, DOUBLE)
    assert math.isfinite(decision.stand_ev)
    assert math.isfinite(decision.hit_ev)
    assert math.isfinite(decision.double_ev)

    outcomes = solver._dealer_outcomes(16, False, agent.deck)
    assert abs(sum(outcomes.values()) - 1.0) < 1e-9

    out = agent.hit(10)
    assert "BUST" in out
    out = agent.undo()
    assert "Hand:" in out
    assert agent.deck == before

    out = agent.start_hand([6, 4, 6])
    assert "D " in out
    out = agent.double_down(10)
    assert "Doubled" in out or "DOUBLE BUST" in out
    if "Doubled" in out:
        out = agent.end_hand([6, 10])
        assert "Result:" in out

    print("self-test passed")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Exact finite-deck Blackjack agent")
    parser.add_argument("--self-test", action="store_true", help="run internal checks")
    parser.add_argument("--h17", action="store_true", help="dealer hits soft 17")
    parser.add_argument("--peek", action="store_true", help="condition on dealer peek/no blackjack")
    parser.add_argument("--bj", type=float, default=1.0, help="blackjack payout")
    parser.add_argument(
        "--score",
        choices=("chip", "flat"),
        default="chip",
        help="reward profile",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0

    rules = Rules(
        dealer_hits_soft_17=args.h17,
        dealer_peek=args.peek,
        blackjack_payout=args.bj,
        scoring_mode="flat_round" if args.score == "flat" else "chip_ev",
    )
    ExactTournamentAgent(rules).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
