#!/usr/bin/env python3
"""
Practice table for the finite-deck Blackjack tournament.

Run this in a second terminal next to exact_tournament_agent.py. This script
simulates the classroom table. By default it does not tell you exact agent
commands; use --hints for a guided drill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import random
import sys
from typing import Iterable, List, Optional, Tuple

from blackjack_game import evaluate_hand
from exact_tournament_agent import (
    Deck,
    ExactFiniteDeckSolver,
    HIT,
    STAND,
    DOUBLE,
    Rules,
    add_to_hand,
    deck_total,
    decrement,
    format_card,
    format_cards,
    initial_deck,
    is_blackjack_pair,
)


ACTION_ALIASES = {
    "h": HIT,
    "hit": HIT,
    "s": STAND,
    "stand": STAND,
    "x": DOUBLE,
    "d": DOUBLE,
    "double": DOUBLE,
}


@dataclass
class Hand:
    cards: List[int] = field(default_factory=list)
    stood: bool = False
    busted: bool = False
    doubled: bool = False
    natural: bool = False
    reward: float = 0.0

    def total(self) -> Tuple[int, bool]:
        return evaluate_hand(self.cards)

    def label(self) -> str:
        total, usable = self.total()
        kind = "soft" if usable else "hard"
        flags = []
        if self.natural:
            flags.append("natural")
        if self.doubled:
            flags.append("double")
        if self.busted:
            flags.append("bust")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        return f"{format_cards(self.cards)} = {total} ({kind}){suffix}"


class PracticeTable:
    """Interactive simulator for classroom-style practice."""

    def __init__(
        self,
        players: int,
        seat: int,
        rounds: int,
        visibility: str,
        rules: Rules,
        seed: Optional[int],
        auto: bool = False,
        strict: bool = False,
        show_hidden_deck: bool = False,
        hints: bool = False,
    ):
        if players < 1:
            raise ValueError("players must be at least 1")
        if seat < 1 or seat > players:
            raise ValueError("seat must be between 1 and players")

        self.players = players
        self.your_index = seat - 1
        self.rounds = rounds
        self.visibility = visibility
        self.rules = rules
        self.auto = auto
        self.strict = strict
        self.show_hidden_deck = show_hidden_deck
        self.hints = hints
        self.rng = random.Random(seed)
        self.deck: List[int] = []
        self.round_no = 0
        self.total_score = 0.0
        self.off_policy = 0
        self.solver = ExactFiniteDeckSolver(rules)
        self.agent_known_deck: Deck = initial_deck(rules.removed_value)
        self.shuffle()

    def hint(self, text: str = "") -> None:
        if self.hints:
            print(text)

    def shuffle(self) -> None:
        counts = initial_deck(self.rules.removed_value)
        self.deck = []
        for value, count in enumerate(counts, start=1):
            self.deck.extend([value] * count)
        self.rng.shuffle(self.deck)

    def draw(self) -> int:
        if len(self.deck) <= self.rules.shuffle_threshold:
            self._announce_shuffle()
        if not self.deck:
            self._announce_shuffle()
        return self.deck.pop()

    def _announce_shuffle(self) -> None:
        self.shuffle()
        self.agent_known_deck = initial_deck(self.rules.removed_value)
        print("\n*** TABLE SHUFFLED / NEW NO-5s DECK ***")
        self.hint("Agent hint: r")
        self.pause()

    def run(self) -> None:
        print("Practice Blackjack table")
        print(f"Players: {self.players} | Your seat: P{self.your_index + 1}")
        print(f"Rounds: {self.rounds} | Visibility: {self.visibility}")
        print(f"Dealer: {'H17' if self.rules.dealer_hits_soft_17 else 'S17'}")
        print("Natural blackjack pays normal win (+1). Doubling is allowed.")
        print()
        print("Use your agent separately if you want recommendations.")
        if self.hints:
            print("Guided hints are ON.")
        self.pause("Press Enter to start, or q to quit: ")

        for round_no in range(1, self.rounds + 1):
            self.round_no = round_no
            if not self.play_round():
                break

        self.print_summary()

    def play_round(self) -> bool:
        print("\n" + "=" * 72)
        print(f"ROUND {self.round_no}/{self.rounds}")
        print("=" * 72)

        hands = [Hand() for _ in range(self.players)]
        dealer = Hand()

        for hand in hands:
            hand.cards.append(self.draw())
        dealer.cards.append(self.draw())  # upcard
        for hand in hands:
            hand.cards.append(self.draw())
        dealer.cards.append(self.draw())  # hidden hole

        dealer_up = dealer.cards[0]
        your_hand = hands[self.your_index]
        for hand in hands:
            hand.natural = self._is_natural(hand.cards)
        dealer.natural = self._is_natural(dealer.cards)

        self._remove_known(your_hand.cards + [dealer_up])

        print(f"Dealer upcard: {format_card(dealer_up)}")
        print(f"Your hand: P{self.your_index + 1} {your_hand.label()}")
        self.print_visible_initial(hands)

        self.hint("\nAgent hints:")
        self.hint(f"  n {format_cards(your_hand.cards)} {format_card(dealer_up)}")
        visible_initial = self.visible_initial_cards(hands)
        if visible_initial:
            self.hint(f"  o {format_cards(visible_initial)}")
            self._remove_known(visible_initial)
        self.pause()

        for index in range(self.players):
            if index == self.your_index:
                if not self.play_your_hand(your_hand, dealer_up):
                    return False
            else:
                self.play_other_hand(index, hands[index], dealer_up)

        self.play_dealer(dealer)
        self.score_round(hands, dealer)
        self.reveal_round(hands, dealer)
        self.total_score += your_hand.reward
        return True

    def print_visible_initial(self, hands: List[Hand]) -> None:
        if self.visibility == "none":
            print("Other initial cards: hidden in this practice mode")
            return

        visible = []
        for index, hand in enumerate(hands):
            if index == self.your_index:
                continue
            visible.append(f"P{index + 1}: {format_cards(hand.cards)}")
        print("Other initial cards: " + " | ".join(visible))

    def visible_initial_cards(self, hands: List[Hand]) -> List[int]:
        if self.visibility == "none":
            return []
        cards: List[int] = []
        for index, hand in enumerate(hands):
            if index != self.your_index:
                cards.extend(hand.cards)
        return cards

    def play_your_hand(self, hand: Hand, dealer_up: int) -> bool:
        print("\n--- YOUR TURN ---")
        if hand.natural:
            hand.stood = True
            print("You have natural blackjack. It pays as a normal win.")
            self.hint("Agent hint: s")
            return True

        while True:
            total, usable = hand.total()
            if total > 21:
                hand.busted = True
                return True
            if total == 21:
                hand.stood = True
                print("You have 21. Stand.")
                return True

            print(f"\nYour hand: {hand.label()} vs dealer {format_card(dealer_up)}")
            best = self.current_best_action(hand, dealer_up)
            if self.auto:
                action = best
                print(f"AUTO action: {action}")
            else:
                rec = self.ask_recommendation(best)
                action = self.ask_action()
                if rec and action != rec:
                    self.off_policy += 1
                    print(f"WARNING: action {action} differs from typed recommendation {rec}.")
                    if self.strict and not self.confirm("Continue anyway?"):
                        continue

            if action == STAND:
                hand.stood = True
                self.hint("Agent hint: s")
                self.pause()
                return True

            if action == DOUBLE:
                if len(hand.cards) != 2 and not self.rules.double_after_hit:
                    print("Double is not legal now. Choose another action.")
                    continue
                card = self.draw()
                hand.cards.append(card)
                hand.doubled = True
                total, _ = hand.total()
                if total > 21:
                    hand.busted = True
                self._remove_known([card])
                print(f"Double card: {format_card(card)}")
                print(f"Your final hand: {hand.label()}")
                self.hint(f"Agent hint: x {format_card(card)}")
                self.pause()
                return True

            if action == HIT:
                card = self.draw()
                hand.cards.append(card)
                self._remove_known([card])
                print(f"Hit card: {format_card(card)}")
                print(f"Your hand: {hand.label()}")
                self.hint(f"Agent hint: h {format_card(card)}")
                self.pause()
                continue

    def ask_recommendation(self, best: str) -> Optional[str]:
        print("Enter your agent's recommendation, or press Enter to skip.")
        while True:
            prompt = "Agent recommendation [h/s/x or Enter]: "
            if self.hints:
                prompt = f"Agent recommendation [{best.lower()[0]} expected]: "
            raw = input(prompt).strip().lower()
            if raw == "":
                return None
            if raw == "q":
                raise SystemExit
            if raw in ACTION_ALIASES:
                return ACTION_ALIASES[raw]
            print("Use h, s, or x.")

    def ask_action(self) -> str:
        while True:
            raw = input("Your action [h/s/x]: ").strip().lower()
            if raw == "q":
                raise SystemExit
            if raw in ACTION_ALIASES:
                return ACTION_ALIASES[raw]
            print("Use h, s, or x.")

    def current_best_action(self, hand: Hand, dealer_up: int) -> str:
        total, usable = hand.total()
        decision = self.solver.decision(
            total,
            usable,
            dealer_up,
            self.agent_known_deck,
            player_natural=hand.natural,
            can_double=(len(hand.cards) == 2 or self.rules.double_after_hit),
        )
        return decision.action

    def play_other_hand(self, index: int, hand: Hand, dealer_up: int) -> None:
        if hand.natural:
            return

        while True:
            action = self.other_policy(hand, dealer_up)
            if action == STAND:
                hand.stood = True
                return
            if action == DOUBLE:
                card = self.draw()
                hand.cards.append(card)
                hand.doubled = True
                total, _ = hand.total()
                if total > 21:
                    hand.busted = True
                self.announce_other_card(index, "double", card)
                return

            card = self.draw()
            hand.cards.append(card)
            total, _ = hand.total()
            if total > 21:
                hand.busted = True
            self.announce_other_card(index, "hit", card)
            if hand.busted:
                return

    def other_policy(self, hand: Hand, dealer_up: int) -> str:
        total, usable = hand.total()
        if total >= 21:
            return STAND

        can_double = len(hand.cards) == 2
        if can_double and not usable:
            if total == 11 and dealer_up != 1:
                return DOUBLE
            if total == 10 and dealer_up in (2, 3, 4, 6, 7, 8, 9):
                return DOUBLE

        if usable:
            if can_double and total == 18 and dealer_up in (3, 4, 6):
                return DOUBLE
            if total <= 17:
                return HIT
            if total == 18 and dealer_up in (9, 10, 1):
                return HIT
            return STAND

        if total <= 11:
            return HIT
        if total == 12:
            return STAND if dealer_up in (4, 6) else HIT
        if 13 <= total <= 16:
            return STAND if dealer_up in (2, 3, 4, 6) else HIT
        return STAND

    def announce_other_card(self, index: int, action: str, card: int) -> None:
        if self.visibility != "all":
            return
        print(f"\nVisible: P{index + 1} {action} card {format_card(card)}")
        self.hint(f"Agent hint: o {format_card(card)}")
        self._remove_known([card])
        self.pause()

    def play_dealer(self, dealer: Hand) -> None:
        while True:
            total, usable = dealer.total()
            if self.dealer_should_stand(total, usable):
                return
            dealer.cards.append(self.draw())

    def dealer_should_stand(self, total: int, usable: bool) -> bool:
        if total > 17:
            return True
        if total < 17:
            return False
        return not (usable and self.rules.dealer_hits_soft_17)

    def score_round(self, hands: List[Hand], dealer: Hand) -> None:
        dealer_total, _ = dealer.total()
        dealer_natural = self._is_natural(dealer.cards)
        for hand in hands:
            total, _ = hand.total()
            reward = self.compare(total, dealer_total, hand.natural, dealer_natural)
            if hand.doubled:
                reward *= 2.0
            hand.reward = reward

    def compare(
        self, player_total: int, dealer_total: int, player_natural: bool, dealer_natural: bool
    ) -> float:
        if player_natural and dealer_natural:
            return 0.0
        if player_natural:
            return 1.0
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

    def reveal_round(self, hands: List[Hand], dealer: Hand) -> None:
        print("\n--- DEALER / ROUND RESULT ---")
        dealer_total, usable = dealer.total()
        kind = "soft" if usable else "hard"
        print(f"Dealer: {format_cards(dealer.cards)} = {dealer_total} ({kind})")
        self.hint(f"Agent hint: e {format_cards(dealer.cards)}")
        self.pause()
        self._remove_known(dealer.cards[1:])

        your_hand = hands[self.your_index]
        print(f"Your final hand: {your_hand.label()}")
        print(f"Round result: {your_hand.reward:+.1f}")
        print(f"Practice total: {self.total_score + your_hand.reward:+.1f}")

        if self.show_hidden_deck:
            print(f"Hidden true deck cards remaining: {len(self.deck)}")

    def print_summary(self) -> None:
        print("\n" + "=" * 72)
        print("PRACTICE SUMMARY")
        print("=" * 72)
        print(f"Total score: {self.total_score:+.1f}")
        print(f"Off-policy/mismatch warnings: {self.off_policy}")
        print("Use another seed or visibility mode for more practice.")

    def _remove_known(self, cards: Iterable[int]) -> None:
        for card in cards:
            try:
                self.agent_known_deck = decrement(self.agent_known_deck, card)
            except ValueError:
                pass

    def _is_natural(self, cards: List[int]) -> bool:
        return len(cards) == 2 and is_blackjack_pair(cards[0], cards[1])

    def confirm(self, prompt: str) -> bool:
        raw = input(f"{prompt} [y/N]: ").strip().lower()
        return raw in ("y", "yes")

    def pause(self, prompt: str = "Press Enter to continue: ") -> None:
        if self.auto:
            return
        raw = input(prompt).strip().lower()
        if raw == "q":
            raise SystemExit
        if raw == "deck":
            print(f"Cards remaining in hidden table deck: {len(self.deck)}")
            self.pause(prompt)
        if raw == "help":
            print("Practice controls: Enter continue, q quit, deck hidden count.")
            self.pause(prompt)


def self_test() -> None:
    rules = Rules(blackjack_payout=1.0)
    table = PracticeTable(
        players=5,
        seat=3,
        rounds=3,
        visibility="all",
        rules=rules,
        seed=7,
        auto=True,
        hints=True,
    )
    for i in range(3):
        table.round_no = i + 1
        assert table.play_round()
    assert isinstance(table.total_score, float)
    print("practice self-test passed")


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Practice finite-deck Blackjack table")
    parser.add_argument("--players", type=int, default=5, help="total table players")
    parser.add_argument("--seat", type=int, default=3, help="your seat number, 1-indexed")
    parser.add_argument("--rounds", type=int, default=10, help="number of rounds")
    parser.add_argument(
        "--visibility",
        choices=("none", "initial", "all"),
        default="all",
        help="which other-player cards are visible to you",
    )
    parser.add_argument("--dealer", choices=("s17", "h17"), default="s17")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--reshuffle-threshold", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="warn and reprompt on mismatch")
    parser.add_argument("--show-hidden-deck", action="store_true", help="debug: show hidden deck count")
    parser.add_argument("--hints", action="store_true", help="print exact-agent command hints")
    parser.add_argument("--classroom", action="store_true", help="use classroom-like defaults")
    parser.add_argument("--self-test", action="store_true", help="run noninteractive checks")
    parser.add_argument("--auto", action="store_true", help="noninteractive auto-play")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    if args.classroom:
        args.players = 5
        args.seat = 3
        args.rounds = 10
        args.visibility = "all"
        args.dealer = "s17"

    rules = Rules(
        blackjack_payout=1.0,
        dealer_hits_soft_17=args.dealer == "h17",
        shuffle_threshold=args.reshuffle_threshold,
    )

    table = PracticeTable(
        players=args.players,
        seat=args.seat,
        rounds=args.rounds,
        visibility=args.visibility,
        rules=rules,
        seed=args.seed,
        auto=args.auto,
        strict=args.strict,
        show_hidden_deck=args.show_hidden_deck,
        hints=args.hints,
    )
    try:
        table.run()
    except SystemExit:
        print("\nPractice stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
