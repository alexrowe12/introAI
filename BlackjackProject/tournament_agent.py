#!/usr/bin/env python3
"""
Tournament agent for live Blackjack play.

Interactive CLI for card counting and optimal decision making
in a finite-deck Blackjack tournament.

Usage:
    python tournament_agent.py
"""

import sys
import readline  # For command history
from typing import List, Tuple, Optional
from finite_deck_tracker import FiniteDeckTracker, parse_card_input, parse_multiple_cards
from adaptive_mdp import AdaptiveMDP
from blackjack_game import evaluate_hand


class TournamentAgent:
    """
    Command-line interface for live Blackjack tournament play.
    """

    def __init__(self, removed_rank: int = 5):
        """Initialize agent with tournament deck configuration."""
        self.removed_rank = removed_rank
        self.deck_tracker = FiniteDeckTracker(removed_rank)
        self.mdp = AdaptiveMDP(self.deck_tracker)

        # Pre-compute initial policy
        print("Computing optimal strategy for initial deck...")
        self.mdp.value_iteration()
        print("Ready!\n")

        # Current hand state
        self.player_cards: List[int] = []
        self.dealer_up: Optional[int] = None
        self.hand_active = False

        # Session statistics
        self.hands_played = 0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.total_result = 0.0

        # History for undo
        self.card_history: List[int] = []

    def _evaluate_player_hand(self) -> Tuple[int, bool]:
        """Evaluate current player hand."""
        return evaluate_hand(self.player_cards)

    def _show_decision(self) -> str:
        """Get and display optimal decision for current hand state."""
        if not self.hand_active or self.dealer_up is None:
            return ""

        player_sum, usable_ace = self._evaluate_player_hand()

        if player_sum > 21:
            return "BUST"

        if player_sum == 21:
            return "STAND (21!)"

        decision, hit_ev, stand_ev = self.mdp.get_decision_with_ev(
            player_sum, self.dealer_up, usable_ace
        )

        ev_diff = abs(hit_ev - stand_ev)
        confidence = "strong" if ev_diff > 0.1 else "marginal" if ev_diff > 0.02 else "close"

        return f"{decision} ({confidence}, H:{hit_ev:+.3f} S:{stand_ev:+.3f})"

    def _format_hand(self) -> str:
        """Format current hand for display."""
        if not self.player_cards:
            return "No cards"

        card_names = []
        for v in self.player_cards:
            if v == 1:
                card_names.append('A')
            elif v == 10:
                card_names.append('10')
            else:
                card_names.append(str(v))

        player_sum, usable_ace = self._evaluate_player_hand()
        hand_type = "soft" if usable_ace else "hard"

        return f"{' + '.join(card_names)} = {player_sum} ({hand_type})"

    def start_hand(self, player_card1: int, player_card2: int, dealer_up: int) -> str:
        """
        Start a new hand with initial deal.

        Args:
            player_card1, player_card2: Player's initial cards (1-10)
            dealer_up: Dealer's showing card (1-10)

        Returns:
            Formatted string with hand info and first decision
        """
        # End previous hand if active
        if self.hand_active:
            self._end_hand_internal()

        self.player_cards = [player_card1, player_card2]
        self.dealer_up = dealer_up
        self.hand_active = True
        self.card_history = [player_card1, player_card2, dealer_up]

        # Remove cards from deck
        self.deck_tracker.remove_card(player_card1)
        self.deck_tracker.remove_card(player_card2)
        self.deck_tracker.remove_card(dealer_up)

        # Recompute policy with updated deck
        self.mdp.value_iteration()

        # Check for blackjack
        player_sum, _ = self._evaluate_player_hand()
        if player_sum == 21:
            return self._format_status() + "\n>>> BLACKJACK! Stand and see dealer."

        return self._format_status()

    def _format_status(self) -> str:
        """Format current game status."""
        lines = []
        lines.append(f"\nYour hand: {self._format_hand()}")
        lines.append(f"Dealer shows: {'A' if self.dealer_up == 1 else self.dealer_up}")
        lines.append(f"Deck: {self.deck_tracker.get_total_remaining()} cards")
        lines.append(f"\n>>> {self._show_decision()}")
        return '\n'.join(lines)

    def hit(self, new_card: int) -> str:
        """
        Record a hit and get next decision.

        Args:
            new_card: Card received from hit (1-10)

        Returns:
            Status string with decision
        """
        if not self.hand_active:
            return "No active hand. Use 'new' to start a hand."

        self.player_cards.append(new_card)
        self.card_history.append(new_card)
        self.deck_tracker.remove_card(new_card)

        # Recompute policy
        self.mdp.value_iteration()

        player_sum, _ = self._evaluate_player_hand()

        if player_sum > 21:
            self.hand_active = False
            self.hands_played += 1
            self.losses += 1
            self.total_result -= 1.0
            return f"\nYour hand: {self._format_hand()}\n>>> BUST! You lose."

        return self._format_status()

    def stand(self) -> str:
        """Confirm standing."""
        if not self.hand_active:
            return "No active hand."

        return "Standing. Enter dealer cards when revealed: end <cards...>"

    def end_hand(self, dealer_cards: List[int], outcome: Optional[str] = None) -> str:
        """
        End current hand and record dealer cards.

        Args:
            dealer_cards: All dealer cards (including up card and hole card)
            outcome: Optional manual outcome ("win", "lose", "push")
        """
        # If hand already ended (bust), just record dealer cards for deck tracking
        if not self.hand_active:
            if not dealer_cards:
                return "No active hand."
            # Record dealer cards for deck tracking only
            for card in dealer_cards:
                self.deck_tracker.remove_card(card)
            self.mdp.value_iteration()
            dealer_card_str = ' + '.join('A' if c == 1 else str(c) for c in dealer_cards)
            return f"Recorded dealer cards: {dealer_card_str} (hand already ended)"

        # Remove dealer cards from deck (excluding up card already removed)
        for i, card in enumerate(dealer_cards):
            if i == 0 and card == self.dealer_up:
                continue  # Skip up card, already removed
            self.deck_tracker.remove_card(card)
            self.card_history.append(card)

        # Calculate dealer total
        dealer_sum, _ = evaluate_hand(dealer_cards)

        player_sum, _ = self._evaluate_player_hand()

        # Determine outcome
        if outcome:
            outcome = outcome.lower()
        else:
            if player_sum > 21:
                outcome = "lose"
            elif dealer_sum > 21:
                outcome = "win"
            elif player_sum > dealer_sum:
                outcome = "win"
            elif player_sum < dealer_sum:
                outcome = "lose"
            else:
                outcome = "push"

        # Record statistics
        self.hands_played += 1
        if outcome == "win":
            self.wins += 1
            # Check for blackjack (3:2 payout)
            if len(self.player_cards) == 2 and player_sum == 21:
                self.total_result += 1.5
                result_str = "+1.5 (Blackjack!)"
            else:
                self.total_result += 1.0
                result_str = "+1.0"
        elif outcome == "lose":
            self.losses += 1
            self.total_result -= 1.0
            result_str = "-1.0"
        else:
            self.pushes += 1
            result_str = "0 (Push)"

        self._end_hand_internal()

        dealer_card_str = ' + '.join('A' if c == 1 else str(c) for c in dealer_cards)
        return (f"\nDealer: {dealer_card_str} = {dealer_sum}"
                f"\nResult: {outcome.upper()} ({result_str})"
                f"\nSession: {self.total_result:+.1f} over {self.hands_played} hands")

    def _end_hand_internal(self):
        """Clean up hand state."""
        self.player_cards = []
        self.dealer_up = None
        self.hand_active = False
        self.card_history = []

    def record_cards(self, cards: List[int]) -> str:
        """
        Record cards seen (e.g., from other plays or burns).

        Args:
            cards: List of card values to remove from deck
        """
        for card in cards:
            self.deck_tracker.remove_card(card)

        # Recompute policy
        self.mdp.value_iteration()

        card_str = ', '.join('A' if c == 1 else str(c) for c in cards)
        return f"Recorded {len(cards)} cards: {card_str}\nDeck: {self.deck_tracker.get_total_remaining()} remaining"

    def undo_last(self) -> str:
        """Undo last card input."""
        if not self.card_history:
            return "Nothing to undo."

        last_card = self.card_history.pop()
        self.deck_tracker.add_card(last_card)

        if self.player_cards and self.player_cards[-1] == last_card:
            self.player_cards.pop()
        elif self.dealer_up == last_card:
            self.dealer_up = None

        self.mdp.value_iteration()

        return f"Undid card: {'A' if last_card == 1 else last_card}"

    def show_deck(self) -> str:
        """Display current deck composition."""
        lines = ["\n=== Deck Status ==="]
        lines.append(f"Remaining: {self.deck_tracker.get_total_remaining()} cards")
        lines.append("")

        probs = self.deck_tracker.get_draw_probabilities()
        initial = self.deck_tracker.initial_counts

        for v in range(1, 11):
            if v == self.removed_rank:
                continue
            count = self.deck_tracker.get_remaining_count(v)
            init = initial.get(v, 0)
            prob = probs.get(v, 0) * 100
            name = 'A' if v == 1 else ('10/J/Q/K' if v == 10 else str(v))
            lines.append(f"  {name:8}: {count:2}/{init:2} ({prob:5.1f}%)")

        return '\n'.join(lines)

    def show_stats(self) -> str:
        """Display session statistics."""
        lines = ["\n=== Session Statistics ==="]
        lines.append(f"Hands played: {self.hands_played}")
        lines.append(f"Wins: {self.wins} | Losses: {self.losses} | Pushes: {self.pushes}")
        lines.append(f"Net result: {self.total_result:+.1f}")
        if self.hands_played > 0:
            avg = self.total_result / self.hands_played
            lines.append(f"Average per hand: {avg:+.3f}")
        return '\n'.join(lines)

    def reset_deck(self) -> str:
        """Reset to fresh deck."""
        self.deck_tracker.reset()
        self.mdp.value_iteration()
        self._end_hand_internal()
        return "Deck reset to full 48 cards."

    def show_policy(self) -> str:
        """Show current optimal policy grids."""
        import io
        import sys

        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        self.mdp.print_policy(usable_ace=False)
        self.mdp.print_policy(usable_ace=True)

        output = buffer.getvalue()
        sys.stdout = old_stdout

        return output

    def show_help(self) -> str:
        """Show help message."""
        return """
=== Blackjack Tournament Agent ===

COMMANDS:
  new <p1> <p2> <dealer>  Start new hand
                          Example: new 7 9 6  or  new A K 10

  h <card> / hit <card>   Record a hit
                          Example: h 5  or  hit K

  s / stand               Confirm standing

  end <d1> <d2> ...       End hand with dealer's cards
                          Example: end 7 K  or  end 6 3 J

  see <cards>             Record cards seen (burns, other hands)
                          Example: see 2 7 K A

  deck                    Show remaining deck composition
  stats                   Show session statistics
  policy                  Show current optimal strategy charts
  undo                    Undo last card input
  reset                   Reset to fresh deck (new shoe)
  help                    Show this help
  quit / q                Exit

CARD INPUT:
  A or 1      = Ace
  2-9         = Number cards
  10 or T     = Ten
  J, Q, K     = Face cards (all count as 10)
"""

    def run_interactive(self):
        """Run interactive CLI loop."""
        print(self.show_help())
        print(f"Deck: {self.deck_tracker.get_total_remaining()} cards (no {self.removed_rank}s)")
        print()

        while True:
            try:
                prompt = "hand> " if self.hand_active else "> "
                line = input(prompt).strip()

                if not line:
                    continue

                parts = line.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if cmd in ('quit', 'q', 'exit'):
                    print("Goodbye!")
                    break

                elif cmd == 'help':
                    print(self.show_help())

                elif cmd == 'new':
                    try:
                        cards = parse_multiple_cards(args)
                        if len(cards) != 3:
                            print("Usage: new <player1> <player2> <dealer>")
                            continue
                        print(self.start_hand(cards[0], cards[1], cards[2]))
                    except ValueError as e:
                        print(f"Error: {e}")

                elif cmd in ('h', 'hit'):
                    try:
                        card = parse_card_input(args)
                        print(self.hit(card))
                    except ValueError as e:
                        print(f"Error: {e}")

                elif cmd in ('s', 'stand'):
                    print(self.stand())

                elif cmd == 'end':
                    try:
                        cards = parse_multiple_cards(args)
                        if len(cards) < 2:
                            print("Usage: end <dealer_cards...> (at least 2 cards)")
                            continue
                        print(self.end_hand(cards))
                    except ValueError as e:
                        print(f"Error: {e}")

                elif cmd == 'see':
                    try:
                        cards = parse_multiple_cards(args)
                        print(self.record_cards(cards))
                    except ValueError as e:
                        print(f"Error: {e}")

                elif cmd == 'deck':
                    print(self.show_deck())

                elif cmd == 'stats':
                    print(self.show_stats())

                elif cmd == 'policy':
                    print(self.show_policy())

                elif cmd == 'undo':
                    print(self.undo_last())

                elif cmd == 'reset':
                    print(self.reset_deck())

                else:
                    print(f"Unknown command: {cmd}. Type 'help' for commands.")

            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")


def main():
    """Run the tournament agent."""
    print("=" * 50)
    print("  BLACKJACK TOURNAMENT AGENT")
    print("  Finite Deck Card Counter")
    print("=" * 50)
    print()

    agent = TournamentAgent(removed_rank=5)
    agent.run_interactive()


if __name__ == "__main__":
    main()
