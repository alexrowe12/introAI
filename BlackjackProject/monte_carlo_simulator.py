#!/usr/bin/env python3
"""
Monte Carlo simulator for finite-deck Blackjack.

Simulates games to evaluate strategies and gather statistics.
Useful for comparing adaptive vs basic strategy approaches.

Usage:
    python monte_carlo_simulator.py
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass
from finite_deck_tracker import FiniteDeckTracker
from adaptive_mdp import AdaptiveMDP
from blackjack_game import evaluate_hand


@dataclass
class HandResult:
    """Result of a single hand."""
    player_cards: List[int]
    dealer_cards: List[int]
    player_total: int
    dealer_total: int
    outcome: str  # "win", "lose", "push", "blackjack"
    reward: float
    deck_remaining_at_start: int


class MonteCarloSimulator:
    """
    Monte Carlo simulator for finite-deck Blackjack.
    """

    def __init__(self, removed_rank: int = 5, seed: Optional[int] = None):
        """
        Initialize simulator.

        Args:
            removed_rank: Rank removed from deck
            seed: Random seed for reproducibility
        """
        self.removed_rank = removed_rank
        self.rng = np.random.default_rng(seed)

    def _draw_card(self, deck: FiniteDeckTracker) -> int:
        """Draw a random card from the deck."""
        probs = deck.get_draw_probabilities()
        values = list(range(1, 11))
        probabilities = [probs[v] for v in values]

        # Normalize in case of floating point issues
        total = sum(probabilities)
        if total == 0:
            raise ValueError("Deck is empty")
        probabilities = [p / total for p in probabilities]

        return self.rng.choice(values, p=probabilities)

    def _deal_initial(self, deck: FiniteDeckTracker) -> Tuple[List[int], int]:
        """Deal initial cards: 2 to player, 1 up card to dealer."""
        player_cards = [self._draw_card(deck), self._draw_card(deck)]
        deck.remove_card(player_cards[0])
        deck.remove_card(player_cards[1])

        dealer_up = self._draw_card(deck)
        deck.remove_card(dealer_up)

        return player_cards, dealer_up

    def _play_dealer(self, deck: FiniteDeckTracker, up_card: int) -> List[int]:
        """Play out dealer's hand according to standard rules."""
        cards = [up_card]

        # Deal hole card
        hole = self._draw_card(deck)
        deck.remove_card(hole)
        cards.append(hole)

        # Dealer hits until 17+
        while True:
            total, _ = evaluate_hand(cards)
            if total >= 17:
                break
            new_card = self._draw_card(deck)
            deck.remove_card(new_card)
            cards.append(new_card)

        return cards

    def simulate_hand(self, deck: FiniteDeckTracker,
                      strategy: Callable[[int, int, bool, FiniteDeckTracker], str]) -> HandResult:
        """
        Simulate a single hand.

        Args:
            deck: Current deck state (will be modified)
            strategy: Function(player_sum, dealer_up, usable_ace, deck) -> "HIT"/"STAND"

        Returns:
            HandResult with details and outcome
        """
        deck_at_start = deck.get_total_remaining()

        # Deal initial cards
        player_cards, dealer_up = self._deal_initial(deck)

        # Check for player blackjack
        player_sum, usable_ace = evaluate_hand(player_cards)

        if player_sum == 21:
            # Player blackjack - check dealer
            dealer_cards = self._play_dealer(deck, dealer_up)
            dealer_sum, _ = evaluate_hand(dealer_cards)

            if dealer_sum == 21 and len(dealer_cards) == 2:
                # Both blackjack - push
                return HandResult(
                    player_cards=player_cards, dealer_cards=dealer_cards,
                    player_total=21, dealer_total=21,
                    outcome="push", reward=0.0,
                    deck_remaining_at_start=deck_at_start
                )
            else:
                # Player blackjack wins
                return HandResult(
                    player_cards=player_cards, dealer_cards=dealer_cards,
                    player_total=21, dealer_total=dealer_sum,
                    outcome="blackjack", reward=1.5,
                    deck_remaining_at_start=deck_at_start
                )

        # Player's turn
        while player_sum < 21:
            decision = strategy(player_sum, dealer_up, usable_ace, deck)
            if decision == "STAND":
                break

            # Hit
            new_card = self._draw_card(deck)
            deck.remove_card(new_card)
            player_cards.append(new_card)
            player_sum, usable_ace = evaluate_hand(player_cards)

        # Check for bust
        if player_sum > 21:
            # Player busts - dealer doesn't need to play
            return HandResult(
                player_cards=player_cards, dealer_cards=[dealer_up],
                player_total=player_sum, dealer_total=0,
                outcome="lose", reward=-1.0,
                deck_remaining_at_start=deck_at_start
            )

        # Dealer's turn
        dealer_cards = self._play_dealer(deck, dealer_up)
        dealer_sum, _ = evaluate_hand(dealer_cards)

        # Determine outcome
        if dealer_sum > 21:
            outcome = "win"
            reward = 1.0
        elif player_sum > dealer_sum:
            outcome = "win"
            reward = 1.0
        elif player_sum < dealer_sum:
            outcome = "lose"
            reward = -1.0
        else:
            outcome = "push"
            reward = 0.0

        return HandResult(
            player_cards=player_cards, dealer_cards=dealer_cards,
            player_total=player_sum, dealer_total=dealer_sum,
            outcome=outcome, reward=reward,
            deck_remaining_at_start=deck_at_start
        )

    def simulate_shoe(self, strategy: Callable,
                      min_cards_before_reshuffle: int = 10) -> List[HandResult]:
        """
        Simulate playing through entire deck/shoe.

        Args:
            strategy: Decision function
            min_cards_before_reshuffle: Reshuffle when this few cards remain

        Returns:
            List of hand results
        """
        deck = FiniteDeckTracker(removed_rank=self.removed_rank)
        results = []

        while deck.get_total_remaining() >= min_cards_before_reshuffle:
            try:
                result = self.simulate_hand(deck, strategy)
                results.append(result)
            except ValueError:
                # Deck exhausted mid-hand
                break

        return results

    def simulate_n_shoes(self, n: int, strategy: Callable,
                         min_cards: int = 10) -> Dict:
        """
        Simulate n complete shoes and aggregate statistics.

        Returns:
            Statistics dict with expected return, variance, etc.
        """
        all_results = []
        shoe_returns = []

        for _ in range(n):
            results = self.simulate_shoe(strategy, min_cards)
            all_results.extend(results)

            shoe_return = sum(r.reward for r in results)
            shoe_returns.append(shoe_return)

        total_hands = len(all_results)
        total_reward = sum(r.reward for r in all_results)

        wins = sum(1 for r in all_results if r.outcome in ("win", "blackjack"))
        losses = sum(1 for r in all_results if r.outcome == "lose")
        pushes = sum(1 for r in all_results if r.outcome == "push")
        blackjacks = sum(1 for r in all_results if r.outcome == "blackjack")

        return {
            "shoes": n,
            "total_hands": total_hands,
            "total_reward": total_reward,
            "expected_return_per_hand": total_reward / total_hands if total_hands > 0 else 0,
            "expected_return_per_shoe": np.mean(shoe_returns),
            "std_per_shoe": np.std(shoe_returns),
            "win_rate": wins / total_hands if total_hands > 0 else 0,
            "loss_rate": losses / total_hands if total_hands > 0 else 0,
            "push_rate": pushes / total_hands if total_hands > 0 else 0,
            "blackjack_rate": blackjacks / total_hands if total_hands > 0 else 0,
            "hands_per_shoe": total_hands / n if n > 0 else 0,
        }

    def compare_strategies(self, strategies: Dict[str, Callable],
                          n_shoes: int = 1000) -> Dict[str, Dict]:
        """
        Compare multiple strategies head-to-head.

        Args:
            strategies: Dict mapping strategy name to decision function
            n_shoes: Number of shoes to simulate per strategy

        Returns:
            Dict mapping strategy name to statistics
        """
        results = {}

        for name, strategy in strategies.items():
            print(f"Simulating {name}...")
            stats = self.simulate_n_shoes(n_shoes, strategy)
            results[name] = stats
            print(f"  EV/hand: {stats['expected_return_per_hand']:+.4f}")

        return results


# ============================================================
# Predefined Strategies
# ============================================================

def create_basic_strategy() -> Callable:
    """
    Create basic strategy for no-5s deck.

    Precomputes optimal policy for full deck and uses it without adaptation.
    """
    # Compute policy for full deck
    deck = FiniteDeckTracker(removed_rank=5)
    mdp = AdaptiveMDP(deck)
    mdp.value_iteration()

    # Copy policy
    policy = mdp.policy.copy()

    def strategy(player_sum: int, dealer_up: int, usable_ace: bool,
                 deck: FiniteDeckTracker) -> str:
        if player_sum >= 21:
            return "STAND"
        action = policy[player_sum, dealer_up, int(usable_ace)]
        return "HIT" if action == 0 else "STAND"

    return strategy


def create_adaptive_strategy() -> Callable:
    """
    Create adaptive strategy that recomputes policy based on deck state.
    """
    mdp_cache = {}

    def strategy(player_sum: int, dealer_up: int, usable_ace: bool,
                 deck: FiniteDeckTracker) -> str:
        if player_sum >= 21:
            return "STAND"

        deck_state = deck.get_state_tuple()

        if deck_state not in mdp_cache:
            # Create a COPY of the deck for the MDP to use
            deck_copy = deck.copy()
            mdp = AdaptiveMDP(deck_copy)
            mdp.value_iteration()
            mdp_cache[deck_state] = mdp.policy.copy()

            # Limit cache size
            if len(mdp_cache) > 1000:
                # Remove oldest entries
                keys = list(mdp_cache.keys())[:500]
                for k in keys:
                    del mdp_cache[k]

        policy = mdp_cache[deck_state]
        action = policy[player_sum, dealer_up, int(usable_ace)]
        return "HIT" if action == 0 else "STAND"

    return strategy


def create_simple_threshold_strategy(stand_on: int = 17) -> Callable:
    """
    Create simple threshold strategy: stand on N or higher.
    """
    def strategy(player_sum: int, dealer_up: int, usable_ace: bool,
                 deck: FiniteDeckTracker) -> str:
        if player_sum >= stand_on:
            return "STAND"
        return "HIT"

    return strategy


def create_hi_lo_strategy() -> Callable:
    """
    Create strategy using Hi-Lo card counting.

    Running count: +1 for 2-6, 0 for 7-9, -1 for 10-A
    Adjusts basic strategy based on count.
    """
    # Base policy
    deck = FiniteDeckTracker(removed_rank=5)
    mdp = AdaptiveMDP(deck)
    mdp.value_iteration()
    base_policy = mdp.policy.copy()

    def get_hi_lo_count(deck: FiniteDeckTracker) -> int:
        """Compute running count based on cards dealt."""
        initial = deck.initial_counts
        remaining = deck.remaining_counts

        count = 0
        # Low cards (2-6): +1 when dealt
        for v in [2, 3, 4, 6]:  # 5 is removed
            dealt = initial.get(v, 0) - remaining.get(v, 0)
            count += dealt

        # High cards (10, A): -1 when dealt
        for v in [1, 10]:
            dealt = initial.get(v, 0) - remaining.get(v, 0)
            count -= dealt

        return count

    def strategy(player_sum: int, dealer_up: int, usable_ace: bool,
                 deck: FiniteDeckTracker) -> str:
        if player_sum >= 21:
            return "STAND"

        count = get_hi_lo_count(deck)
        base_action = base_policy[player_sum, dealer_up, int(usable_ace)]

        # High count (many high cards remaining) favors standing
        # Low count (many low cards remaining) favors hitting
        # These are common deviations from basic strategy

        # Insurance/surrender indices - simplified version
        if count >= 3:
            # With high count, stand more often on stiff hands vs low dealer
            if player_sum == 16 and dealer_up in [9, 10]:
                return "STAND"
            if player_sum == 15 and dealer_up == 10:
                return "STAND"
            if player_sum == 12 and dealer_up in [2, 3]:
                return "STAND"

        if count <= -2:
            # With low count, hit more often
            if player_sum == 12 and dealer_up in [4, 5, 6]:
                return "HIT"
            if player_sum == 13 and dealer_up in [2, 3]:
                return "HIT"

        return "HIT" if base_action == 0 else "STAND"

    return strategy


def main():
    """Run strategy comparison."""
    print("=" * 60)
    print("  MONTE CARLO STRATEGY COMPARISON")
    print("  Finite Deck Blackjack (no 5s)")
    print("=" * 60)
    print()

    simulator = MonteCarloSimulator(removed_rank=5, seed=42)

    strategies = {
        "Basic Strategy": create_basic_strategy(),
        "Adaptive MDP": create_adaptive_strategy(),
        "Hi-Lo Counting": create_hi_lo_strategy(),
        "Stand on 17": create_simple_threshold_strategy(17),
    }

    n_shoes = 500
    print(f"Simulating {n_shoes} shoes per strategy...\n")

    results = simulator.compare_strategies(strategies, n_shoes=n_shoes)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Strategy':<20} {'EV/Hand':>10} {'EV/Shoe':>10} {'Win%':>8} {'BJ%':>6}")
    print("-" * 60)

    # Sort by EV/hand
    sorted_names = sorted(results.keys(),
                          key=lambda k: results[k]['expected_return_per_hand'],
                          reverse=True)

    for name in sorted_names:
        r = results[name]
        print(f"{name:<20} {r['expected_return_per_hand']:>+10.4f} "
              f"{r['expected_return_per_shoe']:>+10.2f} "
              f"{r['win_rate']*100:>7.1f}% "
              f"{r['blackjack_rate']*100:>5.1f}%")

    print("-" * 60)

    # Show improvement from adaptive over basic
    basic_ev = results["Basic Strategy"]['expected_return_per_hand']
    adaptive_ev = results["Adaptive MDP"]['expected_return_per_hand']
    improvement = adaptive_ev - basic_ev

    print(f"\nAdaptive vs Basic improvement: {improvement:+.4f} per hand")
    print(f"Over {n_shoes} shoes ({int(results['Basic Strategy']['total_hands'])} hands):")
    print(f"  Basic total: {results['Basic Strategy']['total_reward']:+.1f}")
    print(f"  Adaptive total: {results['Adaptive MDP']['total_reward']:+.1f}")


if __name__ == "__main__":
    main()
