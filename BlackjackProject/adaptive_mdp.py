#!/usr/bin/env python3
"""
Adaptive MDP solver for finite-deck Blackjack.

Computes optimal hit/stand policy based on current deck composition.
Uses value iteration with warm starts for efficiency.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from finite_deck_tracker import FiniteDeckTracker
from blackjack_game import card_value, evaluate_hand

# Actions
HIT = 0
STAND = 1


class AdaptiveMDP:
    """
    MDP solver that adapts to current deck state.

    The MDP state space is small (~180 states: player_sum x dealer_showing x usable_ace),
    but transition probabilities change based on deck composition.
    """

    def __init__(self, deck_tracker: Optional[FiniteDeckTracker] = None):
        """
        Initialize with a deck tracker.

        Args:
            deck_tracker: Tracker providing current deck probabilities.
                         If None, creates one with 5s removed.
        """
        if deck_tracker is None:
            deck_tracker = FiniteDeckTracker(removed_rank=5)
        self.deck_tracker = deck_tracker

        # Value function: V[player_sum][dealer_showing][usable_ace]
        # player_sum: 4-21, dealer_showing: 1-10, usable_ace: 0 or 1
        self.V = np.zeros((22, 11, 2))

        # Policy: same shape, stores HIT or STAND
        self.policy = np.zeros((22, 11, 2), dtype=int)

        # Cache for dealer outcome probabilities
        self._dealer_probs_cache: Dict[Tuple, Dict[int, float]] = {}

        # Track last deck state used for computation
        self._last_computed_state: Optional[Tuple[int, ...]] = None

    def _get_card_probs(self) -> Dict[int, float]:
        """Get current card draw probabilities from deck tracker."""
        return self.deck_tracker.get_draw_probabilities()

    def compute_dealer_probs(self, dealer_showing: int,
                             deck_state: Optional[Tuple[int, ...]] = None) -> Dict[int, float]:
        """
        Compute probability distribution of dealer's final hand value.

        Args:
            dealer_showing: Dealer's up-card value (1-10, where 1=Ace)
            deck_state: Optional deck state tuple for caching

        Returns:
            Dict mapping final values (17-21, or 22 for bust) to probabilities
        """
        if deck_state is None:
            deck_state = self.deck_tracker.get_state_tuple()

        cache_key = (dealer_showing, deck_state)
        if cache_key in self._dealer_probs_cache:
            return self._dealer_probs_cache[cache_key]

        # Probability distribution of final dealer values
        final_probs = {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0}

        # Use deck tracker for probabilities
        card_probs = self._get_card_probs()

        def dealer_play(current_sum: int, usable_ace: bool, prob: float):
            """Recursively simulate dealer play."""
            if prob < 1e-15:
                return

            if current_sum >= 17:
                if current_sum > 21:
                    final_probs[22] += prob
                else:
                    final_probs[current_sum] += prob
                return

            # Dealer must hit
            for value in range(1, 11):
                p = card_probs.get(value, 0.0)
                if p == 0:
                    continue

                new_prob = prob * p

                if value == 1:  # Ace
                    if current_sum + 11 <= 21:
                        dealer_play(current_sum + 11, True, new_prob)
                    else:
                        dealer_play(current_sum + 1, usable_ace, new_prob)
                else:
                    new_sum = current_sum + value
                    if new_sum > 21 and usable_ace:
                        new_sum -= 10
                        dealer_play(new_sum, False, new_prob)
                    else:
                        dealer_play(new_sum, usable_ace, new_prob)

        # Initialize dealer's hand with showing card
        if dealer_showing == 1:
            dealer_play(11, True, 1.0)
        else:
            dealer_play(dealer_showing, False, 1.0)

        self._dealer_probs_cache[cache_key] = final_probs
        return final_probs

    def expected_stand_value(self, player_sum: int, dealer_showing: int) -> float:
        """
        Compute expected value of standing with given hand vs dealer up-card.
        """
        dealer_probs = self.compute_dealer_probs(dealer_showing)
        expected_value = 0.0

        for dealer_final, prob in dealer_probs.items():
            if dealer_final == 22:  # Dealer bust
                expected_value += prob * 1.0
            elif player_sum > dealer_final:
                expected_value += prob * 1.0
            elif player_sum < dealer_final:
                expected_value += prob * (-1.0)
            # else: push, value = 0

        return expected_value

    def expected_hit_value(self, player_sum: int, dealer_showing: int,
                           usable_ace: bool) -> float:
        """
        Compute expected value of hitting given current state.
        """
        expected_value = 0.0
        card_probs = self._get_card_probs()

        for value in range(1, 11):
            p = card_probs.get(value, 0.0)
            if p == 0:
                continue

            if value == 1:  # Ace
                if player_sum + 11 <= 21:
                    new_sum = player_sum + 11
                    new_usable = True
                else:
                    new_sum = player_sum + 1
                    new_usable = usable_ace
            else:
                new_sum = player_sum + value
                new_usable = usable_ace

            # Handle bust or soft hand conversion
            if new_sum > 21:
                if new_usable:
                    new_sum -= 10
                    new_usable = False
                else:
                    expected_value += p * (-1.0)
                    continue

            # Get value from value function
            if new_sum <= 21:
                expected_value += p * self.V[new_sum, dealer_showing, int(new_usable)]

        return expected_value

    def value_iteration(self, epsilon: float = 1e-9, max_iterations: int = 100) -> int:
        """
        Run value iteration to compute optimal policy.

        Uses warm start from previous values for faster convergence.

        Returns:
            Number of iterations until convergence
        """
        # Clear dealer cache since probabilities may have changed
        self._dealer_probs_cache = {}

        for iteration in range(max_iterations):
            delta = 0.0

            for player_sum in range(4, 22):
                for dealer_showing in range(1, 11):
                    for usable_ace in [0, 1]:
                        if player_sum == 21:
                            # At 21, always stand
                            stand_val = self.expected_stand_value(player_sum, dealer_showing)
                            new_value = stand_val
                            action = STAND
                        else:
                            stand_val = self.expected_stand_value(player_sum, dealer_showing)
                            hit_val = self.expected_hit_value(player_sum, dealer_showing, bool(usable_ace))

                            if hit_val > stand_val:
                                new_value = hit_val
                                action = HIT
                            else:
                                new_value = stand_val
                                action = STAND

                        old_value = self.V[player_sum, dealer_showing, usable_ace]
                        delta = max(delta, abs(new_value - old_value))

                        self.V[player_sum, dealer_showing, usable_ace] = new_value
                        self.policy[player_sum, dealer_showing, usable_ace] = action

            if delta < epsilon:
                self._last_computed_state = self.deck_tracker.get_state_tuple()
                return iteration + 1

        self._last_computed_state = self.deck_tracker.get_state_tuple()
        return max_iterations

    def needs_recomputation(self, threshold: int = 35) -> bool:
        """
        Determine if policy recomputation is needed.

        Recompute when:
        - Never computed before
        - Deck state has changed
        - Deck is significantly depleted

        Args:
            threshold: Recompute if fewer cards remain
        """
        if self._last_computed_state is None:
            return True

        current_state = self.deck_tracker.get_state_tuple()
        if current_state != self._last_computed_state:
            return True

        return False

    def compute_optimal_policy(self, force: bool = False) -> np.ndarray:
        """
        Compute optimal policy for current deck state.

        Args:
            force: If True, always recompute regardless of cache

        Returns:
            Policy array [player_sum, dealer_showing, usable_ace] -> HIT/STAND
        """
        if force or self.needs_recomputation():
            self.value_iteration()
        return self.policy

    def get_decision(self, player_sum: int, dealer_showing: int,
                     usable_ace: bool, recompute: bool = True) -> str:
        """
        Get optimal decision for current state.

        Args:
            player_sum: Player's current hand value
            dealer_showing: Dealer's up card value (1-10)
            usable_ace: Whether player has a usable ace
            recompute: If True, recompute policy first

        Returns:
            "HIT" or "STAND"
        """
        if recompute:
            self.compute_optimal_policy()

        action = self.policy[player_sum, dealer_showing, int(usable_ace)]
        return "HIT" if action == HIT else "STAND"

    def get_decision_with_ev(self, player_sum: int, dealer_showing: int,
                              usable_ace: bool) -> Tuple[str, float, float]:
        """
        Get optimal decision with expected values for both actions.

        Returns:
            (decision, hit_ev, stand_ev)
        """
        self.compute_optimal_policy()

        stand_ev = self.expected_stand_value(player_sum, dealer_showing)
        hit_ev = self.expected_hit_value(player_sum, dealer_showing, usable_ace)

        decision = "HIT" if hit_ev > stand_ev else "STAND"
        return decision, hit_ev, stand_ev

    def get_value(self, player_sum: int, dealer_showing: int, usable_ace: bool) -> float:
        """Get value of a state under optimal policy."""
        self.compute_optimal_policy()
        return self.V[player_sum, dealer_showing, int(usable_ace)]

    def print_policy(self, usable_ace: bool = False):
        """Print the optimal policy as a grid."""
        self.compute_optimal_policy()

        ace_str = "Soft" if usable_ace else "Hard"
        print(f"\n{ace_str} Hands Policy (H=Hit, S=Stand):")
        print("Player\\Dealer  ", end="")
        for d in range(2, 11):
            print(f"{d:3}", end="")
        print("  A")

        for p in range(21, 3, -1):
            print(f"    {p:2}         ", end="")
            for d in range(2, 11):
                action = "H" if self.policy[p, d, int(usable_ace)] == HIT else "S"
                print(f"  {action}", end="")
            action = "H" if self.policy[p, 1, int(usable_ace)] == HIT else "S"
            print(f"  {action}")


def main():
    """Test the adaptive MDP solver."""
    print("Testing AdaptiveMDP with 5s removed...")

    tracker = FiniteDeckTracker(removed_rank=5)
    mdp = AdaptiveMDP(tracker)

    iterations = mdp.value_iteration()
    print(f"Converged in {iterations} iterations")

    # Test a few decisions
    print("\nSample decisions (full deck, no 5s):")
    test_cases = [
        (16, 10, False, "Hard 16 vs 10"),
        (16, 6, False, "Hard 16 vs 6"),
        (12, 3, False, "Hard 12 vs 3"),
        (17, 10, True, "Soft 17 vs 10"),
        (18, 10, True, "Soft 18 vs 10"),
    ]

    for player_sum, dealer, usable_ace, desc in test_cases:
        decision, hit_ev, stand_ev = mdp.get_decision_with_ev(player_sum, dealer, usable_ace)
        print(f"  {desc}: {decision} (Hit EV: {hit_ev:+.4f}, Stand EV: {stand_ev:+.4f})")

    mdp.print_policy(usable_ace=False)
    mdp.print_policy(usable_ace=True)

    # Test with depleted deck
    print("\n\nTesting with depleted deck (many 10s removed)...")
    for _ in range(8):
        tracker.remove_card(10)

    mdp.value_iteration()
    print(f"Deck state: {tracker.get_total_remaining()} cards, 10s remaining: {tracker.get_remaining_count(10)}")

    print("\nDecisions with fewer 10s:")
    for player_sum, dealer, usable_ace, desc in test_cases:
        decision, hit_ev, stand_ev = mdp.get_decision_with_ev(player_sum, dealer, usable_ace)
        print(f"  {desc}: {decision} (Hit EV: {hit_ev:+.4f}, Stand EV: {stand_ev:+.4f})")


if __name__ == "__main__":
    main()
