#!/usr/bin/env python3
"""
Blackjack MDP model and value iteration solver.

Models infinite-deck Blackjack as a Markov Decision Process and computes
the optimal player policy using value iteration.

State: (player_sum, dealer_showing, usable_ace)
Actions: Hit, Stand
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from blackjack_game import card_value, evaluate_hand, compare_hands

# Actions
HIT = 0
STAND = 1


class BlackjackMDP:
    """
    MDP model for infinite-deck Blackjack.

    Attributes:
        card_probs: Dict mapping rank (1-13) to draw probability
        removed_rank: Which rank was removed (None for standard deck)
    """

    def __init__(self, removed_rank: Optional[int] = None):
        """
        Initialize the MDP with optional rank removal.

        Args:
            removed_rank: Rank to remove (1-13), or None for standard deck
        """
        self.removed_rank = removed_rank
        self.card_probs = self._build_card_probs(removed_rank)

        # Cache for dealer outcome probabilities
        self._dealer_probs_cache: Dict[int, Dict[int, float]] = {}

        # State values: V[player_sum][dealer_showing][usable_ace]
        # player_sum: 4-21, dealer_showing: 1-10, usable_ace: 0 or 1
        self.V = np.zeros((22, 11, 2))

        # Policy: same shape, stores HIT or STAND
        self.policy = np.zeros((22, 11, 2), dtype=int)

    def _build_card_probs(self, removed_rank: Optional[int]) -> Dict[int, float]:
        """Build probability distribution over card ranks."""
        probs = {}
        if removed_rank is None:
            # Standard deck: each rank has probability 1/13
            for rank in range(1, 14):
                probs[rank] = 1.0 / 13.0
        else:
            # One rank removed: remaining ranks have probability 1/12
            for rank in range(1, 14):
                if rank == removed_rank:
                    probs[rank] = 0.0
                else:
                    probs[rank] = 1.0 / 12.0
        return probs

    def compute_dealer_probs(self, dealer_showing: int) -> Dict[int, float]:
        """
        Compute probability distribution of dealer's final hand value.

        Args:
            dealer_showing: Dealer's up-card value (1-10, where 1=Ace)

        Returns:
            Dict mapping final values (17-21, or 22 for bust) to probabilities
        """
        if dealer_showing in self._dealer_probs_cache:
            return self._dealer_probs_cache[dealer_showing]

        # Use recursive probability calculation
        # State: (current_sum, usable_ace)
        # We compute P(final_value | current_state) via dynamic programming

        # final_probs[value] = probability of ending at that value
        final_probs = {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0}  # 22 = bust

        def dealer_play(current_sum: int, usable_ace: bool, prob: float):
            """Recursively simulate dealer play."""
            if prob < 1e-15:  # Pruning for numerical stability
                return

            if current_sum >= 17:
                # Dealer stands
                if current_sum > 21:
                    final_probs[22] += prob  # Bust
                else:
                    final_probs[current_sum] += prob
                return

            # Dealer must hit
            for rank in range(1, 14):
                card_prob = self.card_probs[rank]
                if card_prob == 0:
                    continue

                new_prob = prob * card_prob
                value = card_value(rank)

                if rank == 1:  # Ace
                    # Try to use ace as 11
                    if current_sum + 11 <= 21:
                        dealer_play(current_sum + 11, True, new_prob)
                    else:
                        dealer_play(current_sum + 1, usable_ace, new_prob)
                else:
                    new_sum = current_sum + value
                    if new_sum > 21 and usable_ace:
                        # Convert usable ace from 11 to 1
                        new_sum -= 10
                        dealer_play(new_sum, False, new_prob)
                    else:
                        dealer_play(new_sum, usable_ace, new_prob)

        # Initialize dealer's hand with showing card
        if dealer_showing == 1:  # Ace
            # Dealer starts with ace as 11 (usable)
            dealer_play(11, True, 1.0)
        else:
            dealer_play(dealer_showing, False, 1.0)

        self._dealer_probs_cache[dealer_showing] = final_probs
        return final_probs

    def expected_stand_value(self, player_sum: int, dealer_showing: int) -> float:
        """
        Compute expected value of standing with given hand vs dealer up-card.

        Args:
            player_sum: Player's current hand value (not busted)
            dealer_showing: Dealer's up-card value (1-10)

        Returns:
            Expected reward from standing
        """
        dealer_probs = self.compute_dealer_probs(dealer_showing)
        expected_value = 0.0

        for dealer_final, prob in dealer_probs.items():
            if dealer_final == 22:  # Dealer bust
                expected_value += prob * 1.0  # Player wins
            elif player_sum > dealer_final:
                expected_value += prob * 1.0  # Player wins
            elif player_sum < dealer_final:
                expected_value += prob * (-1.0)  # Player loses
            # else: push, value = 0

        return expected_value

    def expected_hit_value(self, player_sum: int, dealer_showing: int,
                           usable_ace: bool) -> float:
        """
        Compute expected value of hitting given current state.

        Args:
            player_sum: Player's current hand value
            dealer_showing: Dealer's up-card value (1-10)
            usable_ace: Whether player has a usable ace

        Returns:
            Expected value from hitting (using current value function)
        """
        expected_value = 0.0

        for rank in range(1, 14):
            card_prob = self.card_probs[rank]
            if card_prob == 0:
                continue

            value = card_value(rank)

            if rank == 1:  # Ace
                # Try to use ace as 11
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
                    # Bust - immediate loss
                    expected_value += card_prob * (-1.0)
                    continue

            # Get value from value function (capped at 21)
            if new_sum <= 21:
                expected_value += card_prob * self.V[new_sum, dealer_showing, int(new_usable)]

        return expected_value

    def value_iteration(self, epsilon: float = 1e-9, max_iterations: int = 1000) -> int:
        """
        Run value iteration to compute optimal policy.

        Args:
            epsilon: Convergence threshold
            max_iterations: Maximum number of iterations

        Returns:
            Number of iterations until convergence
        """
        # Clear dealer cache in case card probs changed
        self._dealer_probs_cache = {}

        for iteration in range(max_iterations):
            delta = 0.0

            # Iterate over all states
            for player_sum in range(4, 22):  # 4-21 (min 2-card hand is 4)
                for dealer_showing in range(1, 11):  # 1-10
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
                return iteration + 1

        return max_iterations

    def get_policy(self, player_sum: int, dealer_showing: int, usable_ace: bool) -> int:
        """Get optimal action for a state."""
        return self.policy[player_sum, dealer_showing, int(usable_ace)]

    def get_value(self, player_sum: int, dealer_showing: int, usable_ace: bool) -> float:
        """Get value of a state under optimal policy."""
        return self.V[player_sum, dealer_showing, int(usable_ace)]

    def compute_expected_return(self) -> float:
        """
        Compute the expected return for the player under optimal play.

        This accounts for:
        - Initial deal probabilities
        - Blackjack payouts
        - Optimal play from non-blackjack hands

        Returns:
            Expected return per hand (negative = house edge)
        """
        expected_return = 0.0

        # Iterate over all possible initial deals
        for player_card1 in range(1, 14):
            p1_prob = self.card_probs[player_card1]
            if p1_prob == 0:
                continue

            for player_card2 in range(1, 14):
                p2_prob = self.card_probs[player_card2]
                if p2_prob == 0:
                    continue

                for dealer_up in range(1, 14):
                    d_prob = self.card_probs[dealer_up]
                    if d_prob == 0:
                        continue

                    deal_prob = p1_prob * p2_prob * d_prob

                    # Evaluate player's initial hand
                    player_sum, usable_ace = evaluate_hand([player_card1, player_card2])
                    player_bj = (player_sum == 21)

                    # Dealer's showing value (1-10)
                    dealer_showing = card_value(dealer_up)
                    if dealer_up == 1:
                        dealer_showing = 1

                    if player_bj:
                        # Check for dealer blackjack probability
                        # Dealer needs a 10-value card to complete blackjack
                        if dealer_up == 1:
                            # Dealer shows ace, needs 10/J/Q/K
                            dealer_bj_prob = sum(self.card_probs[r] for r in [10, 11, 12, 13])
                        elif dealer_showing == 10:
                            # Dealer shows 10-value, needs ace
                            dealer_bj_prob = self.card_probs[1]
                        else:
                            dealer_bj_prob = 0.0

                        # Player blackjack outcomes
                        expected_return += deal_prob * (
                            dealer_bj_prob * 0.0 +  # Push
                            (1 - dealer_bj_prob) * 1.5  # Win 3:2
                        )
                    else:
                        # Regular hand - use value function
                        state_value = self.V[player_sum, dealer_showing, int(usable_ace)]
                        expected_return += deal_prob * state_value

        return expected_return

    def print_policy(self, usable_ace: bool = False):
        """Print the optimal policy as a grid."""
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
            # Dealer ace (shown as 1)
            action = "H" if self.policy[p, 1, int(usable_ace)] == HIT else "S"
            print(f"  {action}")


def main():
    """Test the MDP solver."""
    print("Solving standard Blackjack MDP...")
    mdp = BlackjackMDP()
    iterations = mdp.value_iteration()
    print(f"Converged in {iterations} iterations")

    expected_return = mdp.compute_expected_return()
    print(f"Expected return under optimal play: {expected_return:.6f}")
    print(f"House edge: {-expected_return * 100:.4f}%")

    mdp.print_policy(usable_ace=False)
    mdp.print_policy(usable_ace=True)


if __name__ == "__main__":
    main()
