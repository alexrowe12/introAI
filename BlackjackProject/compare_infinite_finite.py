#!/usr/bin/env python3
"""
Bonus Task: Compare infinite-deck vs finite-deck expected returns.

Compares the expected return of the optimal infinite-deck policy when applied to:
1. Infinite-deck setting (analytical calculation)
2. Finite-deck setting (Monte Carlo simulation)

This demonstrates the difference between the theoretical infinite-deck model
and realistic finite-deck play.
"""

from blackjack_mdp import BlackjackMDP
from monte_carlo_simulator import MonteCarloSimulator
from finite_deck_tracker import FiniteDeckTracker


def create_infinite_deck_strategy(mdp: BlackjackMDP):
    """
    Create a strategy function that uses the infinite-deck optimal policy.

    This policy is fixed and does not adapt to the current deck state.
    """
    policy = mdp.policy.copy()

    def strategy(player_sum: int, dealer_up: int, usable_ace: bool,
                 deck: FiniteDeckTracker) -> str:
        if player_sum >= 21:
            return "STAND"
        action = policy[player_sum, dealer_up, int(usable_ace)]
        return "HIT" if action == 0 else "STAND"

    return strategy


def main():
    print("=" * 70)
    print("  BONUS TASK: Infinite-Deck vs Finite-Deck Comparison")
    print("=" * 70)
    print()

    # Step 1: Compute infinite-deck optimal policy
    print("Computing infinite-deck optimal policy (policy iteration)...")
    mdp = BlackjackMDP()
    policy_iters, eval_iters = mdp.policy_iteration()
    print(f"  Converged in {policy_iters} policy iterations")

    # Step 2: Get analytical expected return for infinite-deck
    infinite_deck_return = mdp.compute_expected_return()
    print(f"\nInfinite-deck expected return (analytical): {infinite_deck_return:.6f}")

    # Step 3: Simulate same policy on finite deck
    print("\nSimulating infinite-deck policy on finite deck...")
    print("  (Standard 52-card deck, no rank removed)")

    # Use removed_rank=None for standard 52-card deck
    simulator = MonteCarloSimulator(removed_rank=None, seed=42)
    strategy = create_infinite_deck_strategy(mdp)

    # Run simulation with enough shoes for statistical significance
    n_shoes = 10000
    print(f"  Running {n_shoes} shoes...")

    stats = simulator.simulate_n_shoes(n_shoes, strategy, min_cards=15)

    finite_deck_return = stats['expected_return_per_hand']
    std_error = stats['std_per_shoe'] / (n_shoes ** 0.5) / stats['hands_per_shoe']

    print(f"\nFinite-deck expected return (simulated):   {finite_deck_return:.6f}")
    print(f"  Standard error: ±{std_error:.6f}")
    print(f"  Total hands simulated: {stats['total_hands']:,}")

    # Step 4: Compute and report difference
    difference = finite_deck_return - infinite_deck_return

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\nInfinite-deck expected return: {infinite_deck_return:.6f}")
    print(f"Finite-deck expected return:   {finite_deck_return:.6f}")
    print(f"\nDifference (finite - infinite): {difference:+.6f}")

    if difference > 0:
        comparison = "LARGER (better for player)"
    elif difference < 0:
        comparison = "SMALLER (worse for player)"
    else:
        comparison = "THE SAME"

    print(f"\nThe finite-deck expected return is {comparison} than the infinite-deck return.")

    # Step 5: Explanation
    print("\n" + "=" * 70)
    print("  EXPLANATION")
    print("=" * 70)
    print("""
In the infinite-deck model, each card draw is independent with fixed
probabilities (1/13 per rank). The optimal policy is computed assuming
these probabilities never change.

In the finite-deck setting, cards are drawn without replacement, so
probabilities shift as cards are dealt. However, since we're using the
infinite-deck policy WITHOUT adapting to the changing deck composition,
we're not exploiting any information about which cards remain.

The observed difference is small because:
1. The infinite-deck policy is still close to optimal for a full deck
2. Most hands are played when the deck is relatively full
3. The policy's decisions are based on player/dealer totals, which
   matters more than small probability shifts

Any difference primarily comes from:
- Early in the shoe: probabilities are close to 1/13, minimal difference
- Late in the shoe: probabilities deviate, but the fixed policy doesn't adapt
- The finite deck introduces slight correlations between consecutive hands

A truly optimal finite-deck strategy would adapt to the remaining cards,
but for basic strategy play, the difference is minimal.
""")

    # Summary for submission
    print("=" * 70)
    print("  ANSWER FOR SUBMISSION")
    print("=" * 70)
    print(f"\n1. Numerical difference: {difference:+.6f}")
    print(f"   (Finite-deck return is {'larger' if difference > 0 else 'smaller' if difference < 0 else 'the same'})")
    print()


if __name__ == "__main__":
    main()
