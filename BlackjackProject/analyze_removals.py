#!/usr/bin/env python3
"""
Analyze which card rank removal maximizes the casino's edge in Blackjack.

For each of the 13 ranks, this script:
1. Creates a modified deck with that rank removed
2. Computes the optimal player policy via value iteration
3. Calculates the expected player return under optimal play
4. Identifies which removal minimizes player return (best for casino)

Usage:
    python analyze_removals.py
"""

from blackjack_mdp import BlackjackMDP
from blackjack_game import RANK_NAMES


def analyze_all_removals():
    """Analyze the effect of removing each rank on player expected return."""

    print("=" * 60)
    print("Blackjack MDP Analysis: Which Rank Should the Casino Remove?")
    print("=" * 60)

    # First, compute baseline (no removal)
    print("\nComputing baseline (standard 13-rank deck)...")
    baseline_mdp = BlackjackMDP(removed_rank=None)
    baseline_mdp.value_iteration()
    baseline_return = baseline_mdp.compute_expected_return()
    print(f"Baseline expected return: {baseline_return:.6f}")
    print(f"Baseline house edge: {-baseline_return * 100:.4f}%")

    print("\n" + "-" * 60)
    print("Analyzing each possible rank removal...")
    print("-" * 60)

    results = []

    for rank in range(1, 14):
        rank_name = RANK_NAMES[rank]
        print(f"\nRemoving {rank_name}...", end=" ")

        mdp = BlackjackMDP(removed_rank=rank)
        iterations = mdp.value_iteration()
        expected_return = mdp.compute_expected_return()
        house_edge = -expected_return * 100

        print(f"Expected return: {expected_return:.6f}, House edge: {house_edge:.4f}%")

        results.append({
            'rank': rank,
            'name': rank_name,
            'expected_return': expected_return,
            'house_edge': house_edge,
            'iterations': iterations
        })

    # Sort by expected return (ascending = best for casino first)
    results.sort(key=lambda x: x['expected_return'])

    print("\n" + "=" * 60)
    print("RESULTS: Rank Removals Sorted by House Edge (Best for Casino First)")
    print("=" * 60)
    print(f"{'Rank':<6} {'Expected Return':>16} {'House Edge':>12} {'Change from Baseline':>20}")
    print("-" * 60)

    for r in results:
        change = r['expected_return'] - baseline_return
        change_str = f"{change:+.6f}"
        print(f"{r['name']:<6} {r['expected_return']:>16.6f} {r['house_edge']:>11.4f}% {change_str:>20}")

    print("-" * 60)

    best = results[0]
    worst = results[-1]

    print(f"\nBEST for CASINO: Remove {best['name']}")
    print(f"  Expected player return: {best['expected_return']:.6f}")
    print(f"  House edge: {best['house_edge']:.4f}%")
    print(f"  Change from baseline: {best['expected_return'] - baseline_return:+.6f}")

    print(f"\nWORST for CASINO: Remove {worst['name']}")
    print(f"  Expected player return: {worst['expected_return']:.6f}")
    print(f"  House edge: {worst['house_edge']:.4f}%")
    print(f"  Change from baseline: {worst['expected_return'] - baseline_return:+.6f}")

    return results, baseline_return


def print_optimal_policies(rank: int):
    """Print the optimal policy for a specific rank removal."""
    rank_name = RANK_NAMES[rank]
    print(f"\n{'=' * 60}")
    print(f"Optimal Policy with {rank_name} Removed")
    print("=" * 60)

    mdp = BlackjackMDP(removed_rank=rank)
    mdp.value_iteration()
    mdp.print_policy(usable_ace=False)
    mdp.print_policy(usable_ace=True)


if __name__ == "__main__":
    results, baseline = analyze_all_removals()

    # Also show optimal policy for the best removal
    best_rank = results[0]['rank']
    print_optimal_policies(best_rank)
