#!/usr/bin/env python3
"""
Blackjack game logic for infinite-deck MDP analysis.

Provides core functionality for hand evaluation, card values, and dealer simulation.
Uses simplified rules: hit/stand only, dealer stands on all 17s, blackjack pays 3:2.
"""

from typing import Tuple, List

# Card rank values: A=1 (or 11), 2-10 face value, J/Q/K=10
# We represent ranks as 1-13 where 1=A, 11=J, 12=Q, 13=K
RANK_VALUES = {
    1: (1, 11),   # Ace can be 1 or 11
    2: (2,),
    3: (3,),
    4: (4,),
    5: (5,),
    6: (6,),
    7: (7,),
    8: (8,),
    9: (9,),
    10: (10,),
    11: (10,),    # Jack
    12: (10,),    # Queen
    13: (10,),    # King
}

RANK_NAMES = {
    1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'
}


def card_value(rank: int) -> int:
    """
    Get the primary numeric value of a card rank.
    Ace returns 1 (soft value handled separately).
    Face cards (J, Q, K) return 10.
    """
    if rank == 1:
        return 1
    elif rank >= 10:
        return 10
    else:
        return rank


def evaluate_hand(cards: List[int]) -> Tuple[int, bool]:
    """
    Evaluate a hand's value and whether it has a usable ace.

    Args:
        cards: List of card ranks (1-13)

    Returns:
        (hand_value, usable_ace): The best hand value and whether an ace counts as 11
    """
    total = 0
    aces = 0

    for rank in cards:
        if rank == 1:
            aces += 1
            total += 1
        elif rank >= 10:
            total += 10
        else:
            total += rank

    # Try to use one ace as 11 if it doesn't bust
    usable_ace = False
    if aces > 0 and total + 10 <= 21:
        total += 10
        usable_ace = True

    return total, usable_ace


def is_bust(hand_value: int) -> bool:
    """Check if a hand value is a bust (over 21)."""
    return hand_value > 21


def is_blackjack(cards: List[int]) -> bool:
    """
    Check if a hand is a natural blackjack (21 with exactly 2 cards).
    """
    if len(cards) != 2:
        return False
    value, _ = evaluate_hand(cards)
    return value == 21


def dealer_showing_value(rank: int) -> int:
    """
    Get the value of the dealer's showing card for state representation.
    Ace = 1, Face cards = 10, others = face value.
    Returns values 1-10.
    """
    if rank == 1:
        return 1
    elif rank >= 10:
        return 10
    else:
        return rank


def compare_hands(player_value: int, dealer_value: int,
                  player_blackjack: bool = False,
                  dealer_blackjack: bool = False) -> float:
    """
    Compare player and dealer hands to determine outcome.

    Args:
        player_value: Player's final hand value
        dealer_value: Dealer's final hand value
        player_blackjack: Whether player has natural blackjack
        dealer_blackjack: Whether dealer has natural blackjack

    Returns:
        Reward: +1.5 for player blackjack win, +1 for regular win,
                -1 for loss, 0 for push
    """
    # Handle blackjacks
    if player_blackjack and dealer_blackjack:
        return 0.0  # Push
    if player_blackjack:
        return 1.5  # Blackjack pays 3:2
    if dealer_blackjack:
        return -1.0

    # Handle busts
    if player_value > 21:
        return -1.0
    if dealer_value > 21:
        return 1.0

    # Compare values
    if player_value > dealer_value:
        return 1.0
    elif player_value < dealer_value:
        return -1.0
    else:
        return 0.0  # Push
