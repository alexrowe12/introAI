#!/usr/bin/env python3
"""
Finite deck tracker for card counting in Blackjack.

Tracks exact card counts and computes dynamic draw probabilities
for a single deck with configurable rank removal.
"""

from typing import Dict, List, Tuple, Optional
from copy import deepcopy


class FiniteDeckTracker:
    """
    Tracks the exact composition of a finite deck.

    For the tournament: 48 cards (standard deck minus all 5s)
    Tracks by card VALUE (1-10), not rank, since J/Q/K are equivalent to 10.
    """

    def __init__(self, removed_rank: int = 5):
        """
        Initialize deck tracker.

        Args:
            removed_rank: Which rank is completely removed (5 for tournament)
        """
        self.removed_rank = removed_rank
        self.remaining_counts: Dict[int, int] = {}
        self.initial_counts: Dict[int, int] = {}
        self._build_initial_deck()
        self.reset()

    def _build_initial_deck(self) -> None:
        """Build the initial deck configuration."""
        # Standard deck: 4 of each rank 1-9, 16 ten-value cards (10, J, Q, K)
        for value in range(1, 10):
            if value == self.removed_rank:
                self.initial_counts[value] = 0
            else:
                self.initial_counts[value] = 4

        # Ten-value cards (10, J, Q, K) - 16 total, but remove 4 if 10 is removed
        if self.removed_rank == 10:
            self.initial_counts[10] = 12  # Remove the four 10s, keep J, Q, K
        else:
            self.initial_counts[10] = 16

    def reset(self) -> None:
        """Reset to full deck state."""
        self.remaining_counts = dict(self.initial_counts)

    def remove_card(self, rank_value: int) -> bool:
        """
        Remove a card from the deck.

        Args:
            rank_value: Value 1-10 (A=1, face cards=10)

        Returns:
            True if card was available and removed, False otherwise
        """
        if rank_value < 1 or rank_value > 10:
            return False
        if self.remaining_counts.get(rank_value, 0) <= 0:
            return False

        self.remaining_counts[rank_value] -= 1
        return True

    def remove_cards(self, rank_values: List[int]) -> int:
        """
        Remove multiple cards at once.

        Args:
            rank_values: List of card values to remove

        Returns:
            Number of cards successfully removed
        """
        removed = 0
        for value in rank_values:
            if self.remove_card(value):
                removed += 1
        return removed

    def add_card(self, rank_value: int) -> bool:
        """
        Add a card back to the deck (for undo operations).

        Args:
            rank_value: Value 1-10

        Returns:
            True if card was added (doesn't exceed initial count)
        """
        if rank_value < 1 or rank_value > 10:
            return False

        initial = self.initial_counts.get(rank_value, 0)
        current = self.remaining_counts.get(rank_value, 0)

        if current >= initial:
            return False

        self.remaining_counts[rank_value] = current + 1
        return True

    def get_remaining_count(self, rank_value: int) -> int:
        """Get count of cards remaining with given value."""
        return self.remaining_counts.get(rank_value, 0)

    def get_total_remaining(self) -> int:
        """Get total cards remaining in deck."""
        return sum(self.remaining_counts.values())

    def get_draw_probabilities(self) -> Dict[int, float]:
        """
        Compute probability distribution for next card draw.

        Returns:
            Dict mapping rank_value (1-10) to probability
        """
        total = self.get_total_remaining()
        if total == 0:
            return {v: 0.0 for v in range(1, 11)}

        return {v: self.remaining_counts.get(v, 0) / total for v in range(1, 11)}

    def get_probability(self, rank_value: int) -> float:
        """Get probability of drawing a specific card value."""
        total = self.get_total_remaining()
        if total == 0:
            return 0.0
        return self.remaining_counts.get(rank_value, 0) / total

    def get_state_tuple(self) -> Tuple[int, ...]:
        """
        Get hashable representation of current deck state.

        Returns:
            Tuple of (count_1, count_2, ..., count_10)
        """
        return tuple(self.remaining_counts.get(v, 0) for v in range(1, 11))

    def copy(self) -> 'FiniteDeckTracker':
        """Create a deep copy of the tracker (for simulations)."""
        new_tracker = FiniteDeckTracker.__new__(FiniteDeckTracker)
        new_tracker.removed_rank = self.removed_rank
        new_tracker.initial_counts = dict(self.initial_counts)
        new_tracker.remaining_counts = dict(self.remaining_counts)
        return new_tracker

    def is_empty(self) -> bool:
        """Check if deck is empty."""
        return self.get_total_remaining() == 0

    def get_cards_dealt(self) -> int:
        """Get number of cards that have been dealt."""
        initial_total = sum(self.initial_counts.values())
        return initial_total - self.get_total_remaining()

    def __str__(self) -> str:
        """Human-readable deck state."""
        lines = [f"Deck: {self.get_total_remaining()} cards remaining"]
        for v in range(1, 11):
            count = self.remaining_counts.get(v, 0)
            initial = self.initial_counts.get(v, 0)
            name = 'A' if v == 1 else str(v)
            if v == 10:
                name = '10/J/Q/K'
            lines.append(f"  {name}: {count}/{initial}")
        return '\n'.join(lines)

    def get_summary(self) -> str:
        """Compact summary of deck state."""
        parts = []
        for v in range(1, 11):
            if v == self.removed_rank:
                continue
            count = self.remaining_counts.get(v, 0)
            name = 'A' if v == 1 else ('T' if v == 10 else str(v))
            parts.append(f"{name}:{count}")
        return ' '.join(parts)


def parse_card_input(s: str) -> int:
    """
    Parse card input string to rank value.

    Accepts: "A", "2"-"10", "J", "Q", "K", "T", or numeric 1-10
    Returns: 1-10 (A=1, face cards=10)

    Raises:
        ValueError: If input is invalid
    """
    s = s.strip().upper()

    card_map = {
        'A': 1, '1': 1,
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        '10': 10, 'T': 10, 'J': 10, 'Q': 10, 'K': 10
    }

    if s in card_map:
        return card_map[s]

    raise ValueError(f"Invalid card: '{s}'. Use A, 2-10, J, Q, K, or T for 10.")


def parse_multiple_cards(s: str) -> List[int]:
    """
    Parse multiple card inputs from a string.

    Accepts space or comma separated cards.
    Returns list of card values.
    """
    # Replace commas with spaces and split
    parts = s.replace(',', ' ').split()
    return [parse_card_input(p) for p in parts]
