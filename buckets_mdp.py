import numpy as np

# Bucket capacities
CAP_A = 77
CAP_B = 3
CAP_C = 2

# Total number of states: 78 * 4 * 3 = 936
NUM_STATES = (CAP_A + 1) * (CAP_B + 1) * (CAP_C + 1)

def state_to_index(a, b, c):
    """Convert state (a, b, c) to a unique index."""
    return a * (CAP_B + 1) * (CAP_C + 1) + b * (CAP_C + 1) + c

def index_to_state(idx):
    """Convert index back to state (a, b, c)."""
    c = idx % (CAP_C + 1)
    idx //= (CAP_C + 1)
    b = idx % (CAP_B + 1)
    a = idx // (CAP_B + 1)
    return (a, b, c)

def pour(from_amt, to_amt, to_capacity):
    """Pour from one bucket to another.
    Returns (new_from_amt, new_to_amt).
    """
    transfer = min(from_amt, to_capacity - to_amt)
    return (from_amt - transfer, to_amt + transfer)

# Define all 12 actions as functions that take (a, b, c) and return new state
def fill_a(a, b, c): return (CAP_A, b, c)
def fill_b(a, b, c): return (a, CAP_B, c)
def fill_c(a, b, c): return (a, b, CAP_C)

def empty_a(a, b, c): return (0, b, c)
def empty_b(a, b, c): return (a, 0, c)
def empty_c(a, b, c): return (a, b, 0)

def pour_a_to_b(a, b, c):
    new_a, new_b = pour(a, b, CAP_B)
    return (new_a, new_b, c)

def pour_b_to_a(a, b, c):
    new_b, new_a = pour(b, a, CAP_A)
    return (new_a, new_b, c)

def pour_a_to_c(a, b, c):
    new_a, new_c = pour(a, c, CAP_C)
    return (new_a, b, new_c)

def pour_c_to_a(a, b, c):
    new_c, new_a = pour(c, a, CAP_A)
    return (new_a, b, new_c)

def pour_b_to_c(a, b, c):
    new_b, new_c = pour(b, c, CAP_C)
    return (a, new_b, new_c)

def pour_c_to_b(a, b, c):
    new_c, new_b = pour(c, b, CAP_B)
    return (a, new_b, new_c)

# All 12 actions
ACTIONS = [
    fill_a, fill_b, fill_c,
    empty_a, empty_b, empty_c,
    pour_a_to_b, pour_b_to_a,
    pour_a_to_c, pour_c_to_a,
    pour_b_to_c, pour_c_to_b
]

def build_transition_matrix():
    """Build the 936x936 transition matrix."""
    P = np.zeros((NUM_STATES, NUM_STATES))

    for idx in range(NUM_STATES):
        a, b, c = index_to_state(idx)
        for action in ACTIONS:
            new_a, new_b, new_c = action(a, b, c)
            new_idx = state_to_index(new_a, new_b, new_c)
            P[idx, new_idx] += 1.0 / len(ACTIONS)

    return P

def verify_transition_matrix(P):
    """Verify that each row sums to 1."""
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Transition matrix rows don't sum to 1!"
    print(f"Transition matrix verified: {NUM_STATES} states, all rows sum to 1.0")

def find_target_states():
    """Find all states where a + b + c = 5."""
    targets = []
    for idx in range(NUM_STATES):
        a, b, c = index_to_state(idx)
        if a + b + c == 5:
            targets.append(idx)
    return targets

def main():
    print("Buckets MDP Exercise")
    print("=" * 50)
    print(f"Bucket A capacity: {CAP_A}L")
    print(f"Bucket B capacity: {CAP_B}L")
    print(f"Bucket C capacity: {CAP_C}L")
    print(f"Number of actions: {len(ACTIONS)}")
    print(f"Total states: {NUM_STATES}")
    print()

    # Build transition matrix
    print("Building transition matrix...")
    P = build_transition_matrix()
    verify_transition_matrix(P)
    print()

    # Find target states (total water = 5)
    target_states = find_target_states()
    print(f"Target states (a + b + c = 5): {len(target_states)} states")
    for idx in target_states:
        print(f"  {index_to_state(idx)}")
    print()

    # Compute P^100
    print("Computing P^100...")
    P_100 = np.linalg.matrix_power(P, 100)
    print("Done!")
    print()

    # Start state is (0, 0, 0)
    start_idx = state_to_index(0, 0, 0)

    # Sum probabilities of target states
    probability = sum(P_100[start_idx, idx] for idx in target_states)

    print("=" * 50)
    print(f"ANSWER: Probability that total water = 5L after 100 steps:")
    print(f"  {probability}")
    print(f"  (approximately {probability:.10f})")
    print("=" * 50)

if __name__ == "__main__":
    main()
