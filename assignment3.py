import numpy as np
import os

def generate_dummy_mdp(file_path, S, A, gamma, mdptype, rseed):
    np.random.seed(rseed)
    with open(file_path, 'w') as f:
        f.write(f"numStates {S}\n")
        f.write(f"numActions {A}\n")
        f.write("end")
        for _ in range(np.random.randint(1, S // 2)):
            f.write(f" {np.random.randint(0, S)}")
        f.write("\n")
        for s1 in range(S):
            for a in range(A):
                for _ in range(np.random.randint(1, 4)):
                    s2 = np.random.randint(0, S)
                    r = round(np.random.uniform(-1, 1), 3)
                    p = round(np.random.rand(), 3)
                    f.write(f"transition {s1} {a} {s2} {r} {p}\n")
        f.write(f"mdptype {mdptype}\n")
        f.write(f"discount {gamma}\n")

os.makedirs("data", exist_ok=True)

mdp_configs = [
    (10, 5, 0.95, "episodic", 1),
    (20, 10, 0.95, "episodic", 2),
    (100, 10, 0.95, "episodic", 3),
    (1000, 20, 0.95, "episodic", 4),
    (1000, 20, 0.95, "continuing", 5),
    (10000, 20, 0.95, "continuing", 6)
]

mdp_files = []
for S, A, gamma, mdptype, rseed in mdp_configs:
    fname = f"data/{mdptype}-mdp-{S}-{A}.txt"
    generate_dummy_mdp(fname, S, A, gamma, mdptype, rseed)
    mdp_files.append(fname)


def parse_mdp(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    num_states = int(lines[0].split()[1])
    num_actions = int(lines[1].split()[1])
    terminal_states = list(map(int, lines[2].split()[1:]))

    transitions = [[] for _ in range(num_states)]
    mdptype = ""
    gamma = 0.0

    for line in lines[3:]:
        tokens = line.strip().split()
        if tokens[0] == 'transition':
            s1, a, s2 = int(tokens[1]), int(tokens[2]), int(tokens[3])
            r, p = float(tokens[4]), float(tokens[5])
            transitions[s1].append((a, s2, r, p))
        elif tokens[0] == 'mdptype':
            mdptype = tokens[1]
        elif tokens[0] == 'discount':
            gamma = float(tokens[1])

    return num_states, num_actions, terminal_states, transitions, mdptype, gamma


def value_iteration(num_states, num_actions, terminal_states, transitions, gamma, epsilon=1e-10):
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    while True:
        delta = 0
        V_new = np.copy(V)

        for s in range(num_states):
            if s in terminal_states:
                continue

            q_values = np.zeros(num_actions)

            for a in range(num_actions):
                q_sa = 0
                for t in transitions[s]:
                    if t[0] == a:
                        _, s2, r, p = t
                        q_sa += p * (r + gamma * V[s2])
                q_values[a] = q_sa

            best_q = np.max(q_values)
            best_action = np.argmax(q_values)
            V_new[s] = best_q
            policy[s] = best_action
            delta = max(delta, abs(V[s] - best_q))

        V = V_new
        if delta < epsilon:
            break

    return V, policy

for mdp_file in mdp_files:
    print(f"\nSolving MDP from file: {mdp_file}")
    num_states, num_actions, terminal_states, transitions, mdptype, gamma = parse_mdp(
        mdp_file)
    V, policy = value_iteration(
        num_states, num_actions, terminal_states, transitions, gamma)

    for s in range(num_states):
        print(f"{V[s]:.6f}     {policy[s]}")
