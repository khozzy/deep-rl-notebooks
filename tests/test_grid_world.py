import numpy as np

from src.environments.grid_world import GridWorldEnv


class TestGridWorldEnv:
    """Test suite for the GridWorldEnv class."""

    def test_initialization(self):
        """Test that the environment initializes correctly with default parameters."""
        env = GridWorldEnv()

        # Check dimensions
        assert env.width == 4
        assert env.height == 3
        assert env.num_states == 12

        # Check spaces
        assert env.observation_space.n == 12
        assert env.action_space.n == 4

        # Check initial position
        state, _ = env.reset()
        assert state == 0  # Should be at (1,1) which maps to state 0
        assert np.array_equal(env.agent_position, np.array([1, 1]))

    def test_custom_initialization(self):
        """Test that the environment initializes correctly with custom parameters."""
        custom_env = GridWorldEnv(
            width=5,
            height=4,
            start_position=(2, 2),
            obstacles=[(3, 3)],
            terminal_states={(5, 4): 10, (4, 2): -5},
        )

        # Check dimensions
        assert custom_env.width == 5
        assert custom_env.height == 4
        assert custom_env.num_states == 20

        # Check spaces
        assert custom_env.observation_space.n == 20
        assert custom_env.action_space.n == 4

        # Check initial position
        state, _ = custom_env.reset()
        assert np.array_equal(custom_env.agent_position, np.array([2, 2]))

        # Check terminal states
        terminal_state_1 = custom_env._coords_to_state(5, 4)
        terminal_state_2 = custom_env._coords_to_state(4, 2)
        assert custom_env.terminal_states[terminal_state_1] == 10
        assert custom_env.terminal_states[terminal_state_2] == -5

    def test_state_coord_conversion(self):
        """Test conversion between state numbers and coordinates."""
        env = GridWorldEnv()

        # Test coordinates to state
        assert env._coords_to_state(1, 1) == 0
        assert env._coords_to_state(4, 3) == 11
        assert env._coords_to_state(2, 2) == 5

        # Test state to coordinates
        assert env._state_to_coords(0) == (1, 1)
        assert env._state_to_coords(11) == (4, 3)
        assert env._state_to_coords(5) == (2, 2)

        # Test round-trip conversions
        for x in range(1, env.width + 1):
            for y in range(1, env.height + 1):
                state = env._coords_to_state(x, y)
                x2, y2 = env._state_to_coords(state)
                assert (x, y) == (x2, y2)

    def test_step_valid_moves(self):
        """Test that valid moves update the agent position correctly."""
        env = GridWorldEnv()
        env.reset()

        # Move right
        next_state, reward, done, _, _ = env.step(0)  # Right
        assert np.array_equal(env.agent_position, np.array([2, 1]))
        assert next_state == 1
        assert reward == 0
        assert not done

        # Move up
        next_state, reward, done, _, _ = env.step(1)  # Up
        assert np.array_equal(env.agent_position, np.array([2, 1]))  # Blocked by wall
        assert next_state == 1
        assert reward == 0
        assert not done

        # Move right again
        next_state, reward, done, _, _ = env.step(0)  # Right
        assert np.array_equal(env.agent_position, np.array([3, 1]))
        assert next_state == 2
        assert reward == 0
        assert not done

    def test_step_obstacle_collision(self):
        """Test that the agent cannot move into obstacles."""
        env = GridWorldEnv()
        env.reset()

        # Move right once to position (2,1)
        env.step(0)

        # Try to move up into the wall at (2,2)
        next_state, reward, done, _, _ = env.step(1)  # Up

        # Position should remain unchanged
        assert np.array_equal(env.agent_position, np.array([2, 1]))
        assert next_state == 1
        assert reward == 0
        assert not done

    def test_step_boundary_collision(self):
        """Test that the agent cannot move outside the grid boundaries."""
        env = GridWorldEnv()
        env.reset()

        # Try to move left (would go out of bounds)
        next_state, reward, done, _, _ = env.step(2)  # Left

        # Position should remain unchanged
        assert np.array_equal(env.agent_position, np.array([1, 1]))
        assert next_state == 0
        assert reward == 0
        assert not done

        # Try to move down (would go out of bounds)
        next_state, reward, done, _, _ = env.step(3)  # Down

        # Position should remain unchanged
        assert np.array_equal(env.agent_position, np.array([1, 1]))
        assert next_state == 0
        assert reward == 0
        assert not done

    def test_positive_reward_terminal(self):
        """Test reaching the positive reward terminal state."""
        env = GridWorldEnv()
        env.reset()

        # Path to positive reward: Right, Right, Up, Up, Right
        env.step(0)  # Right to (2,1)
        env.step(0)  # Right to (3,1)
        env.step(1)  # Up to (3,2)
        env.step(1)  # Up to (3,3)
        next_state, reward, done, _, _ = env.step(0)  # Right to (4,3)

        # Should be at terminal state with +1 reward
        assert np.array_equal(env.agent_position, np.array([4, 3]))
        assert next_state == 11
        assert reward == 1
        assert done

    def test_negative_reward_terminal(self):
        """Test reaching the negative reward terminal state."""
        env = GridWorldEnv()
        env.reset()

        # Path to negative reward: Right, Right, Right, Up
        env.step(0)  # Right to (2,1)
        env.step(0)  # Right to (3,1)
        env.step(0)  # Right to (4,1)
        next_state, reward, done, _, _ = env.step(1)  # Up to (4,2)

        # Should be at terminal state with -1 reward
        assert np.array_equal(env.agent_position, np.array([4, 2]))
        assert next_state == 7
        assert reward == -1
        assert done

    def test_reset(self):
        """Test that reset returns the environment to its initial state."""
        env = GridWorldEnv()

        # Move the agent
        env.step(0)  # Right
        env.step(0)  # Right

        # Reset
        state, _ = env.reset()

        # Agent should be back at the start
        assert np.array_equal(env.agent_position, np.array([1, 1]))
        assert state == 0

    def test_transition_matrix(self):
        """Test that the transition matrix is built correctly."""
        env = GridWorldEnv()

        # Check a few key transitions

        # From start state (0), moving right
        prob, next_state, reward, done = env.P[0][0][0]
        assert prob == 1.0
        assert next_state == 1  # (2,1)
        assert reward == 0
        assert not done

        # From state 2 (3,1), moving up
        prob, next_state, reward, done = env.P[2][1][0]
        assert prob == 1.0
        assert next_state == 6  # (3,2)
        assert reward == 0
        assert not done

        # From state 10 (3,3), moving right (to terminal)
        prob, next_state, reward, done = env.P[10][0][0]
        assert prob == 1.0
        assert next_state == 11  # (4,3)
        assert reward == 1
        assert done

        # From terminal state, any action
        for action in range(4):
            prob, next_state, reward, done = env.P[11][action][0]
            assert prob == 1.0
            assert next_state == 11  # stays in place
            assert reward == 0  # no additional reward for actions in terminal
            assert done

    def test_expected_values_under_random_policy(self):
        """Test expected values of states under a random policy."""
        env = GridWorldEnv()

        # Define a random policy (25% probability for each action)
        random_policy = (
            np.ones((env.num_states, env.action_space.n)) / env.action_space.n
        )

        # Compute value function for random policy using Bellman equation
        # V(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        gamma = 0.9  # Discount factor
        theta = 0.001  # Convergence threshold

        # Initialize value function
        V = np.zeros(env.num_states)

        # Policy evaluation
        while True:
            delta = 0
            for s in range(env.num_states):
                v = V[s]
                new_v = 0

                # For each action
                for a in range(env.action_space.n):
                    # For each possible next state and reward
                    for prob, next_s, r, _ in [
                        env.P[s][a][0]
                    ]:  # Only one outcome per action in our env
                        new_v += random_policy[s, a] * prob * (r + gamma * V[next_s])

                V[s] = new_v
                delta = max(delta, abs(v - V[s]))

            if delta < theta:
                break

        # Check that terminal states have expected values
        terminal_states = [
            env._coords_to_state(4, 2),  # negative terminal
            env._coords_to_state(4, 3),  # positive terminal
        ]

        # Terminal states should have values consistent with their rewards
        assert (
            abs(V[terminal_states[0]]) < 1e-5
        )  # Terminal states have 0 value under proper policy evaluation
        assert abs(V[terminal_states[1]]) < 1e-5

        # States close to positive terminal should have higher values than states close to negative terminal
        assert V[env._coords_to_state(3, 3)] > V[env._coords_to_state(3, 2)]

    def test_random_agent_performance(self):
        """Test performance of a random agent in the environment."""
        env = GridWorldEnv()
        np.random.seed(42)  # For reproducible results

        # Number of episodes to run
        n_episodes = 100

        # Track statistics
        episode_lengths = []
        total_rewards = []
        terminal_visits = {
            env._coords_to_state(4, 2): 0,  # Count visits to negative terminal
            env._coords_to_state(4, 3): 0,  # Count visits to positive terminal
        }

        # Run random agent
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            episode_reward = 0

            while not done and steps < 1000:  # Limit to prevent infinite loops
                action = env.action_space.sample()  # Random action
                next_state, reward, done, _, _ = env.step(action)

                episode_reward += reward
                steps += 1
                state = next_state

                if done:
                    terminal_visits[state] += 1

            episode_lengths.append(steps)
            total_rewards.append(episode_reward)

        # Calculate statistics
        mean_length = np.mean(episode_lengths)
        mean_reward = np.mean(total_rewards)

        # Verify that random agent eventually reaches terminal states
        assert (
            terminal_visits[env._coords_to_state(4, 2)]
            + terminal_visits[env._coords_to_state(4, 3)]
            == n_episodes
        )

        # Check that statistics are reasonable
        assert (
            0 < mean_length < 1000
        )  # Random agent should take some steps but not reach limit
        assert -1 <= mean_reward <= 1  # Mean reward should be between -1 and 1
