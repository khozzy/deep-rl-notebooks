import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    Configurable GridWorld Environment that follows gym interface.
    By default, creates the 4x3 grid world with:
    - Robot starting at position (1,1)
    - Wall/obstacle at (2,2)
    - +1 reward at (4,3) (blue diamond)
    - -1 reward at (4,2) (explosion)
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        width=4,
        height=3,
        start_position=(1, 1),
        obstacles=[(2, 2)],
        terminal_states={(4, 2): -1, (4, 3): 1},
        render_mode=None,
    ):
        super().__init__()

        # Grid dimensions
        self.width = width
        self.height = height
        self.num_states = self.width * self.height

        # Define the action space (0: right, 1: up, 2: left, 3: down)
        self.action_space = spaces.Discrete(4)

        # Define the observation space
        self.observation_space = spaces.Discrete(self.num_states)

        # Action to direction mapping
        self.directions = [
            np.array([1, 0]),  # right
            np.array([0, 1]),  # up
            np.array([-1, 0]),  # left
            np.array([0, -1]),  # down
        ]

        # Convert the start position to numpy array
        self.start_position = np.array(start_position)

        # Define grid cells with obstacles
        self.obstacles = obstacles

        # Define terminal states with rewards
        self.terminal_states = {
            self._coords_to_state(x, y): reward
            for (x, y), reward in terminal_states.items()
        }

        # Initialize transition matrix
        self.P = self._build_transition_matrix()

        self.agent_position = None
        self.render_mode = render_mode

        # Initialize the state
        self.reset()

    def _state_to_coords(self, state):
        """Convert state number to (x,y) coordinates"""
        x = (state % self.width) + 1
        y = (state // self.width) + 1
        return (x, y)

    def _coords_to_state(self, x, y):
        """Convert (x,y) coordinates to state number"""
        x_idx = x - 1
        y_idx = y - 1
        # Ensure coordinates are within grid bounds
        x_idx = max(0, min(x_idx, self.width - 1))
        y_idx = max(0, min(y_idx, self.height - 1))
        return x_idx + y_idx * self.width

    def _build_transition_matrix(self):
        """Build the transition matrix P[s][a] = [(prob, next_state, reward, done)]"""
        P = {s: {a: [] for a in range(4)} for s in range(self.num_states)}

        for s in range(self.num_states):
            # Get coordinates of state s
            x, y = self._state_to_coords(s)

            # Skip if this is an obstacle
            if (x, y) in self.obstacles:
                for a in range(4):
                    P[s][a] = [(1.0, s, 0, True)]
                continue

            # Check if state is terminal
            if s in self.terminal_states:
                for a in range(4):
                    P[s][a] = [(1.0, s, 0, True)]
                continue

            # For each action
            for a in range(4):
                direction = self.directions[a]
                new_x, new_y = x + direction[0], y + direction[1]

                # Check if the new position is valid
                if (
                    1 <= new_x <= self.width
                    and 1 <= new_y <= self.height
                    and (new_x, new_y) not in self.obstacles
                ):
                    next_s = self._coords_to_state(new_x, new_y)
                    reward = self.terminal_states.get(next_s, 0)
                    done = next_s in self.terminal_states

                    P[s][a] = [(1.0, next_s, reward, done)]
                else:
                    # If invalid, stay in the current state
                    P[s][a] = [(1.0, s, 0, False)]

        return P

    def step(self, action):
        state = self._coords_to_state(self.agent_position[0], self.agent_position[1])
        prob, next_state, reward, done = self.P[state][action][0]

        if next_state != state:  # Only update if we actually moved
            self.agent_position = np.array(self._state_to_coords(next_state))

        truncated = False
        info = {}

        return next_state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_position = np.copy(self.start_position)
        state = self._coords_to_state(self.agent_position[0], self.agent_position[1])
        return state, {}

    def render(self):
        if self.render_mode != "human":
            return

        # Get cell representations
        cells = {}

        # Initialize all cells as empty
        for y in range(1, self.height + 1):
            for x in range(1, self.width + 1):
                cells[(x, y)] = "⬜"  # Empty space

        # Mark special cells

        # Terminal states
        for state, reward in self.terminal_states.items():
            x, y = self._state_to_coords(state)
            cells[(x, y)] = "💎" if reward > 0 else "💀"

        # Obstacles
        for x, y in self.obstacles:
            cells[(x, y)] = "🪨"  # Wall

        # Start position
        if not np.array_equal(self.agent_position, self.start_position):
            sx, sy = self.start_position
            cells[(sx, sy)] = "🟢"  # Start position

        # Agent position
        ax, ay = self.agent_position
        cells[(ax, ay)] = "🤖"  # Robot

        # Print the grid
        print("\nGridWorld ({}x{}):".format(self.width, self.height))

        # Grid cells
        for y in range(self.height, 0, -1):
            row_str = f" {y}  "
            for x in range(1, self.width + 1):
                row_str += f" {cells[(x, y)]} "
            print(row_str)

        # Column indices - align directly under each column
        col_indices = "    "
        for i in range(1, self.width + 1):
            col_indices += f" {i}  "
        print(col_indices)

        print(
            "\nLegend: 🟢=Start, 🤖=Robot, 🪨=Wall, 💎=Reward(+1), 💀=Reward(-1), ⬜=Empty"
        )
        print("Actions: 0=right, 1=up, 2=left, 3=down\n")
