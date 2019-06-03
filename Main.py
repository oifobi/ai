import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pygame
import numpy as np
from net import Net, NetModel
import pickle

ELEMENT_NUM: int = 3


class Map:
    MAP: [[int]] = [
        [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 9],
        [0, 1, 1, 1, 1, 0, 8, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 5, 0, 0, 0, 0, 1, 6, 0, 0, 0]
    ]

    ROWS: int = len(MAP)
    COLS: int = len(MAP[0])

    @staticmethod
    def pos_of_table(table) -> (int, int):
        for (row, arr) in enumerate(Map.MAP):
            for (col, val) in enumerate(arr):
                if val == table.index:
                    return col, row

    @staticmethod
    def is_wall(x: int, y: int):
        return x < 0 or y < 0 or x >= Map.COLS or y >= Map.ROWS or Map.MAP[y][x] == 1

    class Waiter:

        IDLE = 0
        FORWARD = 1
        CLOCKWISE = 2
        COUNTERCLOCKWISE = 3

        W = 0
        S = 1
        N = 2
        E = 3

        def __init__(self):
            self.row = 0
            self.col = 0
            self.direction = self.N

        def find_path_to(self, x, y) -> int:

            def cost_function(state, action) -> int:
                return 1

            def score(state, x2, y2) -> int:
                state_x, state_y, state_dir = state
                forward_x, forward_y, _ = Map.Waiter.move(state, Map.Waiter.FORWARD)
                diff_x = float(forward_x - state_x)/float(4)
                diff_y = float(forward_y - state_y) / float(4)
                wall_penalty = 0
                if Map.is_wall(forward_x, forward_y):
                    wall_penalty = 2
                return abs(state_x+diff_x-x2) + abs(state_y+diff_y-y2) + wall_penalty

            def successor(state) -> [int]:
                x, y, dir = Map.Waiter.move(state, Map.Waiter.FORWARD)
                if Map.is_wall(x, y):
                    return [Map.Waiter.CLOCKWISE, Map.Waiter.COUNTERCLOCKWISE]
                else:
                    return [Map.Waiter.CLOCKWISE, Map.Waiter.COUNTERCLOCKWISE, Map.Waiter.FORWARD]

            if self.col == x and self.row == y:
                return Map.Waiter.IDLE

            init_state = (self.col, self.row, self.direction)
            stack = [(init_state, 0)]
            state_space: {(int, int, int): ((int, int, int), int, int)} = {init_state: (init_state, 0)}

            while len(stack) > 0:
                best_state_index = len(stack)-1
                best_score = score(stack[best_state_index][0], x, y)
                for (new_state_index, (new_state, new_cost)) in enumerate(stack):
                    new_score = score(new_state, x, y)
                    if best_score > new_score:
                        best_score = new_score
                        best_state_index = new_state_index

                state, cost = stack.pop(best_state_index)
                actions = successor(state)
                new_states: [(int, int, int)] = [Map.Waiter.move(state, action) for action in actions]
                new_costs: [int] = \
                    [cost + cost_function(new_state, action) for (action, new_state) in zip(actions, new_states)]
                new_scores: [int] = [score(new_state, x, y) for new_state in new_states]
                new_sorted_states_scores_actions_costs: [((int, int, int), int, int, int)] = \
                    list(zip(new_states, new_scores, actions, new_costs))
                new_sorted_states_scores_actions_costs.sort(key=lambda _x: _x[1], reverse=True)

                for (new_state, new_score, new_action, new_cost) in new_sorted_states_scores_actions_costs:
                    new_x, new_y, new_dir = new_state
                    if new_x == x and new_y == y:
                        if state not in state_space:
                            return Map.Waiter.IDLE
                        state_space[new_state] = (state, new_cost, new_action)
                        backtracking_state = new_state
                        state_path = [new_state]
                        action_path = [new_action]
                        while backtracking_state in state_space:
                            prev_backtracking_state, prev_cost, prev_action = state_space[backtracking_state]
                            state_path.append(prev_backtracking_state)
                            action_path.append(prev_action)
                            if prev_backtracking_state == init_state:
                                return prev_action
                            backtracking_state = prev_backtracking_state
                        assert False
                    state_visited = new_state in state_space
                    prev_cost = state_space[new_state][1] if state_visited else 999999
                    if prev_cost > new_cost:
                        state_space[new_state] = (state, new_cost, new_action)
                        if (new_state, new_cost) not in stack:
                            stack.append((new_state, new_cost))
            assert False

        def find_path_to_table(self, table) -> int:
            pos = Map.pos_of_table(table)
            return self.find_path_to(pos[0], pos[1])

        @staticmethod
        def turn_clockwise(dir: int) -> int:
            if dir == Map.Waiter.N:
                return Map.Waiter.E
            elif dir == Map.Waiter.S:
                return Map.Waiter.W
            elif dir == Map.Waiter.E:
                return Map.Waiter.S
            elif dir == Map.Waiter.W:
                return Map.Waiter.N

        @staticmethod
        def turn_counterclockwise(dir: int) -> int:
            if dir == Map.Waiter.N:
                return Map.Waiter.W
            elif dir == Map.Waiter.S:
                return Map.Waiter.E
            elif dir == Map.Waiter.E:
                return Map.Waiter.N
            elif dir == Map.Waiter.W:
                return Map.Waiter.S

        @staticmethod
        def go_forward(x: int, y: int, dir: int) -> (int, int, int):
            if dir == Map.Waiter.N:
                return x, y - 1, dir
            elif dir == Map.Waiter.S:
                return x, y + 1, dir
            elif dir == Map.Waiter.E:
                return x + 1, y, dir
            elif dir == Map.Waiter.W:
                return x - 1, y, dir

        @staticmethod
        def move(state: (int, int, int), action) -> (int, int, int):
            col, row, dir = state
            if action == Map.Waiter.FORWARD:
                return Map.Waiter.go_forward(col, row, dir)
            elif action == Map.Waiter.CLOCKWISE:
                return col, row, Map.Waiter.turn_clockwise(dir)
            elif action == Map.Waiter.COUNTERCLOCKWISE:
                return col, row, Map.Waiter.turn_counterclockwise(dir)
            elif action == Map.Waiter.IDLE:
                return state

        @staticmethod
        def get_action(state_a: (int, int, int), state_b: (int, int, int)) -> int:
            if Map.Waiter.move(state_a, Map.Waiter.FORWARD) == state_b:
                return Map.Waiter.FORWARD
            if Map.Waiter.move(state_a, Map.Waiter.CLOCKWISE) == state_b:
                return Map.Waiter.CLOCKWISE
            if Map.Waiter.move(state_a, Map.Waiter.COUNTERCLOCKWISE) == state_b:
                return Map.Waiter.COUNTERCLOCKWISE
            if Map.Waiter.move(state_a, Map.Waiter.IDLE) == state_b:
                return Map.Waiter.IDLE

        def move_towards_table(self, table):
            action = self.find_path_to_table(table)
            self.col, self.row, self.direction = Map.Waiter.move((self.col, self.row, self.direction), action)

        def current_cell(self) -> int:
            return Map.MAP[self.row][self.col]

        def get_currently_served_table(self):
            return get_table(self.current_cell())

    WAITER = Waiter()

    @staticmethod
    def print():
        print('+' + ("--" * Map.COLS) + '+')
        for (row, arr) in enumerate(Map.MAP):
            print(end='|')
            for (col, cell) in enumerate(arr):
                if row == Map.WAITER.row and col == Map.WAITER.col:
                    if cell == 1:
                        print("Waiter got stuck in wall")
                        exit()
                    print(end='<>')
                else:
                    print(end='  ' if cell == 0 else '##' if cell == 1 else str(cell) * 2)
            print('|')
        print('+' + ("--" * Map.COLS) + '+')

    @staticmethod
    def draw(screen, tables, rect: pygame.Rect):
        cell_h = float(rect.h) / float(Map.ROWS)
        cell_w = float(rect.w) / float(Map.COLS)
        AIR_COLOR = (0, 0, 0)
        TABLE_COLOR = (128, 180, 50)
        WALL_COLOR = (255, 0, 0)
        WAITER_COLOR = (0, 0, 255)
        WAITER_EYE_COLOR = (255, 0, 0)
        CUSTOMER_COLOR = (10, 10, 10)
        for (row, arr) in enumerate(Map.MAP):
            for (col, cell) in enumerate(arr):
                cell_rect = pygame.Rect(rect.x + cell_w * float(col), rect.y + cell_h * float(row), cell_w, cell_h)
                if cell == 0:
                    color = AIR_COLOR
                elif cell == 1:
                    color = WALL_COLOR
                else:
                    color = TABLE_COLOR
                pygame.draw.rect(screen, color, cell_rect)
                if cell > 1:
                    table: Table = tables[cell-2]
                    if table.client is not None:
                        diff = cell_w/float(10)
                        pygame.draw.ellipse(screen, CUSTOMER_COLOR, pygame.Rect(cell_rect.x+diff,
                                                                                cell_rect.y+diff,
                                                                                cell_rect.w-2*diff,
                                                                                cell_rect.h-2*diff))
                if row == Map.WAITER.row and col == Map.WAITER.col:
                    pygame.draw.ellipse(screen, WAITER_COLOR, cell_rect)
                    if Map.WAITER.direction == Map.WAITER.N:
                        dir_dot = (0, -1)
                    elif Map.WAITER.direction == Map.WAITER.S:
                        dir_dot = (0, 1)
                    elif Map.WAITER.direction == Map.WAITER.E:
                        dir_dot = (1, 0)
                    elif Map.WAITER.direction == Map.WAITER.W:
                        dir_dot = (-1, 0)
                    diff = cell_w/float(5)
                    dir_dot = (dir_dot[0]*diff, dir_dot[1]*diff)
                    pygame.draw.ellipse(screen, WAITER_EYE_COLOR, pygame.Rect(cell_rect.x+diff+dir_dot[0],
                                                                              cell_rect.y+diff+dir_dot[1],
                                                                              cell_rect.w-2*diff,
                                                                              cell_rect.h-2*diff))
                    if cell == 1:
                        print("Waiter got stuck in wall")
                        exit()


ELEM_NAME: (str, str, str) = ("Oxygen", "Carbon", "Hydrogen")
MENU = [
    np.array([10, 50, 20]),
    np.array([10, 10, 70]),
    np.array([30, 30, 10]),
    np.array([0, 50, 5]),
    np.array([2, 4, 5])
]

MARKOV_NETWORK = [
    (0, 1, np.array([  # MENU[0] + MENU[1]
        10,  # True + True
        1,  # True + False
        5,  # False + True
        1,  # False + False
    ])),
    (0, 2, np.array([  # MENU[0] + MENU[2]
        4,  # True + True
        1,  # True + False
        5,  # False + True
        3,  # False + False
    ])),
    (0, 3, np.array([  # MENU[0] + MENU[3]
        6,  # True + True
        9,  # True + False
        1,  # False + True
        1,  # False + False
    ])),
    (0, 4, np.array([  # MENU[0] + MENU[4]
        1,  # True + True
        2,  # True + False
        5,  # False + True
        9,  # False + False
    ])),
    (1, 2, np.array([  # MENU[1] + MENU[2]
        1,  # True + True
        1,  # True + False
        8,  # False + True
        9,  # False + False
    ])),
    (1, 3, np.array([  # MENU[1] + MENU[3]
        1,  # True + True
        1,  # True + False
        1,  # False + True
        1,  # False + False
    ])),
    (1, 4, np.array([  # MENU[1] + MENU[4]
        1,  # True + True
        1,  # True + False
        5,  # False + True
        1,  # False + False
    ])),
    (2, 3, np.array([  # MENU[2] + MENU[3]
        1,  # True + True
        1,  # True + False
        5,  # False + True
        1,  # False + False
    ])),
    (2, 4, np.array([  # MENU[2] + MENU[4]
        5,  # True + True
        1,  # True + False
        1,  # False + True
        1,  # False + False
    ])),
    (3, 4, np.array([  # MENU[3] + MENU[4]
        5,  # True + True
        1,  # True + False
        5,  # False + True
        1,  # False + False
    ]))
]


def calculate_probability() -> [([bool], int)]:
    def probability_for(order: [bool], link) -> int:
        lhs = int(order[link[0]])
        rhs = int(order[link[1]])
        return link[2][rhs*2+lhs]

    out = []
    for order in itertools.product(*([[True, False]] * len(MENU))):
        score = 1
        for link in MARKOV_NETWORK:
            score *= probability_for(order, link)
        out += [(order, score)]
    return out


ORDER_AND_COUNT = calculate_probability()
ORDER_COUNT_SUM = sum([x for (_, x) in ORDER_AND_COUNT])
ORDER_AND_PROBABILITY = [(order, count/ORDER_COUNT_SUM) for (order, count) in ORDER_AND_COUNT]


class Client:

    def __init__(self, kind, reaction, tolerance):
        self.kind = kind
        self.reaction = reaction
        self.tolerance = tolerance
        self.ordered_meals = np.zeros(len(MENU))

    def recommend_meal(self) -> int:
        # use self.ordered_meals
        # return most likely next meal
        return 0  # Task for Miguel

    def is_eating_done(self, chemicals) -> bool:
        fuzzy_limit = np.random.randint(int(float(TOLERANCE_THRESHOLD)*0.9), int(float(TOLERANCE_THRESHOLD)*1.1))
        return self.tolerance @ chemicals.T > fuzzy_limit


CLIENTS = [
    Client(0, np.array([1, 2, -1]), np.array([3, 2, 1])),
    Client(1, np.array([0, -2, 3]), np.array([1, 2, 3])),
    Client(2, np.array([-1, 2, 1]), np.array([3, 1, 2]))
]

TOLERANCE_THRESHOLD = 100

CONSUMPTION_END_ESTIMATOR_NET: Net = None


class Table:

    def __init__(self, index: int):
        self.chemicals: np.array = np.array([0, 0, 0])
        self.client: Client = None
        self.index: int = index

    def is_eating_done(self) -> bool:
        return self.client.is_eating_done(self.chemicals)

    def __str__(self):
        return ("free" if self.client is None else ("occupied by " + str(self.client.kind))) + ": " + \
               " + ".join([str(mass) + " " + name for (name, mass) in zip(ELEM_NAME, self.chemicals)])

    def put_random_client(self):
        if self.is_table_free() and self.is_table_clean():
            print("EVENT! Client came to table " + str(self.index))
            self.client = CLIENTS[np.random.randint(0, len(CLIENTS))]

    def remove_client(self):
        if not self.is_table_free():
            print("EVENT! Client left table " + str(self.index))
            self.client = None

    def clean_chemicals(self):
        if self.is_table_free():
            print("EVENT! Cleaning table " + str(self.index))
            self.chemicals.fill(0)

    def is_table_clean(self):
        for val in self.chemicals:
            if val != 0:
                return False
        return True

    def is_table_free(self):
        return self.client is None

    def update(self):
        if self.client is not None:
            if self.is_eating_done():
                self.remove_client()
            else:
                self.chemicals += self.client.reaction
                self.chemicals[self.chemicals < 0] = 0

    def serve(self, menu_item: int):
        if not self.is_table_free():
            print("EVENT! Serving " + str(menu_item) + " to table " + str(self.index))
            self.chemicals += MENU[menu_item]
            self.client.ordered_meals[menu_item] = 1

    def estimate_consumption_end(self):
        if CONSUMPTION_END_ESTIMATOR_NET is not None:
            return CONSUMPTION_END_ESTIMATOR_NET.propagate(np.asmatrix(self.chemicals))
        return 1000  # Task for Aleksander


TABLES = [Table(index + 2) for index in range(10 - 2)]


def assert_all():
    tol_sum = sum(CLIENTS[0].tolerance)
    for client in CLIENTS:
        assert len(client.reaction) == ELEMENT_NUM
        assert len(client.tolerance) == ELEMENT_NUM
        for tolerance in client.tolerance:
            assert tolerance >= 0
        assert sum(client.tolerance) == tol_sum

    for menu_item in MENU:
        assert len(menu_item) == ELEMENT_NUM

    for table in TABLES:
        assert len(table.chemicals) == ELEMENT_NUM

    assert len(ELEM_NAME) == ELEMENT_NUM
    assert len(Map.MAP) == Map.ROWS

    for row in Map.MAP:
        assert len(row) == Map.COLS


assert_all()


def get_table(index: int):
    index -= 2
    return TABLES[index] if 0 <= index < len(TABLES) else None


def print_tables():
    for (index, table) in enumerate(TABLES):
        print(str(index + 2) + ") " + str(table))


# This is the main loop that runs everything as should
def run():
    def draw(screen, font) -> bool:
        screen.fill((0, 0, 0))
        w, h = pygame.display.get_surface().get_size()
        restaurant_rect = pygame.Rect(0, 0, w*0.7, h)
        info_rect = pygame.Rect(w*0.7, 0, w*0.3, h)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        Map.draw(screen, TABLES, restaurant_rect)
        for (i, table) in enumerate(TABLES):
            FONT_COLOR = (255, 255, 255)
            client_type: str = "" if table.client is None else str(table.client.kind)
            t = font.render(str(i)+":"+client_type+":"+"/".join(map(str, table.chemicals)), False, FONT_COLOR)
            screen.blit(t, (info_rect.x, info_rect.y+font.get_height() * i))

        pygame.display.flip()
        return True

    def ascii():
        Map.print()
        print_tables()

    def update():
        for table in TABLES:
            table.update()

        if np.random.randint(0, 4) == 0:
            random_table = TABLES[np.random.randint(0, len(TABLES))]
            if random_table.is_table_free():
                random_table.put_random_client()
            else:
                if 0 == np.random.randint(0, 4):
                    recommended_meal = random_table.client.recommend_meal()
                    random_table.serve(recommended_meal)

        max_estimated_consumption_end: float = -1000000
        table_closest_to_consumption_end: Table = None
        for table in TABLES:
            estimated_consumption_end = table.estimate_consumption_end()
            if estimated_consumption_end > max_estimated_consumption_end:
                max_estimated_consumption_end = estimated_consumption_end
                table_closest_to_consumption_end = table

        Map.WAITER.move_towards_table(table_closest_to_consumption_end)
        served_table = Map.WAITER.get_currently_served_table()
        if served_table is not None and not served_table.is_table_clean() and served_table.is_table_free():
            served_table.clean_chemicals()

    ASCII_MODE = False

    if ASCII_MODE:
        while True:
            ascii()
            update()
            pygame.time.wait(100)
    else:
        pygame.init()
        pygame.font.init()
        info_object = pygame.display.Info()

        screen = pygame.display.set_mode((int(info_object.current_w*0.6), int(info_object.current_h*0.6)), pygame.RESIZABLE)

        font = pygame.font.SysFont(pygame.font.get_default_font(), 30)

        while draw(screen, font):
            update()
            pygame.time.wait(100)


# This is a loop with minimal code needed to simulate customers and generate training data for Aleksander
def gen_plate_recognition_data(client_type: Client, error_threshold: float, model_file: str) -> Net:
    print(",".join(ELEM_NAME) + ",finished")
    xs = []
    ys = []
    zs = []
    colors = []
    for _ in range(1000):
        chemicals = np.array([np.random.randint(0, int(TOLERANCE_THRESHOLD/ELEMENT_NUM)) for _ in range(ELEMENT_NUM)])
        is_eating_done = int(client_type.is_eating_done(chemicals))
        # print(",".join(map(str, chemicals)) + "," + str(is_eating_done))
        xs += [chemicals[0]]
        ys += [chemicals[1]]
        zs += [chemicals[2]]
        colors += [is_eating_done]

    def visualize():
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(xs, ys, zs, c=colors)
        plt.show()

    # visualize()

    def visualize2(input_matrix: np.matrix, network: Net):
        # visualize()
        fig = plt.figure()
        ax = Axes3D(fig)
        output_matrix = list(network.propagate(input_matrix).flatten())
        ax.scatter(input_matrix[:, 0], input_matrix[:, 1], input_matrix[:, 2], c=output_matrix)
        plt.show()

    all = np.column_stack([xs, ys, zs, colors])
    ai_input = all[:, :-1]
    ai_expected = all[:, -1:]
    net = NetModel.build_net(3, [5, 5], 1, 0.0001)
    try:
        with open(model_file, 'rb') as model_file:
            net.weight_matrices = pickle.load(model_file)
            # net.plot_mesh(ai_input, ai_expected)
            # visualize2(ai_input, net)
            return net
    except FileNotFoundError:
        pass
    net.print_accuracy(ai_input, ai_expected)

    for epoch in range(50):
        # visualize2(ai_input, net)
        net.iterate(ai_input, ai_expected, 10000)
        normalised_error_sum = net.print_accuracy(ai_input, ai_expected)
        if abs(normalised_error_sum) <= error_threshold:
            break
    # visualize2(ai_input, net)
    with open(model_file, 'wb+') as model_file:
        pickle.dump(net.weight_matrices, model_file, pickle.HIGHEST_PROTOCOL)
    return net

# This is a loop with minimal code needed to simulate customers and generate training data for Miguel
def gen_plate_recommendation_data():
    # prints 1 if menu item was ordered, 1 if it wasn't
    # and at the end prints number of times such combination was ordered
    print(",".join([str(x) for x in range(len(MENU))]) + ",order_count")
    for (order, count) in ORDER_AND_COUNT:
        print(",".join(map(str, map(int, list(order)))) + "," + str(count))


# gen_plate_recommendation_data()

CONSUMPTION_END_ESTIMATOR_NET = gen_plate_recognition_data(CLIENTS[0], 0.01, 'model.pickle')
# while True:
#     x, y, z = map(int, input().split())
#     r = np.array([[x, y, z]])
#     print(CONSUMPTION_END_ESTIMATOR_NET.propagate(r))
run()
