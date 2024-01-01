import pickle
import random
import pygame
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import copy
from game import DQN, Tictactoe_v0
from keras.layers import Dense
from keras import losses
from keras import optimizers
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt


def load_agent(file_path):
    env = Tictactoe_v0()
    op1 = optimizers.RMSprop(learning_rate=0.00025)

    op2 = optimizers.RMSprop(learning_rate=0.00025)
    agent = DQN(0.7, 0.1, 4096, 1048576)
    agent.training_network.add(Dense(128, activation="relu", input_shape=(9,)))
    agent.training_network.add(Dense(128, activation="relu"))
    agent.training_network.add(Dense(9, activation="linear"))
    agent.training_network.compile(optimizer=op1, loss=losses.mean_squared_error)

    agent.target_network.add(Dense(128, activation="relu", input_shape=(9,)))
    agent.target_network.add(Dense(128, activation="relu"))
    agent.target_network.add(Dense(9, activation="linear"))
    agent.target_network.compile(optimizer=op2, loss=losses.mean_squared_error)
    agent.update_target_network()
    with open(file_path, "rb") as f:
        saved_weights = pickle.load(f)

    agent.set_weight(saved_weights)
    return agent


class TicTacToe:
    X_MARK = "X"
    O_MARK = "O"

    def __init__(self, callback):
        if not callable(callback):
            raise Exception("TicTacToe need a function to retrieve the next move")
        self.callback = callback

    def _resetBoard(self):
        self.board = [[None, None, None], [None, None, None], [None, None, None]]

    def _getMark(self, mark):
        if mark == TicTacToe.X_MARK:
            return TicTacToe.O_MARK
        else:
            return TicTacToe.X_MARK

    def _getEmpty(self):
        empty = []
        for ri, row in enumerate(self.board):
            for ci, cell in enumerate(row):
                if cell is None:
                    empty.append((ri, ci))
        return empty

    def _getRandomMove(self):
        empty = self._getEmpty()
        return random.choice(empty)

    def _playMove(self, move, mark=None):
        if not mark:
            mark = self._getMark(self.mark)
        row, col = move
        if self.board[row][col] != None:
            return -1
        self.board[row][col] = mark
        return 1 if not self._getEmpty() else 0

    def _checkBoard(self):
        b = self.board
        for i in range(3):
            if b[i][0] and b[i][0] == b[i][1] and b[i][1] == b[i][2]:  # row
                return b[i][0]
            if b[0][i] and b[0][i] == b[1][i] and b[1][i] == b[2][i]:  # column
                return b[0][i]
        if b[0][0] and b[0][0] == b[1][1] and b[1][1] == b[2][2]:  # diagonal
            return b[0][0]
        if b[0][2] and b[0][2] == b[1][1] and b[1][1] == b[2][0]:  # diagonal
            return b[0][2]
        return None

    def _printBoard(self):
        p = lambda row, col: self.board[row][col] or " "
        print("\n -----")
        print("|" + p(0, 0) + "|" + p(0, 1) + "|" + p(0, 2) + "|")
        print(" -----")
        print("|" + p(1, 0) + "|" + p(1, 1) + "|" + p(1, 2) + "|")
        print(" -----")
        print("|" + p(2, 0) + "|" + p(2, 1) + "|" + p(2, 2) + "|")
        print(" -----\n")

    def simulateGame(self, mark="X", play_first=False, verbose=False):
        self.mark = mark
        self._resetBoard()
        printBoard = lambda: self._printBoard() if verbose else None
        if not play_first:
            move = self._getRandomMove()
            self._playMove(move)
        empty = self._getEmpty()
        win = None
        while empty and not win:
            printBoard()
            move = self.callback(self.board, empty, mark)
            self._playMove(move, mark)
            win = self._checkBoard()
            if not self._getEmpty() or win:
                break
            printBoard()
            move = self._getRandomMove()
            self._playMove(move)
            empty = self._getEmpty()
            win = self._checkBoard()
        printBoard()

        if win == mark:
            return 1  # win
        elif win == self._getMark(mark):
            return -1  # lose
        else:
            return 0  # draw

    def simulate(self, n_games):
        win = 0
        for _ in range(n_games):
            mark = random.choice([TicTacToe.X_MARK, TicTacToe.O_MARK])
            play_first = random.choice([True, False])
            res = self.simulateGame(mark=mark, play_first=play_first)
            if res == 1:
                win += 1
        return win


'''random.seed(str(datetime.now()).encode())


def placeMark(board_state, empty_cells, mark):
    """Return the position to place the mark.
    Ex:
        board_state: [[X, O, X], [X, None, O], [O, None, X]]
        empty_cells: [(1, 1), (2, 1)]
        mark: 'X'
    """
    return random.choice(empty_cells)'''


def toStr(o):
    """Makes list/tuple readable and clean"""
    if isinstance(o, list):
        return str(o).translate(str.maketrans("", "", "'[]"))
    elif isinstance(o, tuple):
        return str(o).strip("()").replace(", ", "-")


def playGame(n_games):
    games = []
    logs = []

    def placeMark(board_state, empty_cells, mark):
        move = random.choice(empty_cells)  # randomly choose next move from empty cells
        logs.append((copy.deepcopy(board_state), move))  # deepcopy for list of lists
        return move

    tic = TicTacToe(placeMark)
    for _ in range(n_games):
        logs = []
        mark = random.choice([TicTacToe.X_MARK, TicTacToe.O_MARK])
        play_first = random.choice([True, False])
        win = tic.simulateGame(mark=mark, play_first=play_first)
        for i, (board_state, move) in enumerate(reversed(logs)):
            if win == 1:
                if i == 0:
                    result = 1.0
                else:
                    result = 0.6
            elif win == 0:
                if i == 0:
                    result = 0
                else:
                    result = 0.4
            else:
                if i == 0:
                    result = -1.0
                else:
                    result = -0.4

            games.append(
                {
                    "mark": mark,
                    "board_state": toStr(board_state),
                    "move": toStr(move),
                    "result": result,
                }
            )
    return games


N_GAMES = 100000
games = playGame(N_GAMES)
df = pd.DataFrame(games)


train = pd.DataFrame(
    {
        "mark": df["mark"],
        "board_state": df["board_state"],
        "move": df["move"],
        "result": df["result"],
    }
)

bs_encoder = LabelEncoder()
train["board_state"] = bs_encoder.fit_transform(train["board_state"])

mark_encoder = LabelEncoder()
train["mark"] = mark_encoder.fit_transform(train["mark"])

move_encoder = LabelEncoder()
train["move"] = move_encoder.fit_transform(train["move"])

# Assuming df["result"] contains strings like "win", "lose", "draw"
train["result"] = df["result"]

y = train["result"]
X = train.drop("result", axis=1)

"""
model = DecisionTreeRegressor()
model.fit(X, y)"""


def eval():
    # Preprocess the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the models
    models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    # Evaluate the models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1_score = f1_score(y_test, y_pred)
        print(f"{name} accuracy: {accuracy}")
        print(f"{name} recall: {recall}")
        print(f"{name} F1 score: {f1_score}")
        print()

    # Plot the ROC curves
    plt.figure()
    for name, model in models.items():
        y_pred = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (area = {auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def getMoveFromPred(preds, empty):
    """Decode and format the predicted move"""
    p = max(preds, key=lambda x: x[0])  # get the max value for predicted result
    move_dec = move_encoder.inverse_transform([p[1]])[
        0
    ]  # decode from int to categorical value
    row, col = move_dec.split("-")
    return (int(row), int(col))


def placeMark(board_state, empty_cells, mark):
    """Predict the result for each possible move"""
    preds = []
    empty_index = move_encoder.transform(
        [toStr(e) for e in empty_cells]
    )  # transform empty cells to index using encoder
    for i in empty_index:
        p = np.reshape(
            [
                bs_encoder.transform([toStr(board_state)])[0],
                mark_encoder.transform([mark])[0],
                i,
            ],
            (1, -1),
        )
        preds.append(
            (model.predict(p), i)
        )  # predict result for each possible move and store in a list

    for val, i in preds:
        print(f"i: {i}, val: {val}")

    move = getMoveFromPred(preds, empty_cells)
    print(move)
    return move


def find_empty_positions(state):
    """Return a list of all empty positions on the board."""
    return [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0]


class TicTacToeAI:
    X_MARK = "X"
    O_MARK = "O"

    def __init__(self, callback_X, callback_O):
        if not callable(callback_X) or not callable(callback_O):
            raise Exception("TicTacToeAI need two functions to retrieve next moves")
        self.callback_X = callback_X
        self.callback_O = callback_O

    def _resetBoard(self):
        self.board = [[None, None, None], [None, None, None], [None, None, None]]

    def _getEmpty(self):
        empty = []
        for ri, row in enumerate(self.board):
            for ci, cell in enumerate(row):
                if cell is None:
                    empty.append((ri, ci))
        return empty

    def _playMove(self, move, mark):
        row, col = move
        if self.board[row][col] != None:
            return -1
        self.board[row][col] = mark
        return 1 if not self._getEmpty() else 0

    def _checkBoard(self):
        b = self.board
        for i in range(3):
            if b[i][0] and b[i][0] == b[i][1] and b[i][1] == b[i][2]:  # row
                return b[i][0]
            if b[0][i] and b[0][i] == b[1][i] and b[1][i] == b[2][i]:  # column
                return b[0][i]
        if b[0][0] and b[0][0] == b[1][1] and b[1][1] == b[2][2]:  # diagonal
            return b[0][0]
        if b[0][2] and b[0][2] == b[1][1] and b[1][1] == b[2][0]:  # diagonal
            return b[0][2]
        return None

    def _printBoard(self):
        p = lambda row, col: self.board[row][col] or " "
        print("\n -----")
        print("|" + p(0, 0) + "|" + p(0, 1) + "|" + p(0, 2) + "|")
        print(" -----")
        print("|" + p(1, 0) + "|" + p(1, 1) + "|" + p(1, 2) + "|")
        print(" -----")
        print("|" + p(2, 0) + "|" + p(2, 1) + "|" + p(2, 2) + "|")
        print(" -----\n")

    def _getSeq(self, play_first):
        if play_first == "X":
            return [("X", self.callback_X), ("O", self.callback_O)]
        else:
            return [("O", self.callback_O), ("X", self.callback_X)]

    def simulateGame(self, play_first="X", verbose=False):
        self._resetBoard()
        sequence = self._getSeq(play_first)
        printBoard = lambda: self._printBoard() if verbose else None
        empty = self._getEmpty()
        win = None
        while empty and not win:
            for mark, callback in sequence:
                printBoard()
                move = callback(self.board, empty, mark)
                self._playMove(move, mark)
                win = self._checkBoard()
                empty = self._getEmpty()
                if not empty or win:
                    break
        return win if win in ["X", "O"] else "D"

    def simulate(self, n_games):
        win_X = 0
        win_O = 0
        for _ in range(n_games):
            # play_first = random.choice(["X", "O"])
            play_first = "X"
            res = self.simulateGame(play_first=play_first)
            if res == "X":
                win_X += 1
            elif res == "O":
                win_O += 1
        return (win_X, win_O)


def placeMark1(board_state, empty_cells, mark):
    # X
    return random.choice(empty_cells)


def placeMark2(board_state, empty_cells, mark):
    # O
    return random.choice(empty_cells)


def placeMarkMinimax(board_state, empty_cells, mark):
    best_score = float("-inf")
    best_move = None

    for move in empty_cells:
        # Thử mỗi ô trống trên bàn cờ
        row, col = move
        board_state[row][col] = mark
        score = minimax(board_state, 3, False)
        board_state[row][col] = None  # Hoàn trả bàn cờ

        # Cập nhật nước đi tốt nhất
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def minimax(board_state, depth, is_maximizing, alpha=float("-inf"), beta=float("inf")):
    winner = _checkBoard(board_state)
    if winner:
        return 1 if winner == "X" else -1 if winner == "O" else 0
    elif not _getEmpty(board_state):
        return 0

    if is_maximizing:
        max_eval = float("-inf")
        for move in _getEmpty(board_state):
            row, col = move
            board_state[row][col] = "X"
            eval = minimax(board_state, depth + 1, False, alpha, beta)
            board_state[row][col] = None
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Cắt tỉa Alpha-Beta
        return max_eval
    else:
        min_eval = float("inf")
        for move in _getEmpty(board_state):
            row, col = move
            board_state[row][col] = "O"
            eval = minimax(board_state, depth + 1, True, alpha, beta)
            board_state[row][col] = None
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Cắt tỉa Alpha-Beta
        return min_eval


# Hàm kiểm tra bàn cờ
def _checkBoard(board_state):
    b = board_state
    for i in range(3):
        if b[i][0] and b[i][0] == b[i][1] and b[i][1] == b[i][2]:  # row
            return b[i][0]
        if b[0][i] and b[0][i] == b[1][i] and b[1][i] == b[2][i]:  # column
            return b[0][i]
    if b[0][0] and b[0][0] == b[1][1] and b[1][1] == b[2][2]:  # diagonal
        return b[0][0]
    if b[0][2] and b[0][2] == b[1][1] and b[1][1] == b[2][0]:  # diagonal
        return b[0][2]
    return None


# Hàm lấy các ô trống
def _getEmpty(board_state):
    empty = []
    for ri, row in enumerate(board_state):
        for ci, cell in enumerate(row):
            if cell is None:
                empty.append((ri, ci))
    return empty


# Bàn cờ
#
#
#
#

WINDOW_SIZE = 300
BOARD_SIZE = 3
CELL_SIZE = WINDOW_SIZE // BOARD_SIZE

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def draw_board(screen, board):
    screen.fill(WHITE)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1)

            if board[row, col] == 1:
                pygame.draw.line(
                    screen,
                    RED,
                    (x + 10, y + 10),
                    (x + CELL_SIZE - 10, y + CELL_SIZE - 10),
                    2,
                )
                pygame.draw.line(
                    screen,
                    RED,
                    (x + CELL_SIZE - 10, y + 10),
                    (x + 10, y + CELL_SIZE - 10),
                    2,
                )
            elif board[row, col] == -1:
                pygame.draw.circle(
                    screen,
                    BLUE,
                    (x + CELL_SIZE // 2, y + CELL_SIZE // 2),
                    CELL_SIZE // 2 - 10,
                    2,
                )

    pygame.display.flip()


def check_winner(board, player):
    # Kiểm tra hàng và cột
    for i in range(3):
        # Kiểm tra hàng
        if np.all(board[i, :] == player):
            return True
        # Kiểm tra cột
        if np.all(board[:, i] == player):
            return True

    # Kiểm tra đường chéo chính
    if np.all(np.diag(board) == player):
        return True

    # Kiểm tra đường chéo phụ
    if np.all(np.diag(np.fliplr(board)) == player):
        return True

    return False


def play_game(model, first_turn=True):
    game_board = np.zeros((3, 3), dtype=int)
    player_turn = 1 if first_turn else -1

    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("Caro Game (3x3)")

    draw_board(screen, game_board)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN and player_turn == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                col = mouse_x // 100
                row = mouse_y // 100
                move = row * 3 + col

                if 0 <= move <= 48 and game_board[row, col] == 0:
                    game_board[row, col] = player_turn
                    draw_board(screen, game_board)

                    if check_winner(game_board, player_turn):
                        print(f"Player {player_turn} wins!")
                        return

                    if np.all(game_board != 0):
                        print("It's a draw!")
                        return

                    player_turn = player_turn * (-1)
            if player_turn == -1:
                empty_cells = find_empty_positions(game_board)
                board_state = [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ]
                for i in range(3):
                    for j in range(3):
                        if game_board[i][j] == 1:
                            board_state[i][j] = "X"
                        elif game_board[i][j] == -1:
                            board_state[i][j] = "O"
                        else:
                            board_state[i][j] = None
                bot_move = placeMark(board_state, empty_cells, TicTacToe.O_MARK)
                row_bot, col_bot = bot_move
                game_board[row_bot, col_bot] = player_turn

                draw_board(screen, game_board)

                if check_winner(game_board, player_turn):
                    print(f"Player {player_turn} wins!")
                    return

                if np.all(game_board != 0):
                    print("It's a draw!")
                    return

                player_turn = player_turn * (-1)

        draw_board(screen, game_board)
        pygame.display.flip()


modelDQN = load_agent("tictactoe_save_500000.pkl")


def placeMarkDQN(board_state, empty_cells, mark):
    game_board = np.zeros((3, 3), dtype=int)

    for i in range(3):
        for j in range(3):
            if board_state[i][j] == "X":
                game_board[i][j] = 1
            elif board_state[i][j] == "O":
                game_board[i][j] = -1
            else:
                game_board[i][j] == 0

    state = game_board.ravel()

    move = modelDQN.observe(state, [i for i in range(len(state)) if state[i] == 0])
    row, col = move // 3, move % 3
    return (row, col)


with open("model_5000000.pkl", "rb") as f:
    model = pickle.load(f)


if __name__ == "__main__":
    pygame.init()
    # Để tải lại mô hình
    n_games = 10000
    # eval()
    # agent = load_agent("save/tictactoe_save_500000.pkl")
    play_game(model, first_turn=True)
    pygame.quit()

    """win_X, win_O = TicTacToeAI(placeMark1, placeMark).simulate(n_games)
    print(
        f"Player X won {win_X} out of {n_games} games (win rate = {round((win_X/n_games) * 100, 2)}%)"
    )
    print(
        f"Player O won {win_O} out of {n_games} games (win rate = {round((win_O/n_games) * 100, 2)}%)"
    )"""
