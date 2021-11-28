import time
import numpy as np

from agents.a_dummy_agent import Dummy_Agent

PLAYER_TO_SYMBOL = [' ', 'O', 'X']
PLAYER_1_INT = 1
PLAYER_2_INT = -1

BOARD_ROWS = 3
BOARD_COLS = 4

#########################################################
##  (0,0) -> 0, (0,1) ->  1, (0,2) ->  2, (0,3) ->  3  ##
##  (1,0) -> 4, (1,1) ->  5, (1,2) ->  6, (1,3) ->  7  ##
##  (2,0) -> 8, (2,1) ->  9, (2,2) -> 10, (2,3) -> 11  ##
#########################################################
def position_to_action_idx(row_idx, col_idx):
    return 3*row_idx + col_idx


#########################################################
##  0 -> (0,0),  1 -> (0,1),  2 -> (0,2),  3 -> (0,3)  ##
##  4 -> (1,0),  5 -> (1,1),  6 -> (1,2),  7 -> (1,3)  ##
##  8 -> (2,0),  9 -> (2,1), 10 -> (2,2), 11 -> (2,3)  ##
#########################################################
def action_idx_to_position(idx):
    return idx//4, idx % 4


#########################################################
# 게임판 상태의 저장, 출력 그리고 종료 판정을 수행하는 State 클래스   #
#########################################################
class State:
    def __init__(self, board_rows=BOARD_ROWS, board_cols=BOARD_COLS):
        # 게임판 상태는 board_rows * board_cols 크기의 배열로 표현
        # 게임판에서 플레이어는 정수값으로 구분
        # 1 : 선공 플레이어, -1 : 후공 플레이어, 0 : 초기 공백 상태
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.board_size = board_rows * board_cols

        ### [NOTE] ###
        self.data = np.zeros(shape=[board_rows, board_cols], dtype=int)
        ##############

        self.winner = None

    # 현 상태에서 유효한 행동 ID 리스트 반환
    def get_available_actions(self):
        available_actions = []
        if self.is_end_state():
            return available_actions

        available_actions = [i for i in range(12) if self.data.flatten()[i] == 0]

        if len(available_actions) == 12:
            available_actions.remove(6)
            available_actions.remove(7)

        return available_actions

    # 플레이어가 종료 상태에 있는지 판단.
    # 플레이어가 게임을 이기거나, 지거나, 비겼다면 True 반환, 그 외는 False 반환
    def is_end_state(self):
        data_flat = self.data.flatten()

        # 14 win condition idx
        seq3_list = [[0,  1,  2],  # horizontal
                     [1,  2,  3],
                     [4,  5,  6],
                     [5,  6,  7],
                     [8,  9, 10],
                     [9, 10, 11],
                     [0,  4,  8],  # vertical
                     [1,  5,  9],
                     [2,  6, 10],
                     [3,  7, 11],
                     [0,  5, 10],  # diagonal
                     [1,  6, 11],
                     [2,  5,  8],
                     [3,  6,  9]]

        # Check if the match is over
        for seq3 in seq3_list:
            if data_flat[seq3[0]] == 0:
                continue

            if data_flat[seq3[0]] == data_flat[seq3[1]] == data_flat[seq3[2]]:
                self.winner = data_flat[seq3[0]]
                return True

        # Check if the match continues
        for i in data_flat:
            if i == 0:
                return False

        # Draw
        self.winner = 0
        return True

    # 게임판 상태 출력
    def get_state_as_board(self):
        board_str = "┌───┬───┬───┬───┐\n"
        for i in range(self.board_rows):
            board_str += '│'
            for j in range(self.board_cols):
                board_str += ' ' + PLAYER_TO_SYMBOL[int(self.data[i, j])] + ' │'
            board_str += '\n'

            if i < self.board_rows - 1:
                board_str += '├───┼───┼───┼───┤\n'
            else:
                board_str += '└───┴───┴───┴───┘\n'

        return board_str

    def __str__(self):
        return str([''.join(['O' if x == 1 else 'X' if x == -1 else '-' for x in y]) for y in self.data])


################################################################
# 플레이어 1,2 간의 게임 진행을 담당하는 Env 클래스
class TicTacToe343:
    def __init__(self):
        self.BOARD_SIZE = BOARD_ROWS * BOARD_COLS
        self.current_state = None  # 현재 상태 관리
        self.current_agent_int = None  # 현재 에이전트(플레이어) 관리

        self.INITIAL_STATE = State()  # 초기 상태 설정

    def reset(self):
        self.current_agent_int = PLAYER_1_INT
        self.current_state = self.INITIAL_STATE
        return self.current_state

    # 게임 진행을 위해 각 플레이어의 착수 때 마다 호출
    def step(self, action=None):
        # 플레이어의 행동에 의한 다음 상태 갱신
        position = action_idx_to_position(action)

        next_state = self.get_new_state(
            i=position[0], j=position[1],
            state_data=self.current_state.data,
            player_int=self.current_agent_int
        )

        done = next_state.is_end_state()

        if done:
            info = {
                'current_agent_int': self.current_agent_int,
                'winner': next_state.winner
            }
            if next_state.winner == PLAYER_1_INT:
                reward = 1.0
            elif next_state.winner == PLAYER_2_INT:
                reward = -1.0
            else:
                reward = 0.0
        else:
            info = {'current_agent_int': self.current_agent_int}
            reward = 0.0

        self.current_state = next_state

        if self.current_agent_int == PLAYER_1_INT:
            self.current_agent_int = PLAYER_2_INT
        else:
            self.current_agent_int = PLAYER_1_INT

        return next_state, reward, done, info

    def render(self, mode='human'):
        print(self.current_state.get_state_as_board())

    def get_new_state(self, i, j, state_data, player_int):
        new_state = State()

        # 주어진 상태의 게임판 상황 복사
        new_state.data = np.copy(state_data)

        # 플레이어의 행동(i, j 위치에 표시) 반영
        new_state.data[i, j] = player_int

        return new_state

    def print_board_idx(self):
        print()
        print("[[[Tic-Tac-Toe 보드 내 각 셀을 선택할 때 다음 숫자 키패드를 사용하세요.]]]")
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                out += str(position_to_action_idx(i, j)) + ' | '
            print(out)
        print('-------------')


def main():
    env = TicTacToe343()

    state = env.reset()
    observation = state.data.flatten()

    print(state, state.get_available_actions(), observation, "!!!!")

    env.render()

    agent_1 = Dummy_Agent(name="AGENT_1", env=env)
    agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    current_agent = agent_1

    done = False
    total_steps = 0

    while not done:
        total_steps += 1

        action = current_agent.get_action(state)

        next_state, reward, done, info = env.step(action)
        next_observation = next_state.data.flatten()

        print("[{0}] observation: {1}, action: {2}, next_observation: {3}, reward: {4}, "
              "done: {5}, info: {6}, total_steps: {7}".format(
            current_agent.name, observation, action, next_observation, reward,
            done, info, total_steps
        ))

        env.render()

        state = next_state
        observation = next_observation
        time.sleep(2)

        if current_agent == agent_1:
            current_agent = agent_2
        else:
            current_agent = agent_1


if __name__ == "__main__":
    main()
