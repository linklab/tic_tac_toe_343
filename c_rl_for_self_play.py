# 선수 에이전트: RL 에이전트, 후수 에이전트: RL 에이전트
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from common.c_game_stats import draw_performance, print_game_statistics, \
    epsilon_scheduled, GameStatus
from common.a_env_tic_tac_toe_343 import TicTacToe343
from common.d_utils import EarlyStopping, PLAY_TYPE

# from agents.c_dqn_agent import TTTAgentDqn
from agents.c_dqn_agent_solution import TTTAgentDqn

# from agents.d_reinforce_agent import TTTAgentReinforce
from agents.d_reinforce_agent_solution import TTTAgentReinforce

# from agents.e_a2c_agent import TTTAgentA2C
from agents.e_a2c_agent_solution import TTTAgentA2C

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 70_000

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 100_000

STEP_VERBOSE = False
BOARD_RENDER = False


def learning_for_self_play():
    game_status = GameStatus()

    env = TicTacToe343()

    self_agent_1 = TTTAgentDqn(
        name="AGENT_1", env=env, gamma=0.99, learning_rate=0.00001,
        replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=500,
        min_buffer_size_for_training=100
    )
    # self_agent_1 = TTTAgentReinforce(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.001
    # )
    # self_agent_1 = TTTAgentA2C(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.001, batch_size=32
    # )

    self_agent_2 = TTTAgentDqn(
        name="SELF_AGENT", env=env, gamma=0.99, learning_rate=0.00001,
        replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=500,
        min_buffer_size_for_training=100
    )
    # self_agent_2.q = self_agent_1.q

    # self_agent_2 = TTTAgentReinforce(
    #     name="SELF_AGENT", env=env, gamma=0.99, learning_rate=0.001
    # )
    # self_agent_2.policy = self_agent_1.policy

    # self_agent_2 = TTTAgentA2C(
    #     name="SELF_AGENT", env=env, gamma=0.99, learning_rate=0.001, batch_size=32
    # )
    # self_agent_2.actor_critic_model = self_agent_1.actor_critic_model

    total_steps = 0

    early_stopping = EarlyStopping(target_win_rate=50.0)
    early_stop = False
    win_rate = 0.0
    agent_2_episode_td_error = 0.0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        epsilon = epsilon_scheduled(
            episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False
        STATE_2, ACTION_2 = None, None

        agent_1_episode_td_error = 0.0
        agent_2_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # self_agent_1 스텝 수행
            action = self_agent_1.get_action(state)
            next_state, reward, done, info = env.step(action)

            if done:
                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                win_rate = print_game_statistics(
                    info, episode, epsilon, total_steps,
                    game_status, PLAY_TYPE.SELF
                )

                # reward: self_agent_1가 착수하여 done=True
                # agent_1이 이기면 1.0, 비기면 0.0
                agent_1_episode_td_error += self_agent_1.learning(
                    state, action, next_state, reward, done
                )

                # 미루워 두었던 self_agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += self_agent_2.learning(
                        STATE_2, ACTION_2, next_state, -1.0 * reward, done
                    )
            else:
                # 미루워 두었던 self_agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += self_agent_2.learning(
                        STATE_2, ACTION_2, next_state, reward, done
                    )

                # self_agent_1이 방문한 현재 상태 및 수행한
                # 행동 정보를 저장해 두었다가 추후 활용
                STATE_1 = state
                ACTION_1 = action

                # self_agent_2 스텝 수행
                state = next_state
                action = self_agent_2.get_action(state)
                next_state, reward, done, info = env.step(action)

                if done:
                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    win_rate = print_game_statistics(
                        info, episode, epsilon, total_steps,
                        game_status, PLAY_TYPE.SELF
                    )

                    # reward: self_agent_2가 착수하여 done=True
                    # self_agent_2가 이기면 -1.0, 비기면 0.0
                    agent_2_episode_td_error += self_agent_2.learning(
                        state, action, next_state, -1.0 * reward, done
                    )

                    # 미루워 두었던 self_agent_1의 배치에 transition 정보 추가
                    agent_1_episode_td_error += self_agent_1.learning(
                        STATE_1, ACTION_1, next_state, reward, done
                    )
                else:
                    # self_agent_2이 방문한 현재 상태 및 수행한
                    # 행동 정보를 저장해 두었다가 추후 활용
                    STATE_2 = state
                    ACTION_2 = action

                    # 미루워 두었던 self_agent_1의 배치에 transition 정보 추가
                    agent_1_episode_td_error += self_agent_1.learning(
                        STATE_1, ACTION_1, next_state, reward, done
                    )

            state = next_state

        game_status.set_agent_1_episode_td_error(agent_1_episode_td_error)
        game_status.set_agent_2_episode_td_error(agent_2_episode_td_error)

        if episode > 5000:
            early_stop = early_stopping.check(
                self_agent_1.agent_type, PLAY_TYPE.SELF, win_rate, agent_2_episode_td_error, self_agent_1.model
            )
            if early_stop:
                break

    if not early_stop:
        early_stopping.save_checkpoint(
            self_agent_1.agent_type, PLAY_TYPE.BACK, win_rate, agent_2_episode_td_error, self_agent_1.model
        )

    draw_performance(game_status, MAX_EPISODES)


if __name__ == '__main__':
    learning_for_self_play()


