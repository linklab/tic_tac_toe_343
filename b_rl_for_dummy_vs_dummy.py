# 선수 에이전트: Dummy 에이전트, 후수 에이전트: RL 에이전트
from agents.b_human_agent import Human_Agent
from agents.a_dummy_agent import Dummy_Agent
from common.c_game_stats import draw_performance, print_game_statistics, print_step_status, GameStatus
from common.a_env_tic_tac_toe_343 import TicTacToe343

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 50_000

STEP_VERBOSE = False
BOARD_RENDER = False


def learning_for_dummy_vs_dummy():
    game_status = GameStatus()

    env = TicTacToe343()

    agent_1 = Dummy_Agent(name="AGENT_1", env=env)
    agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        current_agent = agent_1

        epsilon = 0

        if BOARD_RENDER:
            env.render()

        done = False
        STATE_2, ACTION_2 = None, None

        agent_2_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # agent_1 스텝 수행
            action_1 = agent_1.get_action(state)
            next_state, reward, done, info = env.step(action_1)
            print_step_status(
                current_agent, state, action_1, next_state,
                reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps,
                    game_status, agent_1, agent_2
                )
            else:
                # agent_2 스텝 수행
                state = next_state
                action = agent_2.get_action(state)
                next_state, reward, done, info = env.step(action)
                print_step_status(
                    agent_2, state, action, next_state,
                    reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps,
                        game_status, agent_1, agent_2
                    )
                else:
                    # agent_2에 방문한 현재 상태 및 수행한 행동 정보를
                    # 저장해 두었다가 추후 활용
                    STATE_2 = state
                    ACTION_2 = action

            state = next_state

    draw_performance(game_status, MAX_EPISODES)

    # 훈련 종료 직후 완전 탐욕적으로 정책 설정
    # 아래 내용 불필요 --> agent_1.get_action(state, epsilon=0.0)으로 해결 가능
    # agent_2.make_greedy_policy()


if __name__ == '__main__':
    learning_for_dummy_vs_dummy()
