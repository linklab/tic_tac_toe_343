# 선수 에이전트: RL 에이전트, 후수 에이전트: Dummy 에이전트
import os
from agents.b_human_agent import Human_Agent
from agents.c_dqn_agent_solution import TTTAgentDqn
from agents.d_reinforce_agent_solution import TTTAgentReinforce
from agents.e_a2c_agent_solution import TTTAgentA2C

from common.a_env_tic_tac_toe_343 import TicTacToe343
from common.d_utils import model_load

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def play_with_agent_1(env, agent_1):

    env.print_board_idx()
    state = env.reset()

    agent_2 = Human_Agent(name="AGENT_2", env=env)
    current_agent = agent_1

    print()

    print("[Q-Learning 에이전트 차례]")
    env.render()

    done = False
    while not done:
        if isinstance(current_agent, TTTAgentDqn):
            action = current_agent.get_action(state, mode="PLAY")
        else:
            action = current_agent.get_action(state)

        next_state, _, done, info = env.step(action)
        if current_agent == agent_1:
            print("     State:", state)
            # print("   Q-value:", current_agent.get_q_values_for_one_state(state))
            # print("    Policy:", current_agent.get_policy_for_one_state(state))
            print("    Action:", action)
            print("Next State:", next_state, end="\n\n")

        print("[{0}]".format(
            "당신(사람) 차례" if current_agent == agent_1 \
            else "Q-Learning 에이전트 차례"
        ))
        env.render()

        if done:
            if info['winner'] == 1:
                print("Q-Learning 에이전트가 이겼습니다.")
            elif info['winner'] == -1:
                print("당신(사람)이 이겼습니다. 놀랍습니다!")
            else:
                print("비겼습니다. 잘했습니다!")
        else:
            state = next_state

        if current_agent == agent_1:
            current_agent = agent_2
        else:
            current_agent = agent_1


if __name__ == '__main__':
    env = TicTacToe343()
    agent_1 = TTTAgentDqn(name="AGENT_1", env=env)
    model_load(agent_1.q, "")

    play_with_agent_1(env, agent_1)
