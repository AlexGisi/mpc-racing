import os
import pickle

member_names = ['steps', 'X', 'Y', 'yaw', 'vx', 'vy',
                'progress', 'error', 'cmd_throttle', 'cmd_steer',
                'next_left_lane_point_x', 'next_left_lane_point_y', 'next_right_lane_point_x',
                'next_right_lane_point_y', 'last_ts']

class Logger:
    def __init__(self, fp, mpc_fp=None) -> None:
        self.fp = fp
        self.mpc_fp = mpc_fp
        self.write_header()

        if mpc_fp:
            os.makedirs(mpc_fp, exist_ok=False)

    def write_header(self):
        with open(self.fp, 'w') as f:
            f.write(','.join(member_names))
            f.write('\n')
    
    def log(self, agent):
        with open(self.fp, 'a') as f:
            f.write(self.log_str(agent))
    
    def log_str(self, agent):
        agent_vars = vars(agent)
        log_str = ""
        for i, prop in enumerate(member_names):
            log_str += str(agent_vars[prop])
            if i != len(member_names) - 1:
                log_str += ","
        log_str += '\n'
        return log_str
    
    def pickle_mpc_res(self, agent):
        data = {}
        data['controlled'] = (agent.steps >= agent.start_control_at)
        data['step'] = agent.steps
        data['predicted_states'] = agent.predicted_states
        data['controls'] = agent.controls
        data['mean_ts'] = agent.mean_ts
        with open(os.path.join(self.mpc_fp, str(agent.steps)), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
