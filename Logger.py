import os
from datetime import datetime 
import pickle

member_names = ['steps', 'X', 'Y', 'yaw', 'vx', 'vy', 'yawdot',
                'progress', 'error', 'cmd_throttle', 'cmd_steer',
                'next_left_lane_point_x', 'next_left_lane_point_y', 'next_right_lane_point_x',
                'next_right_lane_point_y', 'last_ts']

class Logger:
    def __init__(self, runs_fp) -> None:
        self.log_dir = os.path.join(runs_fp, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.fp = os.path.join(self.log_dir, 'steps.csv')
        self.mpc_fp = os.path.join(self.log_dir, 'mpc')

        os.makedirs(self.mpc_fp, exist_ok=False)
        self.write_header()

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
        data['time'] = agent.mpc_time
        with open(os.path.join(self.mpc_fp, str(agent.steps)), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
