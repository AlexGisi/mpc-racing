import csv

member_names = ['steps', 'X', 'Y', 'yaw', 'vx', 'vy',
                'progress', 'error', 'cmd_throttle', 'cmd_steer']

class Logger:
    def __init__(self, fp) -> None:
        self.fp = fp
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
