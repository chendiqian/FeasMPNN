import os
import yaml


def save_run_config(args):
    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        exist_runs = [d for d in os.listdir('logs') if d.startswith(args.wandbname)]
        log_folder_name = f'logs/{args.wandbname}exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(vars(args), outfile, default_flow_style=False)
        return log_folder_name
    return None
