from pathlib import Path
import subprocess
import os
from tqdm import tqdm

module = 'combnet'

checkpoint_file = '00010000.pt'

dir = Path(__file__).parent.parent / 'runs.conv'
assert dir.exists()

dataset = 'chords'

gpu = str(2)

run_dirs = list(dir.iterdir())

for i, run_dir in tqdm(enumerate(run_dirs), 'evaluating all chords conv runs', total=len(run_dirs), dynamic_ncols=True):

    # if i < 28:
    #     continue

    name = run_dir.name
    try:
        _, _, lr, kernel_size, n_channels = name.split('-')
    except:
        _, _, lr0, lr1, kernel_size, n_channels = name.split('-')
        lr = '-'.join([lr0, lr1])
    lr = lr.replace('_', '.')

    env = os.environ.copy()
    env['COMBNET_NO_SEARCH'] = "1"
    env['COMBNET_BASE_CONFIG'] = str(Path(__file__).parent.parent / 'config' / 'chords.py')
    env['COMBNET_LEARNING_RATE'] = str(lr)
    env['COMBNET_KERNEL_SIZE'] = str(kernel_size)
    env['COMBNET_N_CHANNELS'] = str(n_channels)

    config = list(run_dir.glob('*.py'))
    assert(len(config) == 1)
    config = config[0]

    checkpoint = run_dir / checkpoint_file
    if not checkpoint.exists():
        args = [
            'python',
            '-m',
            f'{module}.train',
            '--dataset', dataset,
            '--gpu', gpu,
            '--config', str(config)
        ]
        subprocess.run(args=args, env=env)

    assert checkpoint.exists(), breakpoint()

    args = [
        'python',
        '-m',
        f'{module}.evaluate',
        '--datasets', dataset,
        '--gpu', gpu,
        '--config', str(config),
        '--checkpoint', str(checkpoint)
    ]
    # print('arguments:', args)
    subprocess.run(args=args, env=env)