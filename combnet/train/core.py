import functools

import accelerate
import GPUtil
import torch
import torchutil
import time

import combnet


###############################################################################
# Train
###############################################################################


@torchutil.notify('train')
def train(dataset, directory=combnet.RUNS_DIR / combnet.CONFIG, gpu=None):
    """Train a model"""
    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)


    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(combnet.RANDOM_SEED)
    train_loader = combnet.data.loader(dataset, 'train', gpu=gpu)
    valid_loader = combnet.data.loader(dataset, 'valid', gpu=gpu)

    #################
    # Create models #
    #################

    model = combnet.Model().to(device)

    ####################
    # Create optimizer #
    ####################

    if combnet.PARAM_GROUPS is not None:
        assert hasattr(model, 'parameter_groups')
        groups = model.parameter_groups()
        assert set(groups.keys()) == set(combnet.PARAM_GROUPS.keys())
        param_groups = []
        for name, g in combnet.PARAM_GROUPS.items():
            g['params'] = groups[name]
            param_groups.append(g)
        import pdb; pdb.set_trace()
        optimizer = combnet.OPTIMIZER_FACTORY(param_groups)
    else:
        optimizer = combnet.OPTIMIZER_FACTORY(model.parameters())

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = torchutil.checkpoint.latest_path(directory)

    if path is not None:

        # Load model
        model, optimizer, state = torchutil.checkpoint.load(
            path,
            model,
            optimizer)
        step, epoch = state['step'], state['epoch']

    else:

        # Train from scratch
        step, epoch = 0, 0

    ####################
    # Device placement #
    ####################

    # accelerator = accelerate.Accelerator(mixed_precision='fp16')
    # model, optimizer, train_loader, valid_loader = accelerator.prepare(
    #     model,
    #     optimizer,
    #     train_loader,
    #     valid_loader)

    #########
    # Train #
    #########

    # Setup progress bar
    progress = torchutil.iterator(
        range(step, combnet.STEPS),
        f'Training {combnet.CONFIG}',
        step,
        combnet.STEPS)

    while step < combnet.STEPS:

        for batch in train_loader:

            # TODO - generalize
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # if step % 1000 == 0:
            #     print(torch.cuda.memory_summary(device))

            # Forward pass
            z = model(x)

            # Compute loss
            losses = loss(z, y)

            ##################
            # Optimize model #
            ##################

            # Zero gradients
            optimizer.zero_grad()

            # Backward pass
            losses.backward()

            # Update weights
            optimizer.step()

            ############
            # Evaluate #
            ############

            if step % combnet.EVALUATION_INTERVAL == 0:
                with combnet.inference_context(model):
                    evaluation_steps = (
                        None if step == combnet.STEPS
                        else combnet.DEFAULT_EVALUATION_STEPS)
                    evaluate_fn = functools.partial(
                        evaluate,
                        directory,
                        step,
                        model,
                        evaluation_steps=evaluation_steps,
                        gpu=gpu)
                    evaluate_fn('train', train_loader)
                    evaluate_fn('valid', valid_loader)

            ###################
            # Save checkpoint #
            ###################

            if step and step % combnet.CHECKPOINT_INTERVAL == 0:
                torchutil.checkpoint.save(
                    directory / f'{step:08d}.pt',
                    model,
                    optimizer,
                    step=step,
                    epoch=epoch)

            ########################
            # Termination criteria #
            ########################

            # Finished training
            if step >= combnet.STEPS:
                break

            # Raise if GPU tempurature exceeds 90 C
            # if step % 100 == 0 and (any(gpu.temperature > 90. for gpu in GPUtil.getGPUs())):
            #         raise RuntimeError(f'GPU is overheating. Terminating training.')

            ###########
            # Updates #
            ###########

            # Update progress bar
            progress.update()

            # Update training step count
            step += 1

        # Update epoch
        epoch += 1

    # Close progress bar
    progress.close()

    # Save final model
    checkpoint_file = directory / f'{step:08d}.pt'
    torchutil.checkpoint.save(
        checkpoint_file,
        model,
        optimizer,
        # accelerator=accelerator,
        step=step,
        epoch=epoch)

    combnet.evaluate.datasets(checkpoint=checkpoint_file, gpu=gpu)


###############################################################################
# Evaluation
###############################################################################

stop_at_evaluate = False
def evaluate(
    directory,
    step,
    model,
    # accelerator,
    condition,
    loader,
    evaluation_steps=None,
    gpu=None
):
    if condition == 'valid' and stop_at_evaluate:
        breakpoint()
    """Perform model evaluation"""

    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    with torch.no_grad():
        # Setup evaluation metrics
        metrics = combnet.evaluate.Metrics()

        for i, batch in enumerate(loader):

            # TODO - generalize
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            z = model(x)

            # Update metrics
            metrics.update(
                z, y
            )

            # Stop when we exceed some number of batches
            if evaluation_steps is not None and i + 1 == evaluation_steps:
                break

        # Format results
        scalars = {
            f'{key}/{condition}': value for key, value in metrics().items()}

        # Write to tensorboard
        torchutil.tensorboard.update(directory, step, scalars=scalars)


###############################################################################
# Loss function
###############################################################################


def loss(logits, target):
    """Compute loss function"""
    # if isinstance(target, tuple):
    #     target = torch.tensor(target).to(logits.device)
    return torch.nn.functional.cross_entropy(logits, target)
