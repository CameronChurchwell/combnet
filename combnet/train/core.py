import functools

import accelerate
import GPUtil
import torch
import torchutil
import time
import sys

import combnet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import penn
else:
    from combnet import penn

###############################################################################
# Train
###############################################################################


@torchutil.notify('train')
def train(dataset, directory=combnet.RUNS_DIR / combnet.CONFIG, gpu=None):
    """Train a model"""

    if dataset in ['mdb', 'ptdb'] or dataset == 'mdb+ptdb':
        if dataset == 'mdb+ptdb':
            datasets = ['mdb', 'ptdb']
        else:
            datasets = [dataset]

        # penn model setup
        def Model(name=None):
            return combnet.Model()
        penn.model.Model = Model

        penn_train = sys.modules['penn.train.core']
        original_penn_evaluate = penn_train.evaluate
        def wrapped_evaluate(directory, step, model, gpu, condition, loader):
            log_f0(directory, step, model)
            return original_penn_evaluate(directory, step, model, gpu, condition, loader)
        penn_train.evaluate = wrapped_evaluate
        penn.train(datasets, directory, gpu=gpu)
        return

    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)


    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(combnet.RANDOM_SEED)
    train_loader = combnet.data.loader(dataset, 'train', gpu=gpu)
    valid_loader = combnet.data.loader(dataset, 'valid', gpu=gpu)

    # _train_loader = combnet.data.loader(dataset, 'train', gpu=gpu)
    # valid_loader = combnet.data.loader(dataset, 'valid', gpu=gpu)
    # def tl():
    #     batch = next(iter(_train_loader))
    #     batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
    #     while True:
    #         yield batch

    # train_loader = tl()

    #################
    # Create models #
    #################

    model = combnet.Model().to(device)

    ####################
    # Create optimizer #
    ####################

    if combnet.PARAM_GROUPS is not None:
        print(combnet.PARAM_GROUPS)
        assert hasattr(model, 'parameter_groups')
        groups = model.parameter_groups()
        assert set(groups.keys()) == set(combnet.PARAM_GROUPS.keys())
        param_groups = []
        for name, g in combnet.PARAM_GROUPS.items():
            g['params'] = groups[name]
            param_groups.append(g)
        optimizer = combnet.OPTIMIZER_FACTORY(param_groups)
    else:
        optimizer = combnet.OPTIMIZER_FACTORY(model.parameters())

    ####################
    # Create scheduler #
    ####################

    scheduler: torch.optim.lr_scheduler.LRScheduler
    if combnet.SCHEDULER_FACTORY is not None:
        scheduler = combnet.SCHEDULER_FACTORY(optimizer, **combnet.SCHEDULER_KWARGS)
        # scheduler.step()
    else:
        scheduler = None

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
        if 'scheduler' in state and state['scheduler'] is not None:
            assert scheduler is not None
            scheduler.load_state_dict(state['scheduler'])

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
    # n_params = 0
    # for p in model.parameters():
    #     n_params += p.numel()
    # torchutil.tensorboard.update(directory, step, hyperparameters={
    #     'n_params': n_params
    # })

    # Setup progress bar
    progress = torchutil.iterator(
        range(step, combnet.STEPS),
        f'Training {combnet.CONFIG}, epoch={epoch}',
        step,
        combnet.STEPS)

    while step < combnet.STEPS:

        for batch in train_loader:

            x, y = batch
            x = x.to(device)
            y = y.to(device)

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), combnet.GRAD_CLIP_THRESHOLD, 'inf')

            # Update weights
            optimizer.step()

            ############
            # Evaluate #
            ############
            if step % combnet.LOG_INTERVAL == 0:
                if hasattr(model, 'parameter_groups'):
                    groups = model.parameter_groups()
                    if 'f0' in groups:
                        f = groups['f0'][0] # TODO expand to more than just first?
                        scaling_function = None
                        grouped_scalars = {}
                        for m in model.modules():
                            if hasattr(m, 'scaling_function'):
                                scaling_function: callable = getattr(m, 'scaling_function')
                                break
                        f = f.detach().cpu().flatten() #TODO expand to handle non-flat?
                        if scaling_function:
                            f = scaling_function(f)
                        f = {str(i): f_val for i, f_val in enumerate(f)}
                        grouped_scalars['f0'] = f
                        torchutil.tensorboard.update(directory, step, grouped_scalars=grouped_scalars)

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
                        gpu=gpu)
                    evaluate_fn('train', train_loader, evaluation_steps=evaluation_steps)
                    evaluate_fn('valid', valid_loader, evaluation_steps=None)

            ###################
            # Save checkpoint #
            ###################

            if step and step % combnet.CHECKPOINT_INTERVAL == 0:
                torchutil.checkpoint.save(
                    directory / f'{step:08d}.pt',
                    model,
                    optimizer,
                    step=step,
                    epoch=epoch,
                    scheduler=scheduler.state_dict() if scheduler is not None else None)

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
        progress.set_description(f'Training {combnet.CONFIG}, epoch={epoch}')

        if scheduler is not None:
            scheduler.step()
            print(f'Epoch {epoch}: stepping scheduler: {scheduler.get_last_lr()}')
            # if epoch % 50 == 0:
            # for i, param_group in enumerate(optimizer.param_groups):
            #     print(f"Epoch {epoch+1}, Param group {i}: LR = {param_group['lr']:.6f}")

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
        epoch=epoch,
        scheduler=scheduler.state_dict() if scheduler is not None else None)

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

    model.eval()

    with torch.inference_mode():
        # Setup evaluation metrics
        metrics = combnet.evaluate.Metrics()

        for i, batch in enumerate(loader):

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
        # print(scalars)

        # Write to tensorboard
        torchutil.tensorboard.update(directory, step, scalars=scalars)
    model.train()


def log_f0(directory, step, model):
    if hasattr(model, 'parameter_groups'):
        groups = model.parameter_groups()
        if 'f0' in groups:
            f = groups['f0'][0] # TODO expand to more than just first?
            scaling_function = None
            grouped_scalars = {}
            for m in model.modules():
                if hasattr(m, 'scaling_function'):
                    scaling_function: callable = getattr(m, 'scaling_function')
                    break
            f = f.detach().cpu().flatten() #TODO expand to handle non-flat
            if scaling_function:
                f = scaling_function(f)
            f = {str(i): f_val for i, f_val in enumerate(f)}
            grouped_scalars['f0'] = f
            torchutil.tensorboard.update(directory, step, grouped_scalars=grouped_scalars)

###############################################################################
# Loss function
###############################################################################


def loss(logits, target):
    """Compute loss function"""
    # if isinstance(target, tuple):
    #     target = torch.tensor(target).to(logits.device)
    if combnet.LOSS_FUNCTION is not None:
        return combnet.LOSS_FUNCTION(logits, target)
    else: # classification
        return torch.nn.functional.cross_entropy(logits, target)
