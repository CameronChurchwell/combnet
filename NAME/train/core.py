import functools

import accelerate
import GPUtil
import torch
import torchutil

import NAME


###############################################################################
# Train
###############################################################################


@torchutil.notify('train')
def train(datasets, directory=NAME.RUNS_DIR / NAME.CONFIG):
    """Train a model"""
    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(NAME.RANDOM_SEED)
    train_loader = NAME.data.loader(datasets, 'train')
    valid_loader = NAME.data.loader(datasets, 'valid')

    #################
    # Create models #
    #################

    model = NAME.Model()

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.Adam(model.parameters())

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

    accelerator = accelerate.Accelerator(mixed_precision='fp16')
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        valid_loader)

    #########
    # Train #
    #########

    # Setup progress bar
    progress = torchutil.iterator(
        range(step, NAME.STEPS),
        f'Training {NAME.CONFIG}',
        step,
        NAME.STEPS)

    while step < NAME.STEPS:

        for batch in train_loader:

            # TODO - Unpack batch
            () = batch

            # Forward pass
            () = model(
                # TODO - args
            )

            # Compute loss
            losses = loss(
                # TODO - args
            )

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

            if step % NAME.EVALUATION_INTERVAL == 0:
                with NAME.inference_context(model):
                    evaluation_steps = (
                        None if step == NAME.STEPS
                        else NAME.DEFAULT_EVALUATION_STEPS)
                    evaluate_fn = functools.partial(
                        evaluate,
                        directory,
                        step,
                        model,
                        accelerator,
                        evaluation_steps=evaluation_steps)
                    evaluate_fn('train', train_loader)
                    evaluate_fn('valid', valid_loader)

            ###################
            # Save checkpoint #
            ###################

            if step and step % NAME.CHECKPOINT_INTERVAL == 0:
                torchutil.checkpoint.save(
                    directory / f'{step:08d}.pt',
                    model,
                    optimizer,
                    accelerator=accelerator,
                    step=step,
                    epoch=epoch)

            ########################
            # Termination criteria #
            ########################

            # Finished training
            if step >= NAME.STEPS:
                break

            # Raise if GPU tempurature exceeds 80 C
            if any(gpu.temperature > 80. for gpu in GPUtil.getGPUs()):
                raise RuntimeError(f'GPU is overheating. Terminating training.')

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
    torchutil.checkpoint.save(
        directory / f'{step:08d}.pt',
        model,
        optimizer,
        accelerator=accelerator,
        step=step,
        epoch=epoch)


###############################################################################
# Evaluation
###############################################################################


def evaluate(
    directory,
    step,
    model,
    accelerator,
    condition,
    loader,
    evaluation_steps=None
):
    """Perform model evaluation"""
    # Setup evaluation metrics
    metrics = NAME.evaluate.Metrics()

    for i, batch in enumerate(loader):

        # TODO - unpack batch
        () = batch

        # Forward pass
        () = model(
            # TODO - args
        )

        # Update metrics
        metrics.update(
            # TODO - args
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
    # TODO
    pass
