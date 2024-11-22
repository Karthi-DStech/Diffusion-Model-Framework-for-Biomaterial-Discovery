import os
import sys
import argparse
from torch.utils.data import DataLoader

from option.train_options import TrainOptions
from call_methods import make_dataset, make_model
from utils.utils import set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(opt: argparse.Namespace) -> None:
    """
    This function runs the training loop for the DDPM model.

    Process
    -------
    1. Set the seed
    2. Create the DDPM model
    3. Create the dataset
    4. Split the dataset into training and testing
    5. Create the dataloader
    6. Run the training loop
    7. Save the model and optimizer
    """

    set_seed(opt.seed)

    # Create the DDPM model
    model = make_model(model_name=opt.model_name, T=opt.Time_steps_FD, opt=opt)

    # Create dataset
    dataset = make_dataset(dataset_name=opt.dataset_name, opt=opt)

    if isinstance(dataset, tuple) and len(dataset) == 2:
        train_dataset, test_dataset = dataset
    else:
        train_dataset = dataset[0]

    train_dataloader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True
    )

    # Training loop
    for epoch in range(opt.n_epochs):
        for i, images in enumerate(train_dataloader):
            images = images.to(model._device)

            # Forward pass and training step
            eps, eps_predicted = model.train(batch_size=opt.batch_size, dataset=images)
            loss = model._compute_loss(eps, eps_predicted)  # Calculate the loss

            if i % opt.print_freq == 0:
                print(
                    f"Epoch {epoch} [{i}/{len(train_dataloader)}] - Loss: {loss.item()}"
                )

        # Save checkpoint periodically.
        if epoch > 0 and epoch % opt.save_freq == 0:
            model.save_networks(epoch)

    # Final save of the model and optimizer
    model.save_networks("final")
    print("Training complete. Model and optimizer saved.")


if __name__ == "__main__":

    opt = TrainOptions().parse()
    run(opt)
