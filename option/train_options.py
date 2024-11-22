import os
import sys

from option.base_options import BaseOptions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainOptions(BaseOptions):

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        BaseOptions.initialize(self)

        self._parser.add_argument(
            "--model_name",
            type=str,
            default="ddpm_35m",
            help="Name of the model to use",
        )

        self._parser.add_argument(
            "--lr", type=float, default=2e-5, help="Learning rate"
        )

        self._parser.add_argument(
            "--Time_steps_FD",
            type=int,
            default=1000,
            help="Number of time steps for the diffusion model",
        )

        self._parser.add_argument(
            "--nb_images", type=int, default=16, help="Number of images to generate"
        )

        self._parser.add_argument(
            "--unet_ch",
            type=int,
            default=128,
            help="Number of filters for the UNet",
        )

        self._parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "adamw"],
            help="Optimizer to use",
        )

        self._parser.add_argument(
            "--mean",
            type=float,
            default=0.5,
            help="Mean of the dataset",
        )

        self._parser.add_argument(
            "--std",
            type=float,
            default=0.5,
            help="Standard deviation of the dataset",
        )

        self._parser.add_argument(
            "--continue_train",
            type=bool,
            default=False,
            help="Continue training",
        )

        self._parser.add_argument(
            "--print_freq",
            type=int,
            default=20,
            help="Print frequency",
        )

        self._parser.add_argument(
            "--save_freq",
            type=int,
            default=4000,
            help="Checkpoint saving frequency of the model over epochs",
        )

        # New parameters should be added here

        self._is_train = True
