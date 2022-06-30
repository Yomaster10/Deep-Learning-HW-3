import os
import abc
import sys
import torch
import torch.nn as nn
import tqdm.auto
from torch import Tensor
from typing import Any, Tuple, Callable, Optional, cast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cs236781.train_results import FitResult, BatchResult, EpochResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(
        self, model: nn.Module, device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs: int,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every: int = 1,
        post_epoch_fn: Callable = None,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """

        actual_num_epochs = 0
        epochs_without_improvement = 0

        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        best_acc = None

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.

            if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            # TODO: Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================

            # TODO:
            #  - Optional: Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            #  - Optional: Implement checkpoints. You can use the save_checkpoint
            #    method on this class to save the model to the file specified by
            #    the checkpoints argument.
            if best_acc is None or test_result.accuracy > best_acc:
                # ====== YOUR CODE: ======
                raise NotImplementedError()
                # ========================
            else:
                # ====== YOUR CODE: ======
                raise NotImplementedError()
                # ========================

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def save_checkpoint(self, checkpoint_filename: str):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        dirname = os.path.dirname(checkpoint_filename) or "."
        os.makedirs(dirname, exist_ok=True)
        torch.save({"model_state": self.model.state_dict()}, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and updates weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses=losses, accuracy=accuracy)


class RNNTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

    def train_epoch(self, dl_train: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        # TODO:
        #  Train the RNN model on one batch of data.
        #  - Forward pass
        #  - Calculate total loss over sequence
        #  - Backward pass: truncated back-propagation through time
        #  - Update params
        #  - Calculate number of correct char predictions
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        # Note: scaling num_correct by seq_len because each sample has seq_len
        # different predictions.
        return BatchResult(loss.item(), num_correct.item() / seq_len)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        with torch.no_grad():
            # TODO:
            #  Evaluate the RNN model on one batch of data.
            #  - Forward pass
            #  - Loss calculation
            #  - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================

        return BatchResult(loss.item(), num_correct.item() / seq_len)

class VAETrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            optimizer: Optimizer,
            device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the classifier model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        super().__init__(model, device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)
        # TODO: Train a VAE on one batch.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return BatchResult(loss.item(), 1 / data_loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)

        with torch.no_grad():
            # TODO: Evaluate a VAE on one batch.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================

        return BatchResult(loss.item(), 1 / data_loss.item())
