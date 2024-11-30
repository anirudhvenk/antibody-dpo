import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import transformers

from collections import defaultdict
from packaging import version
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from torch.utils.data import Dataset
from trl import CPOTrainer, CPOConfig
from contextlib import nullcontext
from transformers import Trainer
from trl.data_utils import maybe_extract_prompt, maybe_apply_chat_template
from trl.trainer.utils import pad_to_length
from typing import Any, Callable, Literal, Optional, Union, Dict
from accelerate import PartialState
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from esm.sdk.api import ESMProtein, GenerationConfig

class ESMDataCollator:
    def __call__(self, features):
        batch = {}
        for batch_dict in features:
            for k in batch_dict:
                if k not in batch:
                    batch[k] = []
                batch[k].append(batch_dict[k])
        
        for k in batch:
            if k.endswith(("_input_ids", "_labels")):
                batch[k] = torch.tensor(batch[k])

        return batch

class ESMCPOTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
        compute_metrics=None,
    ):
        if peft_config:
            model = get_peft_model(model, peft_config)

        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else processing_class.pad_token_id
        self.truncation_mode = args.truncation_mode
        self.max_completion_length = args.max_completion_length
        self.processing_class = processing_class

        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.cpo_alpha = args.cpo_alpha

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        with PartialState().local_main_process_first():
            train_dataset = train_dataset.map(
                self.tokenize_row, 
                num_proc=args.dataset_num_proc, 
                load_from_cache_file=True
            )
            eval_dataset = eval_dataset.map(
                self.tokenize_row, 
                num_proc=args.dataset_num_proc, 
                load_from_cache_file=True
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def tokenize_row(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        batch = {}
        chosen = f"EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC{feature['chosen']}WGQGTLVTVSS"
        rejected = f"EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC{feature['rejected']}WGQGTLVTVSS"

        chosen_tokens = self.processing_class(
            chosen, truncation=False
        )

        rejected_tokens = self.processing_class(
            rejected, truncation=False
        )

        batch["chosen_input_ids"] = chosen_tokens["input_ids"]
        batch["rejected_input_ids"] = rejected_tokens["input_ids"]
        batch["chosen_labels"] = chosen_tokens["input_ids"]
        batch["rejected_labels"] = rejected_tokens["input_ids"]

        return batch

    def concatenated_inputs(
        self,
        batch: dict[str, Union[list, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 1,
        device: Optional[torch.device] = None,
    ):
        concatenated_batch = {}
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        return concatenated_batch

    def cpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the CPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        logits = (policy_chosen_logps - policy_rejected_logps).to(self.accelerator.device)

        # The beta is a temperature parameter for the CPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative CPO loss.

        if self.loss_type == "simpo":
            gamma_logratios = self.simpo_gamma / self.beta
            logits = logits - gamma_logratios
            # This reduces to Equation 3 from the CPO paper when label_smoothing -> 0.
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "sigmoid":
            # This reduces to Equation 3 from the CPO paper when label_smoothing -> 0.
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'simpo']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        loss_mask = labels != label_pad_token_id

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
    
    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                sequence_tokens=concatenated_batch["concatenated_input_ids"]
            )
        all_logits = outputs.sequence_logits[:,97:107]

        def cross_entropy_loss(logits, labels):
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.contiguous().view(-1, logits.shape[-1])
            labels = labels.contiguous().view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"][:,97:107].clone()

        if self.cpo_alpha == 0:
            nll_loss = torch.tensor(0.0).to(self.accelerator.device)
        else:
            nll_loss = cross_entropy_loss(all_logits, labels)

        all_logps = self.get_batch_logps(
            all_logits,
            labels,
            average_log_prob=self.loss_type in ["ipo", "simpo"],
            label_pad_token_id=self.label_pad_token_id
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the CPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]

        losses, chosen_rewards, rejected_rewards = self.cpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )

        loss = losses.mean() + self.cpo_alpha * policy_nll_loss
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        return loss, metrics

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        print(metrics)

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    
    def generate_from_model(self, model) -> str:
        prompt = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC__________WGQGTLVTVSS"
        protein = ESMProtein(sequence=prompt)
        protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=4, temperature=0.1))

        return protein.sequence[96:106]

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)
    
    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples

            generated_sequence = self.generate_from_model(self.model)

            self.log(
                {
                    "seq_log": wandb.Table(
                        columns=["Sequences"],
                        rows=[[generated_sequence]]
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float` or `None`, *optional*, defaults to `None`):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super().log(logs, start_time)
        else:  # transformers<=4.46
            return super().log(logs)