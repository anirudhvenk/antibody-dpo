import torch
import torch.nn as nn

from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from torch.utils.data import Dataset
from trl import CPOTrainer, CPOConfig
from transformers import Trainer
from trl.data_utils import maybe_extract_prompt, maybe_apply_chat_template
from trl.trainer.utils import pad_to_length
from typing import Any, Callable, Literal, Optional, Union, Dict
from accelerate import PartialState

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
        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else processing_class.pad_token_id
        # self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_completion_length = args.max_completion_length
        self.processing_class = processing_class

        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.cpo_alpha = args.cpo_alpha

        with PartialState().local_main_process_first():
            train_dataset = train_dataset.map(
                self.tokenize_row, 
                num_proc=args.dataset_num_proc, 
                load_from_cache_file=False
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

        print(chosen_tokens["input_ids"])

        batch["chosen_input_ids"] = chosen_tokens["input_ids"]
        batch["rejected_input_ids"] = rejected_tokens["input_ids"]
        batch["chosen_labels"] = [1] * len(chosen_tokens["input_ids"])
        batch["rejected_labels"] = [0] * len(chosen_tokens["input_ids"])

        return batch

    def concatenated_inputs(
        self,
        batch: dict[str, Union[list, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
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
        len_chosen = batch["chosen_input_ids"].shape[0]

        outputs = model(
            sequence_tokens=concatenated_batch["concatenated_input_ids"]
        )
        all_logits = outputs.sequence_logits

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()

        if self.cpo_alpha == 0:
            nll_loss = torch.tensor(0.0).to(self.accelerator.device)
        else:
            nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type in ["ipo", "simpo"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)