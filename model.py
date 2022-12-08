import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers import (T5TokenizerFast, AutoTokenizer, MT5ForConditionalGeneration) 
import torch
import torch.nn as nn


class T5PromptTuningMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        n_tokens: int = None,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        return model

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = super().get_input_embeddings().weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        self.soft_prompt = nn.Embedding(n_tokens, 1536)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = super().get_input_embeddings()(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] =None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        penalty_alpha: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]]=None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]]=None,
        logits_processor= None,
        renormalize_logits: Optional[bool] = None,
        stopping_criteria= None,
        constraints= None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[int, float]]=None,
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        **model_kwargs,
        ):

        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )


        # Drop most of the args for now
        return super().generate(max_length=max_length,
                                min_length=min_length,
                                do_sample=do_sample,
                                early_stopping=early_stopping,
                                num_beams=num_beams,
                                temperature=temperature,
                                penalty_alpha=penalty_alpha,
                                top_k=top_k,
                                top_p=top_p,
                                typical_p=typical_p,
                                repetition_penalty=repetition_penalty,
                                bad_words_ids=bad_words_ids,
                                force_words_ids=force_words_ids,
                                bos_token_id=bos_token_id,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id,
                                length_penalty=length_penalty,
                                no_repeat_ngram_size=no_repeat_ngram_size,
                                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                                num_return_sequences=num_return_sequences,
                                max_time=max_time,
                                max_new_tokens=max_new_tokens,
                                decoder_start_token_id=decoder_start_token_id,
                                use_cache=use_cache,
                                num_beam_groups=num_beam_groups,
                                diversity_penalty=diversity_penalty,
                                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                logits_processor=logits_processor,
                                renormalize_logits=renormalize_logits,
                                stopping_criteria=stopping_criteria,
                                constraints=constraints,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                output_scores=output_scores,
                                return_dict_in_generate=return_dict_in_generate,
                                forced_bos_token_id=forced_bos_token_id,
                                forced_eos_token_id=forced_eos_token_id,
                                remove_invalid_values=remove_invalid_values,
                                synced_gpus=synced_gpus,
                                exponential_decay_length_penalty=exponential_decay_length_penalty,
                                suppress_tokens=suppress_tokens,
                                begin_suppress_tokens=begin_suppress_tokens,
                                forced_decoder_ids=forced_decoder_ids,
                                **{"inputs_embeds": inputs_embeds},
                                #**{"encoder_outputs": inputs_embeds}, # **{"inputs_embeds": inputs_embeds}, for seq2seq
                                )

 

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class T5PromptTuningLM(T5PromptTuningMixin, MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
