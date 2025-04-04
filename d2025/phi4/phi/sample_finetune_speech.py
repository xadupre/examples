"""
finetune Phi-4-multimodal-instruct on an speech task

scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.46.1
accelerate==1.3.0
"""

import argparse
import json
import os
from pathlib import Path

import torch
import sacrebleu
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
)


INSTSRUCTION = {
    "en_zh-CN": "Translate the audio to Mandarin.",
    "en_id": "Translate the audio to Indonesian.",
    "en_sl": "Translate the audio to Slovenian.",
}
TOKENIZER = {
    "en_zh-CN": "zh",
    "en_ja": "ja-mecab",
}
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100
_TRAIN_SIZE = 50000
_EVAL_SIZE = 200

class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """Initialize the multiple token batch stopping criteria.

        Args:
            stop_tokens: Stop-tokens.
            batch_size: Batch size.

        """

        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)

class CoVoSTDataset(Dataset):
    def __init__(self, processor, data_dir, split, 
                 lang="en_zh-CN", rank=0, world_size=1):

        self.data = load_dataset("facebook/covost2", 
                           lang, 
                           data_dir=data_dir, 
                           split=split,
                           trust_remote_code=True
                           )
        self.training = "train" in split
        self.processor = processor
        self.instruction = INSTSRUCTION[lang]
        
        if world_size > 1:
            self.data = self.data.shard(world_size, rank) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        {'client_id': '0013037a1d45cc33460806cc3f8ecee9d536c45639ba4cbbf1564f1c051f53ff3c9f89ef2f1bf04badf55b3a2e7654c086f903681a7b6299616cff6f67598eff',
        'file': '{data_dir}/clips/common_voice_en_699711.mp3',
        'audio': {'path': '{data_dir}/clips/common_voice_en_699711.mp3',
        'array': array([-1.28056854e-09, -1.74622983e-09, -1.16415322e-10, ...,
                3.92560651e-10,  6.62794264e-10, -3.89536581e-09]),
        'sampling_rate': 16000},
        'sentence': '"She\'ll be all right."',
        'translation': '她会没事的。',
        'id': 'common_voice_en_699711'}
        """
        data = self.data[idx]
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + self.instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, audios=[(data["audio"]["array"], data["audio"]["sampling_rate"])], return_tensors='pt')
        
        answer = f"{data['translation']}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        if  self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
        }

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def covost_collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool)
        )

    try:
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=0)
        audio_attention_mask = (
            pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        print(e)
        print(input_ids_list)
        print(labels_list)
        raise
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_audio_embeds': input_audio_embeds,
            'audio_embed_sizes': audio_embed_sizes,
            'audio_attention_mask': audio_attention_mask,
            'input_mode': 2,  # speech mode
        }
    )



def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to('cuda')

    return model


@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1
):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()
    all_generated_texts = []
    all_labels = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=covost_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(f'cuda:{local_rank}')

    for inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc='running eval'
    ):
        stopping_criteria=StoppingCriteriaList([MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=inputs.input_ids.size(0))])
        inputs = inputs.to(f'cuda:{local_rank}')
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64,
            stopping_criteria=stopping_criteria,
        )

        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.size(0), -1)[:, 0]

        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        generated_text = [
            processor.decode(_pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        all_generated_texts.extend(generated_text)
        labels = [processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX) for _label_ids in inputs["labels"]]
        all_labels.extend(labels)

    all_generated_texts = gather_object(all_generated_texts)
    all_labels = gather_object(all_labels)
    
    if rank == 0:
        assert len(all_generated_texts) == len(all_labels)
        bleu = sacrebleu.corpus_bleu(all_generated_texts, [all_labels])
        print(bleu)
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'all_generated_texts': all_generated_texts,
                    'all_labels': all_labels,
                    'score': bleu.score,
                }
                json.dump(save_dict, f)

        return bleu.score
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-4-multimodal-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument(
        "--common_voice_dir",
        type=str,
        default="CommonVoice/EN",
        help="Unzipped Common Voice Audio dataset directory, refer to https://commonvoice.mozilla.org/en/datasets, version 4.0",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en_sl",
        help="Language pair for translation.",
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=32,
        help='Batch size per GPU (adjust this to fit in GPU memory)',
    )
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    args = parser.parse_args()

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )

    model.set_lora_adapter('speech')


    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    eval_dataset = CoVoSTDataset(processor,
                                 data_dir=args.common_voice_dir,
                                 split=f'test[:{_EVAL_SIZE}]',
                                 lang=args.lang,
                                 rank=rank,
                                 world_size=world_size)
    
    train_dataset = CoVoSTDataset(processor,
                                  data_dir=args.common_voice_dir,
                                  split=f'train[:{_TRAIN_SIZE}]',
                                  lang=args.lang)

    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // (num_gpus * args.batch_size_per_gpu)

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy='no',
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,  # for unused SigLIP layers
    )

    # eval before fine-tuning
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    score = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_before.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    if accelerator.is_main_process:
        print(f'BLEU Score before finetuning: {score}')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=covost_collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    # eval after fine-tuning (load saved checkpoint)
    # first try to clear GPU memory
    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()

    # reload the model for inference
    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'sdpa',
    ).to('cuda')

    score = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    if accelerator.is_main_process:
        print(f'BLEU Score after finetuning: {score}')


if __name__ == '__main__':
    main()
