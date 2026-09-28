# ClipCap implementation

This directory implements a compact image-captioning model inspired by
ClipCap. ClipCap connects a pretrained visual encoder to a pretrained causal
language model, allowing the language model to generate a caption conditioned
on an image without building a separate encoder-decoder architecture.

## How this implementation works

The model uses:

- **ChineseCLIP ViT-B/16** as the image encoder;
- a trainable two-layer MLP as the visual-prefix mapper;
- **Chinese GPT-2** as the autoregressive caption generator.

For an input image, ChineseCLIP produces one 512-dimensional normalized image
embedding. The mapper converts it into ten prefix-token embeddings, each with
the same 768-dimensional width as a GPT-2 word embedding:

```text
image
  -> frozen ChineseCLIP image encoder
  -> [512] image feature
  -> trainable MLP mapper
  -> [10, 768] visual-prefix tokens
  -> Chinese GPT-2
  -> generated caption
```

During training, the ten visual tokens are concatenated before the caption
token embeddings. Next-token cross-entropy is calculated only for valid
caption tokens; visual-prefix and padding positions do not become language
targets. In this project the optimizer updates both the mapper and GPT-2,
while ChineseCLIP is used only to extract image features.

## Datasets

Two training modes are supported:

- `demo` is the default and uses the original two-image, 38-caption dataset
  stored in `caption_image.pkl`;
- `flickr8k` loads 6,000 training images with five captions per image, giving
  approximately 30,000 image-caption pairs.

For Flickr8k, every unique image is encoded by ChineseCLIP once. The resulting
features are cached under `cache/`, and all five captions share the same image
embedding. Later runs reuse the cache.

## Main files

- `model.py`: visual-prefix mapper, GPT-2 integration, and CLIP feature helper;
- `clipcap_dataset.py`: demo/Flickr8k loading, caption tokenization, masks, and
  cached image embeddings;
- `train.py`: supervised caption training and `model.pt` checkpoint saving;
- `infer.py`: image encoding and autoregressive caption generation;
- `config.py`: local model paths, dimensions, datasets, and batch sizes;
- `submit-clipcap-train.sh` and `submit-clipcap-infer.sh`: A100 Slurm jobs.

## Server workflow

Train on Flickr8k and run inference only after training succeeds:

```bash
cd ./clipcap

TRAIN_JOB_ID=$(sbatch --parsable \
  submit-clipcap-train.sh \
  --dataset flickr8k \
  --epochs 20)

sbatch --dependency="afterok:${TRAIN_JOB_ID}" \
  submit-clipcap-infer.sh
```

Without `--dataset flickr8k`, training uses the small demonstration dataset.
The checkpoint is written to `model.pt`, and inference currently captions
`trump.jpeg` and `pokemon.jpeg` several times using stochastic sampling.

This is an educational implementation, not a reproduction of full-scale
ClipCap. Flickr8k is small, and its English captions are not an ideal match for
the Chinese GPT-2 tokenizer and language model. Better caption quality would
normally require a language-matched caption dataset/model, validation-based
checkpoint selection, and broader image-caption training data. See
`understand_clipcap.md` for additional technical details.
