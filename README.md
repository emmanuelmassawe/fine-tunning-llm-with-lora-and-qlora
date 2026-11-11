# fine-tunning-llm-with-lora-and-qlora

# Fine-tuning ya LLM kwa LoRA na QLoRA (Google Colab)

Mradi huu unaonyesha jinsi ya kufundisha (fine-tune) model ndogo ya lugha kwa kutumia **LoRA** na **QLoRA** kwenye Google Colab, hata kwa GPU ya bure ya T4.

## Yaliyomo

1. [Utangulizi](#utangulizi)
2. [LoRA dhidi ya QLoRA](#lora-dhidi-ya-qlora)
3. [Mahitaji](#mahitaji)
4. [Jinsi ya kuanza](#jinsi-ya-kuanza)
5. [Muundo wa notebook (cells)](#muundo-wa-notebook-cells)
6. [Mipangilio (CONFIG)](#mipangilio-config)
7. [Jinsi code inavyofanya kazi](#jinsi-code-inavyofanya-kazi)
8. [Kutumia dataset yako mwenyewe](#kutumia-dataset-yako-mwenyewe)
9. [Kutumia adapter baada ya training](#kutumia-adapter-baada-ya-training)
10. [Kutatua matatizo](#kutatua-matatizo)

---

## Utangulizi

Fine-tuning ni kuchukua model iliyokwisha fundishwa (hapa ni `Qwen/Qwen2.5-0.5B`) na kuifundisha zaidi kwa data maalum, ili ijifunze kazi au mtindo fulani. Haifundishwi kuanzia sifuri.

Kufundisha uzito wote wa model kunahitaji memory kubwa ya GPU. **LoRA** inaepuka hilo kwa kugandisha uzito wa model ya msingi na kufundisha matrices ndogo za ziada (adapter) pekee. Katika mradi huu, parameters zinazofundishwa ni takriban **1.75%** ya zote (8.8M kati ya 502M).

## LoRA dhidi ya QLoRA

Zote mbili zinafundisha adapter ndogo. Tofauti ni jinsi model ya msingi inavyohifadhiwa kwenye memory.

| | LoRA | QLoRA |
|---|---|---|
| Uzito wa model ya msingi | 16-bit | 4-bit (NF4, quantized) |
| Memory ya GPU | Kubwa zaidi | Chini sana |
| Kasi | Haraka kidogo | Polepole kwa takriban 20-30% |
| Ubora | Bora kidogo | Karibu sawa |
| Inahitaji | GPU au CPU | GPU ya NVIDIA (bitsandbytes) |

**Lini utumie ipi**

| Hali | Chaguo |
|---|---|
| Colab ya bure (T4) + model ya 7B | QLoRA |
| Colab ya bure (T4) + model ya 0.5B-1.5B | LoRA au QLoRA |
| GPU ya 24GB + model ya 7B | LoRA ikitosha, vinginevyo QLoRA |
| GPU ya 24GB + model ya 13B-30B | QLoRA |
| Hakuna GPU ya NVIDIA | LoRA |

Kanuni rahisi: anza na LoRA, na ukipata `CUDA out of memory`, badili kwenda QLoRA.

## Mahitaji

- Akaunti ya Google (Colab na Drive)
- GPU: *Runtime → Change runtime type → T4 GPU*
- Maktaba: `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`

## Jinsi ya kuanza

1. Fungua notebook mpya kwenye [colab.research.google.com](https://colab.research.google.com).
2. Washa GPU: *Runtime → Change runtime type → T4 GPU*.
3. Endesha cells moja baada ya nyingine kwa mpangilio ulio kwenye sehemu inayofuata.
4. Subiri hadi uone `Adapter imehifadhiwa kwenye: ...`.

> **Muhimu:** Colab ya bure inaweza kuanza upya session ukikaa bila kutumia. Variables zote hupotea (utaona `NameError`). Ikitokea, endesha cells zote upya kwa mpangilio. Adapter iliyohifadhiwa kwenye Drive haipotei.

## Muundo wa notebook (cells)

Endesha kwa mpangilio huu. Cells zenye `def` hazitoi output, hiyo ni kawaida.

| # | Cell | Kazi | Output |
|---|---|---|---|
| 1 | `!pip install ...` | Kusakinisha maktaba | Ndiyo (kidogo) |
| 2 | Imports | Kuleta maktaba | Hapana |
| 3 | GPU na Drive | Kuangalia GPU, kuunganisha Drive | Ndiyo |
| 4 | CONFIG | Mipangilio yote | Hapana |
| 5 | DTYPE | Kuchagua fp16 au bf16 | Hapana |
| 6 | `load_model_and_tokenizer` | Kupakia model | Hapana (ni function tu) |
| 7 | `add_lora` | Kuweka adapter | Hapana |
| 8 | `build_dataset` | Kuandaa data | Hapana |
| 9 | `train` | Training na kuhifadhi adapter | Hapana |
| 10 | `quick_test` | Kujaribu model | Hapana |
| 11 | `merge_adapter` na `main()` | Kuanzisha kila kitu | **Ndiyo, hapa ndipo output zote zinapoonekana** |

Cells 6 hadi 10 zinatengeneza functions tu. Hakuna kinachoendeshwa mpaka `main()` iitwe kwenye cell ya mwisho.

### Cell ya imports

```python
import gc
import os

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
```

### Cell ya DTYPE

```python
# T4 haina bf16 halisi, kwa hiyo tunaangalia uwezo wa GPU
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    DTYPE = torch.bfloat16   # Ampere na kuendelea (A100, L4, RTX 30xx)
else:
    DTYPE = torch.float16    # T4
```

## Mipangilio (CONFIG)

`OUTPUT_DIR` lazima iwe **ndani ya cell hii hii**, baada ya `MODE`.

```python
MODE = "qlora"                       # "lora" (16-bit) au "qlora" (4-bit)
MODEL_NAME = "Qwen/Qwen2.5-0.5B"     # model ndogo kwa majaribio
DATASET = "yahma/alpaca-cleaned"     # dataset ya Hugging Face
DATA_FILE = None                     # mfano: "data.json" (inapita DATASET)
MAX_SAMPLES = 1000
MAX_LENGTH = 512
EPOCHS = 1
LR = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 8
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MERGE_AFTER_TRAINING = False
OUTPUT_DIR = f"/content/drive/MyDrive/out-{MODE}"
```

| Mpangilio | Maana |
|---|---|
| `MODE` | `"lora"` (16-bit) au `"qlora"` (4-bit) |
| `MODEL_NAME` | Model ya msingi kutoka Hugging Face |
| `DATASET` | Dataset ya Hugging Face yenye safu `instruction`, `input`, `output` |
| `DATA_FILE` | Faili lako la JSON (linapita `DATASET` likiwekwa) |
| `MAX_SAMPLES` | Idadi ya mifano ya kutumia |
| `MAX_LENGTH` | Urefu wa juu wa maandishi (tokens) |
| `EPOCHS` | Mara ngapi model inapitia data nzima |
| `LR` | Learning rate |
| `BATCH_SIZE` | Mifano kwa kila hatua (punguza kuokoa memory) |
| `GRAD_ACCUM` | Hatua za kukusanya gradient kabla ya kusasisha. Batch halisi = `BATCH_SIZE × GRAD_ACCUM` |
| `LORA_R` | Rank ya adapter (kubwa = uwezo zaidi, memory zaidi) |
| `LORA_ALPHA` | Scaling ya adapter (scale = alpha / r) |
| `LORA_DROPOUT` | Dropout ili kupunguza overfitting |
| `MERGE_AFTER_TRAINING` | `True` kuunganisha adapter na model mwishoni |
| `OUTPUT_DIR` | Mahali pa kuhifadhi adapter (Drive) |

## Jinsi code inavyofanya kazi

### 1. Kupakia model (`load_model_and_tokenizer`)

Hapa ndipo **quantization** inapotokea kwa QLoRA.

- **QLoRA:** uzito wa model unabanwa hadi biti 4 kwa `BitsAndBytesConfig`:
  - `nf4`: aina ya quantization (NormalFloat4) inayofaa uzito wa neural network.
  - `bnb_4bit_use_double_quant=True`: inabana hata constants za quantization, kuokoa memory zaidi.
  - `bnb_4bit_compute_dtype`: hesabu zinafanyika kwa 16-bit ingawa uzito umehifadhiwa kwa 4-bit.
  - `prepare_model_for_kbit_training` inaandaa model ya 4-bit kwa training.
- **LoRA:** model inapakiwa kwa 16-bit, na gradient checkpointing inawashwa.

### 2. Kuweka adapter (`add_lora`)

Uzito wa model ya msingi unagandishwa. Kwenye kila layer ya linear (`target_modules="all-linear"`) tunaongeza matrices mbili ndogo, A na B, zenye rank `r`. Hizi pekee ndizo zinazofundishwa. Parameters za LoRA zinageuzwa kuwa fp32 ili kuepuka hitilafu za fp16.

### 3. Kuandaa data (`build_dataset`)

Kila mfano unageuzwa kuwa muundo huu, kisha unatokenizwa na kukatwa hadi `MAX_LENGTH`:

```
### Instruction:
<maagizo>

### Input:
<input, ikiwepo>

### Response:
<jibu>
```

### 4. Training (`train`)

Inatumia `Trainer` ya Hugging Face.

- `warmup_steps=5`: hatua za kupandisha learning rate polepole mwanzoni. (`warmup_ratio` haikubaliki kwenye matoleo mapya ya `transformers`.)
- `optim="paged_adamw_8bit"` kwa QLoRA: inapunguza matumizi ya memory.
- `gradient_checkpointing=True`: inabadilisha kasi kwa memory.
- Mwishoni, adapter na tokenizer vinahifadhiwa kwenye `OUTPUT_DIR`. Hii ni MB chache tu, si model nzima.

### 5. Kujaribu (`quick_test`)

Inaipa model swali na kuchapisha jibu, ili uone kama inafanya kazi.

### 6. Kuunganisha (`merge_adapter`, si lazima)

Inapakia model ya msingi kwa 16-bit, inaunganisha adapter ndani yake kwa `merge_and_unload()`, na kuhifadhi model kamili kwenye `OUTPUT_DIR-merged`. Haina haja ya `peft` ikitumika baadaye. Usiunganishe kwenye model ya 4-bit, kwa sababu usahihi unapotea.

## Kutumia dataset yako mwenyewe

Faili la JSON liwe na muundo huu:

```json
[
  {"instruction": "Eleza maana ya mvua", "input": "", "output": "Mvua ni maji yanayoanguka kutoka mawinguni..."},
  {"instruction": "Tafsiri kwa Kiingereza", "input": "Habari za asubuhi", "output": "Good morning"}
]
```

1. Upload `data.json` kwenye Colab (ikoni ya folder upande wa kushoto).
2. Kwenye CONFIG weka `DATA_FILE = "data.json"`.

Au tumia dataset ya Hugging Face yenye safu `instruction`, `input` (si lazima), `output`, kwa kubadilisha `DATASET`.

Ushauri: mifano 500-1000 safi inazidi mifano 10,000 mibovu. Ubora wa data ndio unaoamua ubora wa model.

## Kutumia adapter baada ya training

Adapter iko kwenye `OUTPUT_DIR` ndani ya Drive yako. Kuipakia baadaye (hata kwenye session mpya):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ADAPTER_DIR = "/content/drive/MyDrive/out-qlora"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

prompt = "### Instruction:\nEleza maana ya mvua.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Kutatua matatizo

| Kosa au tatizo | Sababu | Suluhisho |
|---|---|---|
| `IndentationError: unexpected indent` | Ulinakili sehemu ya code kutoka ndani ya function, ikiwa na nafasi za mwanzo | Nakili kuanzia mstari wa kwanza bila nafasi za mwanzoni |
| `NameError: name 'model' is not defined` | `model` iko ndani ya `main()` tu, au session ilianza upya | Hifadhi hufanyika ndani ya `train()`. Usiendeshe cell ya kuhifadhi peke yake |
| `NameError: name 'MODE' is not defined` | Session ya Colab ilianza upya | Endesha cells zote upya kwa mpangilio |
| `TypeError: ... unexpected keyword argument 'warmup_ratio'` | Toleo jipya la `transformers` | Tumia `warmup_steps=5` |
| `DTYPE=torch.bfloat16` kwenye T4 | `is_bf16_supported()` inaweza kurudisha `True` kwa emulation | Tumia ukaguzi wa `get_device_capability()` |
| `CUDA out of memory` | Memory ya GPU imejaa | Weka `BATCH_SIZE = 1`, punguza `MAX_LENGTH`, kisha *Runtime → Restart session* |
| Hakuna output baada ya kuendesha cells | Hakuna kinachoita `main()` | Ongeza cell yenye `main()` mwishoni na kuiendesha |
| Training inachukua muda mrefu | T4 ni polepole | Punguza `MAX_SAMPLES` (mfano 200) kwa jaribio la kwanza |
| `HF_TOKEN` warning | Unapakua bila kuingia Hugging Face | Ni onyo tu, si kosa. Kwa model za Llama/Gemma, ingia kwa `huggingface_hub.login()` |

## Leseni na shukrani

- Model: [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- Dataset: [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- Maktaba: Hugging Face `transformers`, `peft`, `datasets`, `bitsandbytes`
