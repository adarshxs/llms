# Just LLMs and their moving parts

Work in progress. Reach out: @adarshxs on X/Twitter/LinkedIn
This is an old report I wrote for fun. I am trying best to keep it updated.

#### All working and moving parts of an LLM in its ecosystem from **training to inference deployment**. A detailed mechanism is later followed:

- Training
- Evaluation
- Inference

![data](https://i.imgur.com/GZPwPzQ.png)

## Awesome Blogs
- https://sweet-hall-e72.notion.site/Why-are-Modern-Neural-Nets-the-way-they-are-And-Hidden-Hypernetworks-6c7195709e7b4abbada921875a951c54
- https://astralord.github.io/posts/transformer-inference-optimization-toolset/
- https://arxiv.org/abs/1802.01528
- https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
- https://wangkuiyi.github.io/positional_encoding.html
- [a very cool blog on Rotary Position Embeddings](https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding)
- [Reinforcement Learning From Human Feedback - System, Mathematics and Code in TRL PPO](https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html)
- [fast llm inference from scratch](https://andrewkchan.dev/posts/yalm.html)
- https://carpedm30.notion.site/AI-Compiler-Study-2cc71f48eb1140d09a439ab0b10bdb7b

## Tokenizers
*A tokenizer refers to a tool or algorithm that breaks down text into smaller, meaningful units called tokens. These tokens can be individual words, subwords, or even characters, depending on the specific tokenization approach used.*

Here I'll start with collating all information about how tokenizers works, best practices, etc. 

Ofc I have to start with the legendary Karpathy tutorial: [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&pp=ygUTa2FycGF0aHkgdG9rZW5pemVycw%3D%3D)

Next is an excerpt from this amazing [document I found](https://docs.google.com/document/d/166xQsrcFmtTDp1okgy7iZTA7I3zs1S9KjFOGJmVY0uU/edit) on everything about tokenizers. A few points from the same for tokenizer choice in a general purpose conversational LLM aiming to cover non-english languages:
### Prefer

#### Framework
- HuggingFace Tokenizer 
- HuggingFace with special care for:
  - Whitespace

#### Vocabulary  
- Vocabulary Size
  - For Korean or single language:
    - 32k suffice? (UL2, T5, LLaMA)
  	- 50K for 1 lang + multilingual
  - For Multilingual : 256k total (PaLM, NLLB, BLOOM,mT5 etc)
  - (byte-level fallback ~ 1% of 32k)
  - Investigate Larger vocabularies(XLM series 512k, 1M)
    - Push more calculations to embed layers
  	- Linear increase at the last layer

#### Tokens
- Whitespace
  - _, __, ___ (up to 16)
- Unicode Exemplars
  - 现代汉语常用字表(simplified chinese 3500)
  - Joyo Kanj (Japanese 2136)
  - Includes most basic characters from all unicode supported languages
 - control characters(<FIM>, </FIM>) etc 200

- Tokenizer Training Data
  - As much as possible without harming low resource languages
  - Temperature sampling(alpha of 0.3 or 1 of 3.33), Unimax
- Digits and Punctuation
  - Individual vs hard coded(0-999)
  - GOAT supports individual digits tokenizations
    - Shown to be superior for numbers
- Individual Punctuation
  - Galactica, Minerva does this
- Unicode (UTF-8)
  - Use normalization
    - NKFC can be destructive for some languages
    - NFC is used in many datasets. Not supported in spn
- Algorithms
  - Unigram 
    - Unigram allows easier tokenizer modifications
    - Unigram provides Negative log probs
    - Can be superior with SR
	- Is NOT known to be superior without subword sampling
  - BPE
    - Widely used
- Take or train from pretrained tokenizer (FOSS)
  - XLM-V : 1M vocab spm
    - take vocab, new model
  - Flores-sacrebleu spm200
  - umT5(SPM 256k ulm)
  - NLLB200 :256k spm(bpe) (cc-by-nc)
  - Tiktoken (does not have training framework)
    - Cl100k, p50k, r50k 


## Training - Pre Training
The training process of LLMs involves four steps: data collection (preprocessing), model configuration, model training, and model evaluation. The data collection step involves collecting a training dataset, which can be derived from multiple sources like books, articles, website content, and open datasets. The data must next be cleansed and made training ready. The model configuration step involves selecting the architecture of the model, the number of layers, and the number of neurons per layer. The model training step involves iteratively adjusting parameter values until the model correctly predicts the next token from the previous sequence of input tokens. Finally, the model evaluation step involves benchmarking and evaluating the performance of the model on different hardware accelerators. LLMs are artificial neural networks following a transformer architecture, and they work by taking an input text and repeatedly predicting the next token or word.

Some great open source LLMs:
- https://huggingface.co/meta-llama/Llama-2-70b
- https://huggingface.co/Qwen/Qwen-72B
- https://huggingface.co/01-ai/Yi-34B
- https://huggingface.co/upstage/SOLAR-10.7B-v1.0
- https://huggingface.co/microsoft/phi-2
- https://huggingface.co/mistralai/Mistral-7B-v0.1
- https://huggingface.co/tiiuae/falcon-40b

### Fine tuning
Fine-tuning in LLMs refers to the process of adjusting the parameters of a pre-trained large language model to unlock its full potential in specific domains or tasks. This is done by further training the model on a smaller, domain-specific dataset. Fine-tuning enhances LLMs and allows them to specialize in a particular domain while retaining their general language understanding. It is helpful when you need to adapt your model to specific custom datasets or tasks. There are two main fine-tuning techniques for LLMs: repurposing and full fine-tuning. Repurposing is a technique where you use an LLM for a task that is different from the task it was originally trained on, while full fine-tuning involves creating a dataset of data that contains examples of the input and output for the task and training the LLM on this dataset using a supervised learning approach. 

![finetuning](https://i.imgur.com/BKYlvui.png)

The purpose of fine-tuning is to adapt the general capabilities of the model to specific applications or requirements. This could include specializing in a particular language, domain (like legal or medical texts), style (like casual conversation or technical writing), or task (like question-answering or summarization).

Fine tuning Frameworks-
- Axolotl - https://github.com/OpenAccess-AI-Collective/axolotl
- LLaMA-Factory - https://github.com/hiyouga/LLaMA-Factory
- HF autotrain advanced - https://github.com/huggingface/autotrain-advanced
- Ludwig - https://ludwig.ai/latest/user_guide/llms/finetuning/

### HARDWARE REQUIREMENTS for Fine Tuning (GPU VRAM): 
![reqs](https://i.imgur.com/iZe3qUa.png)

Llama-Factory: - Support:
![support](https://i.imgur.com/YOxwggl.png)

Fine tuning DATASET FORMATS:
![format](https://i.imgur.com/yCnXqFB.png)

### List of some famous instruction tuning format:
Stanford Alpaca (en)
Stanford Alpaca (zh)
GPT-4 Generated Data (en&zh)
Self-cognition (zh)
Open Assistant (multilingual)
ShareGPT (zh)
Guanaco Dataset (multilingual)
UltraChat (en)
LIMA (en)
OpenPlatypus (en)
CodeAlpaca 20k (en)
Alpaca CoT (multilingual)
OpenOrca (en)

# Some Thoughts on DATA
###### This section is shamelessly stolen from @georgejrjrjr's [google doc](https://docs.google.com/document/d/1tza0OIdTZNNjTqhkWZLRC9ha9Sp7lumGF5ytthx_Ozw/edit)

![data_harm](https://i.imgur.com/hGWfrGI.png)

From the paper : https://arxiv.org/abs/2206.14486 (Beyong Neural Scaling Laws)

They say: Ninety percent of the data is actively harmful to the performance of the model! As in, training is worse than useless, it causes artificial brain damage. With apologies for the repetition: the model depicted does not reach peak performance until and unless the least informative ~90% of the training set is omitted.
This was a tantalizing result, but this paper pertained to image models, and it wasn’t necessarily obvious to everyone that the results would transfer. One ‘gotcha’ from the authors’ data-pruning theory is that the usefulness of a data-pruning metric is bounded by its accuracy. Will we even find data-pruning metrics that meet the criteria for the performance gains theory predicts?


This wondering and waiting came to an end with “Textbooks are All You Need”, the Phi-1 paper. Here Microsoft used a simple data pruning metric (and a bit of synthetic data) to create a code model ~11x smaller, and ~200x cheaper to train than its nearest peer, StarCoder.

The reason why Falcon 180B sucked for its size since in terms of Performance:

Llama 2 70B ~ Falcon 180B

MISTRAL AI's Strategic Memo: https://drive.google.com/file/d/1gquqRqiT-2Be85p_5w0izGQGgHvVzncQ/view

Data quality is where the Open Source AI community can move the needle on model  performance.
Training models is expensive and glamorous, yet the gains are ephemeral: the lifespan of a state of the art model is measured in months at most.
- Datasets provide intelligence yields for years (e.g., The Pile) or more –check the publication dates on your evals!
- Training is hard to distribute; dataset distillation is embarrassingly parallel. We can literally just do it together.
- This document covers things we know work, but dataset distillation is still early. In coming years the line between “filtered data” and “synthetic data” may become very fuzzy, as data transformation becomes more of a thing.
- Publishing superlative datasets moves the bottleneck on training frontier models from ‘expertise limited’ (the pretext of Mistral’s incredible a-round), to ‘checkbook limited’
- ‘Checkbook limited’ makes it easy, and so more common, for wealthy interests to fund models the GPU poor consume.
- The easier a frontier model is to train, the more competitive the open source AI space will become.
- The most plausible path to AGI at home runs through dataset distillation.

tl;dr “it’s the data”


#### How to build those datasets: (cc @georgejrjrjr's idea):

An open source dataset distillation pipeline should probably begin with the text quality heuristics together.ai has calculated for Redpajama-Data-v2 (useful for selecting clean prose), [semantic de-duplication](https://arxiv.org/abs/2303.09540) (to reduce redundancy), and '[ssl prototype filtering](https://arxiv.org/abs/2206.14486)' a method to filter out uninformative documents as measured by distance to the nearest document cluster’s centroid. Meta has recently demonstrated strong results [combining them (‘D4’)](https://arxiv.org/abs/2308.12284), finding that semantic de-duplication and re-clustering are helpful for the downstream performance of ssl prototype filtering.

The other thread of work is domain weight optimization: let's say you have a training corpus, embed and cluster it, and call those clusters data domains. Methods like DoReMi and its (reportedly more stable) successor DoGE use small proxy models to estimate how many tokens it will take a larger model to learn a given domain. This avoids wasting precious compute and model capacity on uninformative (too easy) or noisy (too hard) domains.
Domain weighting methods are presumably composable with and complementary to semantic de-duplication and ssl prototype filtering. Domain weight determination answers the question, "How big should this subset of the dataset be", while SemDeDup and SSL Prototypes provide the means of trimming them to size.
[Shi et al](https://arxiv.org/abs/2310.10638) from researchers at Meta, UW, and AI2 found that grouping data into training batches of related content further increases model efficiency.

The Chinese labs had already taken note, too: Alibaba released a data distillation framework and accompanying paper, [Data Juicer](https://github.com/alibaba/data-juicer), which presumably fueled the data behind their Qwen model family.
![datajuicer](https://camo.githubusercontent.com/66b0e54c62b3eced843fef02c047b41a20c499ffe208d96a45553a400a82ec18/68747470733a2f2f696d672e616c6963646e2e636f6d2f696d6765787472612f69322f4f31434e3031494d506544313178595255594c6d584b4f5f2121363030303030303030363435352d322d7470732d333632302d313630342e706e67)
Also among the most powerful open weight large language models is DeepSeek-67B. It comes to us from a previously-obscure Chinese team. 


Build on the great work [together.ai](http://together.ai/) has delivered with Redpajama-Data-v2:
- Extract text from CommonCrawl
- De-duplicate (exact, document level)
- De-duplicate (exact, line-level)
- Identify the language
- De-duplicate (fuzzy w/ MinHashLSH, document level)
- Calculate quality heuristics


RP2 has ~20T tokens of English after de-duplication, annotated with 43 text quality heuristics. It is an incredible asset.


Downstream refinement:
- Train a classifier that takes the 43 text quality heuristics from Redpajama-Data-v2, and output p(good). There are lots of ways to approach this problem, but normalizing each heuristic (e.g. with t-digest, an efficient streaming percentile function) is probably a good place to start.
- Set a p(good) threshold. This determination is best made by training small proxy models.
- Embed & cluster documents.
- De-duplicate (semantic, document-level). Similarity threshold is probably around .1 or a little less from eyeballing the SemDeDup and D4 papers, but this is a hyperparameter in need of tuning.
- Re-cluster (which, per D4, prevents cluster distortion from the near-duplicates).
- For each cluster, how much of each domain should be kept to maximize the model’s generality? Smart domain weighting algorithms (e.g., [DoReMi](https://arxiv.org/abs/2305.10429) or [DoGE](https://arxiv.org/abs/2310.15393)) answer this question.
- Trim each domains to size with more heuristic, SemDeDup, and/or ssl prototype filtering –optimal mix TBD.
- Cluster documents into curricular training batches for more efficient training.




## LLM eval leaderboard : https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
![eval](https://i.imgur.com/uRmdMlV.png)

chatbot arena for comparing: https://chat.lmsys.org/
![compare](https://i.imgur.com/UQR8jvT.png)

Some popular fine tunes:
- https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
- https://huggingface.co/TheBloke/airoboros-13b-gpt4-1.4-SuperHOT-8K-GPTQ
- https://huggingface.co/Intel/neural-chat-7b-v3-1
- https://huggingface.co/openchat/openchat-3.5-1210

### Post Training - RLHF, Policy optimization techniques
reinforcement learning and human feedback into natural language processing (NLP) to train models based on human-provided feedback. RLHF learns a reward model for a certain task based on human feedback and then trains a policy to optimize this reward model. It is designed to help fine-tune model behavior and improve sample efficiency.

**RLHF becomes a very manual and laborious task best for crowdsourcing**, But we do have other methods also:
- Direct Preference Optimization (DPO) - https://huggingface.co/papers/2305.18290
- Identity Preference Optimisation (IPO) - https://huggingface.co/papers/2310.12036
- Kahneman-Taversky Optimisation (KTO) - https://github.com/ContextualAI/HALOs

Essentially human centered loss functions that take humans into the loop to preferably align LLMs. OpenAI uses RLHF heavily.
![image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pref_tuning/dpo.png)

![image of rlhf](https://i.imgur.com/INNwC9j.png)

## Evaluations

cool eval blogs: 
- https://ivanleo.com/blog/are-your-eval-improvements-just-pure-chance

How to run Evals you ask? Maybe use this: https://github.com/allenai/open-instruct/tree/main/eval
Recently, several comprehensive benchmarks have been released for the evaluation of LLMs. several widely used benchmarks, such as MMLU, BIG-bench,  HELM, and a series of human exam benchmarks exist to test an LLMs capabilities. 
```
MMLU Base/Fine-tuned/Specialized General Human exam/practice
BIG-bench Base/Fine-tuned/Specialized General Human annotation
HELM Base/Fine-tuned/Specialized General Benchmark collection
Open LLM Leaderboard Base/Fine-tuned/Specialized General Benchmark collection
AGIEval Base/Fine-tuned/Specialized General Human exam/practice
MMCU Base/Fine-tuned/Specialized General Human exam/practice
M3KE Base/Fine-tuned/Specialized General Human exam/practice
C-Eval Base/Fine-tuned/Specialized General Human exam/practice
Xiezhi Base/Fine-tuned/Specialized General Human exam/practice
OpenCompass Base/Fine-tuned/Specialized General Benchmark collection
Chain-of-Thought Hub Base/Fine-tuned General Benchmark collection
KoLA Base/Fine-tuned Knowledge utilization Web
ARB Fine-tuned Complex reasoning Human exam/practice
APIBench Base/Fine-tuned Tool manipulation Web
APIBank Fine-tuned Tool manipulation Synthesis
ToolAlpaca Base/Fine-tuned Tool manipulation Synthesis
T-Bench Fine-tuned Tool manipulation Synthesis
ToolBench Fine-tuned Tool manipulation Synthesis
BOLAA Base/Fine-tuned Environment interaction Benchmark collection
AgentBench Base/Fine-tuned Environment interaction Human annotation/Synthesis
HaluEval Base/Fine-tuned Human alignment Human annotation/Synthesis
PromptBench Base/Fine-tuned Robustness Benchmark collection
HumanEval Base/Fine-tuned/Specialized Code synthesis Human annotation
MultiMedQA Specialized Healthcare Benchmark collection
FLUE Specialized Finance Benchmark collection
LegalBench Specialized Legal Human annotation
Human Chatbot Arena Base/Fine-tuned/Specialized Human Alignment Human annotation
SciBench Fine-tuned Complex reasoning Human exam/practice Model
AlpacaEval Fine-tuned Instruction following Synthesis
MT-bench Fine-tuned Human alignment Human annotation
TrustGPT Base/Fine-tuned Human alignment Benchmark collection
LMExamQA Base/Fine-tuned Knowledge utilization Synthesis
ChatEval Base/Fine-tuned Knowledge utilization Benchmark collection
```

## Inference Stacks for Open Source LLMs
Running inference on these LLMs is a heavy task requiring GPUs for the same. SOTA LLM inference stacks include inference engines such as [vLLM](https://github.com/vllm-project/vllm) and [HF TGI](https://github.com/huggingface/text-generation-inference) along with various methods to optimize them for better performance and cost cutting - quantization, speculative decoding etc.
![inference](https://i.imgur.com/hVf3jm1.png)


From the paper: [Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward](https://arxiv.org/abs/2402.01799):
![inference_table](https://i.imgur.com/Lp4Mx4n.png)

- To scale inference workloads - open source - https://www.ray.io/
- Run LLMs, AI, and Batch jobs on any cloud. Get maximum savings, highest GPU availability, and managed execution: https://github.com/skypilot-org/skypilot
- competition  in hosting LLMs:
	- https://www.together.ai/
	- https://www.anyscale.com/
	- https://xylem.ai/

# More Details:

Large models have a large memory footprint both due to the trained model parameters as well as the transient state needed during decoding. The model parameters generally do not fit in the memory of a single accelerator chip, Hence a large total memory bandwidth required to meet a given latency target. Finally, inference cost from the attention mechanism scales quadratically with input sequence length.

We measure the inference cost in terms of the following metrics: **latency**, **throughput**, and **model FLOPS utilization**. The latency is the total time for an inference and can be broken down into the time to process the input tokens present at the start of the inference (which we call “prefill”) and the time to autoregressively generate output tokens (which we term “decode”). The decode latency can also be measured “per step”, i.e. divided by the number of tokens in each sequence. The throughput of prefill or decode is the number of *tokens generated per second*. The model FLOPS utilization (MFU) is the ratio of the observed throughput to the theoretical maximum throughput if the benchmarked hardware setup were operating at peak FLOPS with no memory or communication overhead.

We will comprehensively go over inference costs for 3 SOTA open source LLMs of private deployment:

- LLAMA 2 / Mistral / Mixtral
- Falcon
- MPT

Model | Inference GPU Required | Monthly Inference cost($)* | Commercial License for Fine tuned versions+
------ | ------ | ------ | ------
LLAMA 2 / Mistral  | LLAMA 2 70B: GPU 80 GB RAM (2x Nvidia A100) LLAMA 2 7B: GPU 15-20 GB RAM (1x Nvidia A10G) | A10G: <$1400(Due to available optimizations for inference including cpu inference and quantization thanks to [Georgi](http://ggml.ai/)). Therefore, 1 Dev, 1 Notebook, 150 hrs/month ~<**$287**  | ✅
FALCON | FALCON 40B-8bit: GPU 45 GB RAM (1x Nvidia A100) Falcon 7B: GPU 15-20 GB    | A10G: ~$1400; ~**$287**  | No Finetuned Version by the organization
MPT   | MPT 30B: GPU 80 GB RAM (1x Nvidia A100) MPT 7B: GPU 15-20 GB RAM (1x Nvidia A10G)   | A10G: ~$1400; ~**$287**  | ❌

*Calculated AWS pricing on the basis of https://fullstackdeeplearning.com/cloud-gpus/ and [Sagemaker](https://calculator.aws/#/addService/SageMaker)
+For certain fine tuned versions such as instruct fine tuned

---
Weighing in above factors and based on latency, throughput, and model FLOPS utilization, we rank the models as such:
**LLAMA 2 > FALCON > MPT**
---
![Throughput vs Latency](https://github.com/adarshxs/LLM-comparitive-analysis/blob/main/bafa3f2a-27b8-40b5-b815-4c5c9bd9270e.jpg?raw=true)
![Evals](https://github.com/adarshxs/LLM-comparitive-analysis/blob/main/25fe4ec5-eeea-4161-b687-d7bc84a41a6d.jpg?raw=true)

## Subjective comparison of the models:

### LLAMA 2
- Llama 2 is available for free (including commercial license).
- Llama 2 outperforms all other open-source models including Falcon and MPT, and has three variants including 7B, 13B, and 70B; the 70B variant achieves top performance across the board. (or rather its fine tuned versions took over the base version) [LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- Llama 2 is trained on 2 Trillion tokens, with 4 variants, ranging from 7-70B parameters.
- Llama is intended to be used in English, with almost 90% of the pre-training data being in English.
- The commercial license specifies a number of harmful use cases that violate the license, including spam!
- Ghost attention (GAtt) for multi-turn consistency after RLHF (Context Distillation to remember previous/initial instructions). Testing, alongside traditional safety, helpfulness, bias, and truthfulness analysis, also includes red teaming
-  Lower violation percentages and higher safety and helpfulness ratings compared to MPT, Vicuna, Falcon, PaLM, and ChatGPT.
- Better than other open-source models on MMLU (Massive Multitask Language Understanding), Q&A, Human-Eval and MBPP (code generation)
- Llama 2 is very comparable to ChatGPT 3.5 in most benchmarks (particularly, it beats ChatGPT in human evaluation on helpfulness: Win 36%; Tie 32%; - - Loss 32%) other than coding, looking at the data mix coding data is still quite small (classified under the - unknown language category)
- Benchmarks were done both on standardized ones (like MMLU) and head to head competition against other models, including PaLM-2 Bison and ChatGPT 3.5.
- A large portion of the paper focuses on RLHF improvements and objectives which is super neat.
- Model toxicity and evaluation is another large focus, including evaluations like red-teaming which were found in the Claude 2 model card. Generally Llama 2 performed very well with fewer safety violations than ChatGPT in human evaluations.
- The tokenizer is the same as Llama 1 which is interesting, but the context length is now 4k, double the original 2k!
- There’s both a regular and chat variation, as has been the trend in recent papers.
- Llama 2 (with fine tuning) offers better domain-specificity via fine-tuning at lower cost, and better guardrails.
- Llama 2 is trained on 40% more data than Llama 1 and performs well against benchmarks.

### Mistral (7B and Mixtral)
- An Apache 2.0 licensed model free for commercial use. 
- The state of the art 7B model &
- State of the art open source MOE model(7x8) 56B param model
- Fine tunes break eval leaderboards.


### MPT
- MPT-7B is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code. This model was trained by MosaicML.
- MPT-7B is part of the family of MosaicPretrainedTransformer (MPT) models, which use a modified transformer architecture optimized for efficient  training and inference.
- These architectural changes include performance-optimized layer implementations and the elimination of context length limits by replacing positional embeddings with Attention with Linear Biases (ALiBi). Thanks to these modifications, MPT models can be trained with high throughput efficiency and stable convergence. MPT models can also be served efficiently with both standard HuggingFace pipelines and NVIDIA’s FasterTransformer.
- MPT-7B is licensed for the possibility of commercial use.
- MPT-7B is trained on a large amount of data (1T tokens like LLaMA vs. 300B for Pythia, 300B for OpenLLaMA, and 800B for StableLM).
- MPT-7B is prepared to handle extremely long inputs, due to ALiBi (they finetuned MPT-7B-StoryWriter-65k+ on up to 65k inputs and can handle up to - 84k vs. 2k-4k for other open source models).
- MPT-7B is equipped with highly efficient open-source training code via the llm-foundry repository.

### FALCON
- Falcon-40B is a 40B parameters causal decoder-only model built by TII and trained on 1,000B tokens of RefinedWeb enhanced with curated corpora. It is made available under the TII Falcon LLM License.
- It features an architecture optimized for inference, with FlashAttention (Dao et al., 2022) and multiquery attention (Shazeer et al., 2019).
---
## MLops to deploy LLAMA 2 on personal infrastructure
#### Here I suppose we are deploying on a AWS gpu cloud. We will be using Hugging Face Text Generation Inference. Combo Langchain collab haha.

![HF TGI](https://user-images.githubusercontent.com/3841370/253379177-38ba1531-ea0d-4851-b31a-a6d4ddc944b0.png)
LLAMA 2 is a gated model so apply for access: https://ai.meta.com/resources/models-and-libraries/llama-downloads/
**Make sure you use the email id you use for Hugging Face to be able to download the model from HF(also available in HF format)**
The best way to use the HF TGI is to use their official docker container. So use that lol.
I will be following the steps particularly mentioned in the here: https://github.com/huggingface/text-generation-inference

Start by downloading the model and the HF TGI docker container. Just run the commands below it will pull the container from the ghcr registry automatically. **Make sure CUDA>=11.8**
You have the option to utilize the HUGGING_FACE_HUB_TOKEN environment variable for configuring the token employed by text-generation-inference. This allows you to use a private or gated model
```
Go to https://huggingface.co/settings/tokens
Copy your cli READ token

Export HUGGING_FACE_HUB_TOKEN=<your cli READ token> 
or
`huggingface-cli login` and then paste your token.
```
```
model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.1 --model-id $model
```
To start the tgi server:
```shell
text-generation-launcher --model-id  meta-llama/Llama-2-7b-chat-hf --num-shard 1 # Change the shard value according to the number of GPUs in your cluster
```
You can then query the model using either the `/generate` or `/generate_stream routes`:
```
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```
OR
`pip install text-generation`

```
from text_generation import Client

client = Client("http://127.0.0.1:8080")
print(client.generate("What is Deep Learning?", max_new_tokens=20).generated_text)

text = ""
for response in client.generate_stream("What is Deep Learning?", max_new_tokens=20):
    if not response.token.special:
        text += response.token.text
print(text)
```
The server is naturally hosted on `127.0.0.1:8080`. 

#### Onto Langchain integration:
Very Very ez: https://python.langchain.com/docs/integrations/llms/huggingface_textgen_inference
```python
from langchain.llms import HuggingFaceTextGenInference

llm = HuggingFaceTextGenInference(
    inference_server_url=""http://127.0.0.1:8080", # whatever url the inference server is hosted on
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)
llm("What did foo say about bar?")
```
---
## Pure Efficiency and cost cutting and metrics:

LLMs have revolutionized natural language processing tasks with remarkable success. However, their formidable size and computational demands present significant challenges for practical deployment, especially in resource-constrained environments. Addressing the imperative need for efficient deployment, we delve into various methodologies, encompassing quantization, pruning, knowledge distillation, and more. the [GPT-175B model](https://arxiv.org/abs/2005.14165), with an impressive 175 billion parameters, demands a minimum of 320GB (using multiples of 1024) of storage in half-precision (FP16) format. Furthermore, deploying this model for inference necessitates at least five A100 GPUs(!!!), each featuring 80GB of memory, to efficiently run inference upon.

We will start with inferencing and then go upon various methods of model compression.

**Sharding:** Model sharding is a technique used to distribute the weights of a large neural network model across multiple devices or storage mediums. This is particularly useful when the model is too large to fit into the memory of a single device, such as a GPU. By sharding the model, each device only needs to store a portion of the model's weights, allowing for parallel computation across multiple devices, allowing for training and inference with models that are too large to fit into the memory of a single device
- Memory Limitations: Modern deep learning models, especially those used in natural language processing like GPT-3 or BERT, can have billions of parameters. These models can be too large to fit into the memory of a single GPU or even a CPU.
- Sharding: To address this, the model's weights can be divided (or "sharded") into smaller chunks. Each chunk is then loaded onto a separate device.
- Parallel Computation: Once the model is sharded, forward and backward passes can be computed in parallel across the devices. Each device computes its portion of the model and then communicates with the other devices to aggregate the results.
- Challenges: While model sharding can alleviate memory constraints, it introduces challenges related to communication overhead between devices. Efficiently coordinating the devices and ensuring they work in harmony is crucial.
- Use Cases: Model sharding is especially useful in training very large models and serving them in production environments where you might want to distribute the model across multiple servers or GPUs for efficient inference.
- Complementary to Model Parallelism: Model sharding is often used in conjunction with model parallelism. While model sharding divides the model's weights across devices, model parallelism divides the computation of the model's layers or operations across devices.


## MODEL COMPRESSION:
There are various methods for compressing LLMs, including **pruning, knowledge distillation, quantization, Low-Rank Factorization**

![compression](https://github.com/adarshxs/LLM-comparitive-analysis/blob/main/edd8192c-c29a-484e-9324-faa52b727db1.jpg?raw=true)

WE WILL ONLY GO OVER QUANTIZATION. FOR PROD WE WILL USE [ONNX MODEL](https://learn.microsoft.com/en-us/windows/ai/windows-ml/get-onnx-model) format ONLY. 

**Quantization** has emerged as a widely embraced technique to alleviate the storage and computational overhead of deep learning models. While traditional representation employs floatingpoint numbers, quantization converts them to integers or other discrete forms. This transformation significantly reduces storage requirements and computational complexity. Although some precision loss is inherent, careful quantization techniques can achieve substantial model compression with only minimal accuracy degradation.

We will go about on how to implement one of the many Quantization techniques: ***8-bit Quantization with LLM.int8()***:

In a study conducted by [ Dettmers et al. (2022)](https://arxiv.org/abs/2208.07339), a technique called LLM.int8() was introduced as a solution to address outliers. This approach tackles the issue by utilizing a quantization method based on the maximum absolute value within vectors (absmax), and it incorporates mixed-precision quantization. In this context, outlier features are dealt with using a higher precision FP16 format to maintain their accuracy, while the remaining values are processed using a lower precision INT8 format. Given that outlier values typically make up only around 0.1% of the total values, this strategy effectively reduces the memory usage of the LLM by nearly half, achieving approximately 2x reduction.
LLM.int8() works by conducting matrix multiplication computation in three key steps:

- Extract columns from the input hidden states X containing outlier features using a custom threshold.
- Perform the matrix multiplication of the outliers using FP16 and the non-outliers using INT8 with vector-wise quantization (row-wise for the hidden state X and column-wise for the weight matrix W).
- Dequantize the non-outlier results (INT8 to FP16) and add them to the outlier results to get the full result in FP16.


We can easily use this technique thanks to the integration of the bitsandbytes library into the Hugging Face ecosystem. We just need to specify `load_in_8bit=True` when loading the model (it also requires a GPU).
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_int8 = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto',
                                             load_in_8bit=True,
                                             )
print(f"Model size: {model_int8.get_memory_footprint():,} bytes")
```
```
Model size: 176,527,896 bytes
```
With this extra line of code, we achieve an almost **3x smaller** model with a similar performance. We can even compare the distribution of the original and quantized weights
![llmint8](https://github.com/adarshxs/LLM-comparitive-analysis/blob/main/9e080645-a50d-482c-b4e6-c7379096aadc.jpg?raw=true)
![quantization](https://github.com/adarshxs/LLM-comparitive-analysis/blob/main/c33f15d6-73b3-4cb4-8fa3-d509efa780b5.jpg?raw=true)

We distinguish two main families of weight quantization techniques in the literature:

- Post-Training Quantization (PTQ) is a straightforward technique where the weights of an already trained model are converted to lower precision without necessitating any retraining. Although easy to implement, PTQ is associated with potential performance degradation.
- Quantization-Aware Training (QAT) incorporates the weight conversion process during the pre-training or fine-tuning stage, resulting in enhanced model performance. However, QAT is computationally expensive and demands representative training data.

Why you need a A10 or A100 GPU for production inference?
![bfloat](https://github.com/adarshxs/LLM-comparitive-analysis/blob/main/766e8e52-ebbb-486a-829d-94e2043c32a5.jpg?raw=true)
V100 architecture that I have here at VIT does not support bfloat16 precision format. Bfloat16 is only supported on the new >=ampere architectures. 

### Metrics:

Inference efficiency of LLMs can be measured using various metrics, which capture different aspects of performance. These metrics are commonly presented alongside accuracy and zero-shot ability to comprehensively evaluate the LLM.

***Number of Parameters:*** in a LLM refers to the total count of learnable weights or variables that the LLM needs to optimize during training.
In LLMs, parameters represent the weights in the connections between neurons or attention layers. In general, the more parameters a LLM has, the more expressive it can be, but it also requires more computational resources and memory for both training and inference.

***Model Size:*** typically refers to the disk space or memory footprint required to store the entire LLM, including weights, biases, and other necessary components. The model size is closely related to the number of parameters, as more parameters usually lead to a larger model size. However, other factors, like the data type used to represent the parameters and model architecture, can also influence the overall size.

***Compression Ratio:*** represents the ratio between the original size of the uncompressed LLM and the size of the compressed LLM. A higher compression ratio indicates a more efficient compression, as the LLM has been significantly reduced in size while preserving its functionality and performance.

***Inference time:*** measures the time taken by the LLM to process and generate responses for input data during inference or prediction. Inference time is particularly crucial for real-world applications where the LLM needs to respond to user queries or process large amounts of data in real-time.

***Floating point operations (FLOPs):*** measures the number of arithmetic operations involving floating-point numbers (typically 32-bit or
16-bit) that the LLM performs when processing input data. FLOPs provide a useful way to estimate the computationalrequirements of a LLM and compare the efficiency of different LLMs or compression techniques.

---
##### References:
[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)
[weight quant blog](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c)
[model compression paper](https://arxiv.org/abs/2308.07633)
[Inference optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
[scaling inference eff](https://arxiv.org/pdf/2211.05102.pdf)
[Pricing](https://fullstackdeeplearning.com/cloud-gpus/)
[Sagemaker](https://calculator.aws/#/addService/SageMaker)
[Lol I am biased to LLAMA2](https://aman.ai/primers/ai/LLM/#llama-2)
https://rentry.org/llm-training
[comparing-opensource-large-language-models](https://www.shakudo.io/blog/comparing-opensource-large-language-models)
[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245.pdf)

check
