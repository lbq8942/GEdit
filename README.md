# GEdit

This repository provides the official implementation and code for the paper **"Editing Large Language Models with Geometry-Aware Regularization"**.

GEdit is a geometry-aware method for knowledge editing in large language models. It protects the core subspace of pretrained knowledge and adjusts regularization according to the editability of each request.


## Method

### Geometry-aware regularization

For a pretrained FFN key vector $k$, the paper assumes the low-rank-plus-diagonal model

$$
k=b+Uz+\epsilon,
\qquad
z\sim\mathcal N(0,I),
\qquad
\epsilon\sim\mathcal N(0,D),
$$

where $b=\mathbb E[k]$, $U$ spans the core knowledge subspace, and $D$ is diagonal. Therefore,

$$
\mathrm{Cov}(k)=UU^\top+D,
\qquad
\bar D=bb^\top+D.
$$

Given the empirical covariance $S_0=Q\Lambda Q^\top$, GEdit estimates

$$
U=[q_1,\ldots,q_r]\Lambda_{1:r,1:r}^{1/2},
\qquad
D=\mathrm{diag}(S_0-UU^\top).
$$

The smallest rank $r$ retaining an energy ratio $\gamma$ is selected:

$$
r=\min\{j:
\frac{\sum_{i=1}^{j}\lambda_i}
     {\sum_{i=1}^{d}\lambda_i}
\geq\gamma\}.
$$

Let $M$ be an orthonormal basis for the complement of $U$, so $M^\top U=0$. For edit keys $K_1$, residual $R_1=V_1-WK_1$, and regularization strength $\mu$, the paper gives

$$
\Delta_G=
R_1K_1^\top M
[
M^\top(K_1K_1^\top+\mu\bar D)M
]^{-1}
M^\top.
$$

### Adaptive regularization

For a single edit key $k_e$, the paper defines subspace-aware editability as

$$
\mathcal E(k_e)=
\frac{1}{\mu_e}
k_e^\top M
(M^\top\bar D M)^{-1}
M^\top k_e.
$$

Setting the desired editability to $c>0$, namely $\mathcal E(k_e)=c$, gives

$$
\mu_e=
\frac{1}{c}
k_e^\top M
(M^\top\bar D M)^{-1}
M^\top k_e.
$$

Our code implementation extends this definition to a batch $K_e\in\mathbb R^{d\times b}$ by averaging the editabilities of its $b$ columns (batch size $b=1$ is its special case):

$$
\mu_e=
\frac{1}{cb}
\mathrm{tr}[
K_e^\top M
(M^\top\bar D M)^{-1}
M^\top K_e
].
$$

For sequential editing, let

$$
\bar K_e=M^\top K_e,\qquad
\bar D_M=M^\top\bar D M,\qquad
H_{e-1}=M^\top K_pK_p^\top M,
$$

where $K_p$ contains previous edit keys. This repository stores only the accumulated second moment $H_{e-1}$, not the individual historical keys. Its update is

$$
\Delta_e=
R_e\bar K_e^\top
(
\bar K_e\bar K_e^\top+
H_{e-1}+
\mu_e\bar D_M
)^{-1}
M^\top,
$$

followed by

$$
H_e=H_{e-1}+\bar K_e\bar K_e^\top.
$$

## Setup

Install dependencies:

~~~bash
pip install -r requirements.txt
~~~

Place dataset files under <code>data/raw/</code>. The default registry includes:

- <code>multi_counterfact_20877.json</code>
- <code>zsre_mend_eval_19086.json</code>

Local checkpoints can be placed in the sibling <code>llms/</code> directory using the paths in <code>main.py</code>. If a local checkpoint is absent, GEdit falls back to the Hugging Face model identifier. The first run computes or loads the pretrained key means and second moments under <code>cache/stats/</code>.

## Run

Example: 10,000 sequential CounterFact edits on LLaMA3-8B-Instruct:

~~~bash
python main.py \
  llms=llama3-8b \
  data=multi_counterfact_20877 \
  num_edits=10000 \
  gpu=0 \
  bs=1 \
  eval_every=10000 \
  save_name=gedit-10k
~~~

The command-line arguments mean:

- <code>llms</code>: model preset from <code>configs/llms/</code>, such as <code>gpt2-xl</code>, <code>gpt-j-6b</code>, or <code>llama3-8b</code>.
- <code>data</code>: dataset name registered in <code>data/adapters/registry.py</code>.
- <code>num_edits</code>: number of records taken in dataset order.
- <code>bs</code>: edits applied at each step. Use <code>1</code> for fully sequential editing.
- <code>eval_every</code>: evaluation interval measured in completed edits. The final state is always evaluated.
- <code>save_name</code>: label used in result and optional checkpoint filenames.
- <code>gpu</code>: CUDA device index visible to the process.

The two main method hyperparameters are:

- <code>algs.prior_rank</code> ($\gamma$): covariance energy retained by the protected core subspace. A larger value protects more principal directions and leaves a smaller editable subspace.
- <code>algs.c</code>: desired editability. Since $\mu_e\propto 1/c$, a larger <code>c</code> produces weaker regularization and a more aggressive edit.

Model-specific defaults are defined in <code>configs/algs/gedit.yaml</code>. They can be overridden explicitly:

~~~bash
python main.py \
  llms=gpt2-xl \
  data=multi_counterfact_20877 \
  num_edits=100 \
  bs=1 \
  algs.prior_rank=0.7 \
  algs.c=1.5
~~~

Computed statistics and resumable value caches are written to <code>cache/</code>; evaluation outputs are written to <code>results/</code>.
