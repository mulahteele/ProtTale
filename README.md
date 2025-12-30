\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{longtable}

\title{\textbf{ProtTale: Protein Function Generation with Pairwise Training}}
\author{}
\date{}

% ---------- Code style ----------
\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  columns=fullflexible,
  backgroundcolor=\color{gray!5}
}

\begin{document}
\maketitle

\section{Introduction}

ProtTale is a protein function generation framework trained with \textbf{pairwise supervision}.
Given a protein sequence, the model generates free-text functional descriptions by aligning
protein representations with textual function embeddings.

\section{Method Overview}

ProtTale learns a shared embedding space between proteins and function descriptions.
Given two proteins $P_i$ and $P_j$ with similarity score $s_{ij}$, the training objective enforces:

\begin{equation}
\mathcal{L}_{\text{pair}} =
\left\| \langle f_i, f_j \rangle - s_{ij} \right\|_2^2
\end{equation}

where $f_i$ and $f_j$ denote the learned function embeddings.

During inference, ProtTale performs autoregressive generation:

\begin{equation}
p(y \mid x) = \prod_{t=1}^{T} p(y_t \mid y_{<t}, x)
\end{equation}

where $x$ is the protein sequence and $y$ is the generated function description.

\section{Environment Setup}

\subsection{Installation}

\subsubsection{Create Conda Environment}

\begin{lstlisting}[language=bash]
conda env create -n ProtTale -f ProtT3_environment.yml
\end{lstlisting}

\subsubsection{Activate Environment}

\begin{lstlisting}[language=bash]
conda activate ProtTale
\end{lstlisting}

\subsubsection{Install Packages with Dependency Conflicts}

\begin{lstlisting}[language=bash]
pip install salesforce-lavis==1.0.2 --no-deps
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
pip install opendelta==0.3.2
\end{lstlisting}

\section{Checkpoints and Data}

\begin{itemize}
  \item \textbf{Pretrained checkpoints (OneDrive):} \\
  \url{[Link here]}
  \item \textbf{Sample testing data (OneDrive):} \\
  \url{[Link here]}
\end{itemize}

\section{Inference on New Proteins}

\subsection{Input JSON Format}

For inference, prepare a JSON file with the following format:

\begin{lstlisting}[language=json]
[
  [
    "MMRGFKQRLIKKTTGSSSSSSSKKKDKEKEKEKSSTTSSTSKKPASASSSSHGTTHSSASSTGSKSTTEKGKQSGSVPSQGKHHSSSTSKTKTATTPSSSSSSSRSSSVSRSGSSSTKKTSSRKGQEQSKQSQQPSQSQKQGSSSSSAAIMNPTPVLTVTKDDKSTSGEDHAHPTLLGAVSAVPSSPISNASGTAVSSDVENGNSNNNNMNINTSNTQDANHASSQSIDIPRSSHSFERLPTPTKLNPDTDLELIKTPQRHSSSRFEPSRYTPLTKLPNFNEVSPEERIPLFIAKVDQCNTMFDFNDPSFDIQGKEIKRSTLDELIEFLVTNRFTYTNEMYAHVVNMFKINLFRPIPPPVNPVGDIYDPDEDEPVNELAWPHMQAVYEFFLRFVESPDFNHQIAKQYIDQDFILKLLELFDSEDIRERDCLKTTLHRIYGKFLSLRSFIRRSMNNIFLQFIYETEKFNGVAELLEILGSIINGFALPLKEEHKVFLVRILIPLHKVRCLSLYHPQLAYCIVQFLEKDPLLTEEVVMGLLRYWPKINSTKEIMFLNEIEDIFEVIEPLEFIKVEVPLFVQLAKCISSPHFQVAEKVLSYWNNEYFLNLCIENAEVILPIIFPALYELTSQLELDTANGEDSISDPYMLVEQAINSGSWNRAIHAMAFKALKIFLETNPVLYENCNALYLSSVKETQQRKVQREENWSKLEEYVKNLRINNDKDQYTIKNPELRNSFNTASENNTLNEENENDCDSEIQ",
    "unknown function.",
    -1.0,
    ["GO:0000000"]
  ]
]
\end{lstlisting}

\subsection{Field Description}

\begin{longtable}{lll}
\toprule
Field & Type & Description \\
\midrule
Sequence & string & Protein amino acid sequence \\
Function & string & Placeholder text for inference \\
Score & float & Dummy value (e.g., -1.0) \\
GO terms & list & Optional placeholder GO term list \\
\bottomrule
\end{longtable}

\section{Run Inference}

Run the following command to generate protein function descriptions:

\begin{lstlisting}[language=bash]
torchrun --nproc_per_node=1 --master_port=29505 stage2.py \
  --devices 1 \
  --mode eval \
  --init_checkpoint [PATH_TO_CHECKPOINT] \
  --filename stage2_func_generation \
  --plm_model esmc_300m \
  --encoder_type auto \
  --num_query_token 2 \
  --plm_tune lora \
  --plm_lora_r 16 \
  --plm_lora_alpha 16 \
  --text_max_len 256 \
  --max_inference_len 256 \
  --llm_name facebook/galactica-6.7b \
  --llm_tune lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --batch_size 16 \
  --init_lr 1e-4 \
  --inference_batch_size 4 \
  --precision bf16-mixed \
  --num_workers 8 \
  --root [PATH_TO_TEST_DATA_ROOT] \
  --max_epochs 1 \
  --caption_eval_epoch 1 \
  --save_every_n_epochs 1 \
  --head generation
\end{lstlisting}

Example:
\begin{lstlisting}[language=bash]
--root ./data/SwissProtV3
\end{lstlisting}

\section{Notes}

\begin{itemize}
  \item This document focuses on inference using pretrained checkpoints.
  \item Training follows the same data format with pairwise supervision.
  \item Both PLM and LLM support LoRA-based fine-tuning.
\end{itemize}

\section{Citation}

If you use ProtTale in your research, please consider citing our work (paper coming soon).

\end{document}
