%Version 5.0 March 2026 — 15-variant ablation (C-series trained-φ)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Single-file LaTeX manuscript — DriftingMol (ZINC250K Pipeline)  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass[pdflatex,sn-nature]{sn-jnl} % Nature Portfolio style

%%%% Standard Packages
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{listings}
\usepackage{url}
\usepackage[numbers,sort&compress]{natbib}
\graphicspath{{figures/}{docs/figures/}}

\raggedbottom

%%%% Theorem styles
\theoremstyle{thmstyleone}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}

\theoremstyle{thmstyletwo}
\newtheorem{example}{Example}
\newtheorem{remark}{Remark}

\theoremstyle{thmstylethree}
\newtheorem{definition}{Definition}

\begin{document}

\title[DriftingMol]{DriftingMol: Property-Conditional Molecular Generation via Single-Step Latent Drifting}

%% Authors
\author[1]{\fnm{Jiangjie} \sur{Qiu}}
\author[1]{\fnm{Yijun} \sur{Li}}
\author*[1]{\fnm{Xiaonan} \sur{Wang}}\email{wangxiaonan@tsinghua.edu.cn}

\affil[1]{\orgname{Beijing Key Laboratory of Artificial Intelligence for Advanced Chemical Engineering Materials, Department of Chemical Engineering, Tsinghua University},
\orgaddress{\city{Beijing}, \country{China}}}

\abstract{
We propose \textit{DriftingMol}, a two-stage framework for single-step (1-NFE) property-conditional molecular generation on SELFIES representations. The key innovation is a \textit{coupled decoder drift} mechanism: the frozen VAE decoder's intermediate representation serves as the drift feature space $\varphi$, through which gradients flow back to the DiT-based generator. SELFIES encoding guarantees 100\% chemical validity by construction. On ZINC250K, DriftingMol achieves $>$94\% uniqueness and Spearman $\rho(\text{QED})$ up to $0.510$ at guidance scale $\alpha{=}5.0$, extending to simultaneous 4-property conditioning with mean $\bar{\rho}$ up to $0.598$ under a fair no-binning protocol. A comprehensive ablation across 15 architectural variants reveals that coupled decoder drift is the core enabler of property control ($\rho = 0.493$ vs.\ $0.286$ for z-space drift), gradient flow through $\varphi$ is a necessary condition (blocking it collapses uniqueness to $<$1\%), and z-diversity regularization is essential for preventing mode collapse. Notably, even a well-trained external feature extractor (LatentMAE, $R^2_\text{QED} = 0.68$) achieves only $\rho = 0.329$---no better than a random frozen MLP ($\rho = 0.276$)---demonstrating that end-to-end gradient coupling through the decoder, not feature quality \textit{per se}, is the mechanism underlying property control.
}

\keywords{Molecular Generation, Drifting Models, Single-Step Generation, SELFIES, Latent Space, Classifier-Free Guidance, Property Control}

\maketitle

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Introduction}

De novo molecular design is crucial for drug discovery and materials science. The goal is to generate novel, valid molecular structures that exhibit desired pharmacological properties. However, molecular generation is challenging because the mapping from continuous latent spaces to valid discrete structures introduces information bottlenecks, and because controllable property steering demands that the generative model learn meaningful property-latent correspondences.

Existing methods span autoregressive models on SMILES strings~\cite{Gomez2018}, VAE-based approaches (JT-VAE~\cite{JT-VAE}), normalizing flows (MoFlow~\cite{MoFlow}), and diffusion models (DiGress~\cite{Digress}, GDSS~\cite{GDSS}). While diffusion models achieve the highest quality, they require hundreds of denoising steps. Graph-based continuous latent models offer faster inference but face a critical limitation: the argmax discretization required for graph decoding collapses continuous property signals, making fine-grained property control infeasible. Among string-based approaches, autoregressive SMILES models can achieve validity through rejection sampling but lack a continuous latent space for gradient-based property steering; recent latent optimization methods (LIMO~\cite{Eckmann2022LIMO}) operate in SMILES VAE latent spaces but require expensive per-molecule gradient descent rather than amortized single-pass generation. To our knowledge, single-step amortized generation with continuous property control on drug-like molecules remains underexplored.

Recently, Drifting Models~\cite{corso2024drifting} introduced a paradigm where a kernel-based drift field trains a generator to map noise to data in a single forward pass (1-NFE), eliminating iterative denoising. We propose \textbf{DriftingMol}, which applies this drifting paradigm to molecular generation via SELFIES~\cite{Krenn2020SELFIES}---a molecular string representation where every token sequence maps to a valid molecule by construction~\cite{Krenn2020SELFIES}. This eliminates the validity bottleneck and preserves continuous property modulation through decoding.

A key technical contribution is our \textit{coupled decoder drift} mechanism. Rather than training a separate feature extractor (e.g., SimCLR or Latent-MAE), we repurpose the frozen VAE decoder as the drift feature space $\varphi$. The decoder's 512-dimensional intermediate representation provides a semantically rich space in which the drift field operates, and---critically---gradients flow through $\varphi$ back to the generator, enabling the drift loss to directly shape the generator's latent mapping. Through a comprehensive ablation study on ZINC250K with 15 architectural variants, we demonstrate that this coupled decoder drift is the primary mechanism enabling property control, while alternatives such as z-space drift, decoupled drift, stop-gradient drift, or externally trained feature extractors either provide substantially weaker signals or fail catastrophically.

Our main contributions are:
\begin{itemize}
    \item A two-stage SELFIES-based latent drifting framework achieving $>$94\% uniqueness and Spearman $\rho(\text{QED})$ up to $0.510$ on ZINC250K with 1-NFE generation, where 100\% validity is guaranteed by SELFIES encoding.
    \item A \textbf{coupled decoder drift} mechanism that converts the frozen VAE decoder into a rich feature space for drifting, eliminating the need for separate feature extractor training.
    \item A comprehensive \textbf{15-variant ablation study} dissecting the contributions of drift space, gradient flow, temperature schedules, diversity regularization, and externally trained feature extractors.
    \item Demonstration that \textbf{gradient flow through the drift feature space} is a necessary condition for property control---stop-gradient variants collapse to near-zero uniqueness.
    \item Evidence that \textbf{end-to-end gradient coupling, not feature quality}, is the key mechanism: even a well-trained LatentMAE feature extractor ($R^2_\text{QED} = 0.68$) achieves only $\rho = 0.329$, no better than a random frozen MLP ($\rho = 0.276$), whereas the coupled decoder reaches $\rho = 0.493$.
    \item Extension to \textbf{simultaneous 4-property conditioning} (QED, SA Score, LogP, MolWt) achieving mean $\bar{\rho}$ up to $0.598$ under the fair no-binning protocol.
\end{itemize}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Related Work}

\textbf{Molecular Generation.}
Molecular generation methods include GAN-based (MolGAN~\cite{MolGAN}), VAE-based (JT-VAE~\cite{JT-VAE}, GraphVAE~\cite{GraphVAE}), normalizing flow (MoFlow~\cite{MoFlow}, GraphNVP~\cite{GraphNVP}), and diffusion models (DiGress~\cite{Digress}, GDSS~\cite{GDSS}). While diffusion achieves top quality, it requires ${\sim}1000$ denoising steps per molecule. Flow-based models enable 1-NFE generation but struggle with validity and property control on drug-like molecules.

\textbf{SELFIES for Molecular Generation.}
SELFIES (Self-Referencing Embedded Strings)~\cite{Krenn2020SELFIES} guarantee that \emph{every} string decodes to a valid molecule, eliminating the validity bottleneck of SMILES and graph-based representations. SELFIES have been used in VAE and generative settings~\cite{Krenn2022SELFIES}, but prior work has not combined SELFIES with drifting models or Classifier-Free Guidance for property-conditional generation.

\textbf{Generative Modeling via Drifting.}
Deng et al.~\cite{corso2024drifting} proposed Drifting Models for single-pass generation: a kernel-based drift field with multi-temperature softmax and bi-dimensional normalization trains a generator without iterative denoising. Originally demonstrated on image generation, we pioneer its application to molecular generation and introduce the coupled decoder drift mechanism.

\textbf{Classifier-Free Guidance (CFG).}
CFG~\cite{Ho2022CFG} enables conditional generation by interpolating between conditional and unconditional predictions. Several adaptations exist: standard two-pass CFG computes separate conditional and unconditional outputs, while training-integrated variants embed the guidance scale directly into the model. We adopt the latter approach, following the drifting framework formulation~\cite{corso2024drifting} with $\log$-space guidance weights on unconditional negative samples during training.

\textbf{Property-Conditional Molecular Generation.}
Conditional molecular generation has been explored via conditional VAEs~\cite{JT-VAE}, reinforcement learning~\cite{Zhou2019MolDQN}, latent optimization (LIMO~\cite{Eckmann2022LIMO}), fragment-based exploration (FREED~\cite{Yang2024FREED}), and out-of-distribution steering (MOOD~\cite{Lee2023MOOD}). Unlike RL-based and optimization-based methods that require per-molecule iterative computation, our approach achieves property control through the drift field itself in a single amortized forward pass.

\textbf{Conditional Flow Matching.}
Conditional flow matching~\cite{Lipman2023CFM} learns continuous-time velocity fields for sample transport, sharing the noise-to-data mapping philosophy with drifting models. However, flow matching typically requires multi-step ODE solving at inference, whereas drifting achieves generation in a single step (1-NFE).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Methodology}

\subsection*{Overview: Two-Stage Pipeline}

DriftingMol consists of two sequentially trained stages:
\begin{enumerate}
    \item A \textbf{SELFIES $\beta$-VAE} mapping molecules to a regularized continuous latent space $z \in \mathbb{R}^{256}$.
    \item A \textbf{DiT-based drifting generator} with coupled decoder drift and Classifier-Free Guidance.
\end{enumerate}
The VAE is trained first and frozen; the frozen VAE decoder then serves dual roles---it decodes generated latents to molecules \emph{and} provides the 512-dimensional feature space $\varphi$ for drift field computation. No separate feature extractor training is needed.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig1_main.pdf}
\caption{\textbf{DriftingMol overview and mechanism.} (a) Two-stage SELFIES $\beta$-VAE plus DiT drifting generator. (b) Drift-space configurations and gradient-flow pathways. (c) QED ablation tiers showing that coupled decoder-$\varphi$ drift separates from z-space, random-$\varphi$, trained-$\varphi$, and collapsed controls.}
\label{fig:main}
\end{figure}

\subsection*{Stage 1: SELFIES $\beta$-VAE}

The encoder is a Transformer encoder with attention pooling that maps SELFIES token sequences (vocab${} = 108$, max\_len${} = 64$) to Gaussian parameters $\mu, \sigma \in \mathbb{R}^{256}$. The decoder is a one-shot non-causal Transformer decoder that expands a latent code across sequence positions and predicts all SELFIES tokens in parallel. Training uses $\beta$-VAE loss:
\begin{equation}
    \mathcal{L}_\text{VAE} = \mathcal{L}_\text{recon} + \beta \cdot D_\text{KL}(q(z|x) \| \mathcal{N}(0, I))
\end{equation}
with $\beta = 0.01$. Since every SELFIES string maps to a valid molecule by construction~\cite{Krenn2020SELFIES}, $\text{Validity} = 100\%$ holds for \emph{any} latent vector $z$---this is a property of the representation, not the model. The trained VAE achieves 82.3\% exact reconstruction and 99.1\% token-level accuracy on held-out data.

After VAE training, latent codes for all ${\sim}250$K ZINC molecules are pre-computed and cached alongside molecular properties (QED, SA Score, LogP, MolWt).

\subsection*{Stage 2: Drifting Generator with Coupled Decoder Drift}

\textbf{Generator Architecture.}
The generator $f_\theta$ is a DiT (Diffusion Transformer~\cite{Peebles2023DiT}) adapted for single-step generation: 20.2M parameters, 8 Transformer layers ($d{=}384$, 8 heads), 16 learnable tokens, FiLM conditioning, and attention pooling to $\hat{z} \in \mathbb{R}^{256}$. It maps noise $\varepsilon \sim \mathcal{N}(0, I_{64})$ plus an optional property condition $c$ to a latent vector in a single forward pass.

\textbf{Coupled Decoder Drift ($\varphi$-space).}
The central mechanism of DriftingMol is the use of the frozen VAE decoder as the feature extractor $\varphi$ for the drift field. Given a generated latent $\hat{z}$, we pass it through the frozen decoder and extract the 512-dimensional intermediate representation $\varphi(\hat{z})$. The drift field operates in this $\varphi$-space:

\begin{equation}
    V_\tau = \sum_j W^\tau_{ij} \big(\varphi(z_j^\text{ref}) - \varphi(\hat{z}_i)\big)
\end{equation}
where $z_j^\text{ref}$ are reference data latents (positive or negative samples), and $W^\tau$ are bi-dimensional softmax attention weights following Deng et al.~\cite{corso2024drifting}:
\begin{equation}
    W^\tau = \sqrt{\text{softmax}_\text{row}\!\left(\frac{S}{\tau}\right) \cdot \text{softmax}_\text{col}\!\left(\frac{S}{\tau}\right)}
\end{equation}
with the kernel similarity $S_{ij} = -\|\varphi(\hat{z}_i) - \varphi(z_j^\text{ref})\| / d_\text{global}$ using L2 distance and global distance normalization.

\textbf{Key Property:} The decoder is frozen (no weight updates), but gradients flow through $\varphi$ via the chain rule:
\begin{equation}
    \frac{\partial \mathcal{L}_\text{drift}}{\partial \theta} = \frac{\partial \mathcal{L}_\text{drift}}{\partial \varphi} \cdot \frac{\partial \varphi}{\partial \hat{z}} \cdot \frac{\partial \hat{z}}{\partial \theta}
\end{equation}
This means the drift loss directly informs the generator about how to place latents in the decoder's semantic space. We term this ``coupled'' drift because the generator and feature space are linked through shared computation.

\textbf{Multi-Temperature Aggregation.}
The drift aggregates over temperatures $\tau \in \{0.5, 1.0, 2.0\}$ with per-$\tau$ normalization. Each $\lambda_\tau$ is pre-computed once from the training data before training begins, using the expected drift magnitude under a $p \approx q$ baseline:
\begin{equation}
    \lambda_\tau = \sqrt{\mathbb{E}\!\left[\frac{\|V_\tau\|^2}{D}\right]}, \quad
    V_\text{total} = \sum_{\tau \in \mathcal{T}} \frac{V_\tau}{\lambda_\tau}
\end{equation}

\textbf{Drift Loss.}
The generator is trained to move $\varphi(\hat{z})$ toward the drift target:
\begin{equation}
    \mathcal{L}_\text{drift} = \left\|\varphi(\hat{z}) - \text{sg}\big(\varphi(\hat{z}) + V_\text{total}\big)\right\|^2
\end{equation}
where $\text{sg}(\cdot)$ denotes stop-gradient applied to the {\em target} only. The prediction $\varphi(\hat{z})$ on the left retains gradients, so the loss backpropagates through $\varphi \to \hat{z} \to \theta$ (the generator parameters). The target $\varphi(\hat{z}) + V_\text{total}$ is treated as a fixed reference point for each training step.

\textbf{z-Diversity Regularization.}
To prevent mode collapse, we add a kNN-based repulsion regularizer in z-space:
\begin{equation}
    \mathcal{L}_\text{zdiv} = \frac{1}{K}\sum_{k=1}^{K} \max(0, m - \|z_i - z_{\text{nn}_k(i)}\|)
\end{equation}
where $K{=}5$ nearest neighbors and $m{=}3.0$ is the margin. The total loss is:
\begin{equation}
    \mathcal{L} = \lambda_\text{drift} \cdot \mathcal{L}_\text{drift} + \lambda_\text{zdiv} \cdot \mathcal{L}_\text{zdiv}
\end{equation}
with $\lambda_\text{drift} = 1.0$ and $\lambda_\text{zdiv} = 2.0$.

\subsection*{Classifier-Free Guidance (CFG)}

For conditional generation, we augment the generator with a condition embedding via FiLM layers, and embed the guidance scale $\alpha$ as an additional input via a learned projection. During training, molecules are binned by property quantile into $N_\text{bins} = 20$ groups. Each training step samples $N_\text{gen} = 32$ target conditions with $N_\text{pos} = 2048$ positive references (same property group, selected via hybrid kNN + binning) and negative references (different groups), plus $N_\text{unc} = 32$ unconditional negatives. The guidance scale $\alpha$ is sampled from a power-law distribution $p(\alpha) \propto \alpha^{-3}$ on $[1, 4]$ and used to weight the unconditional negatives in the drift field via $\log$-space logit adjustment:
\begin{equation}
    w = \frac{(\alpha - 1)(N_\text{neg} - 1)}{N_\text{unc}}, \quad
    \ell_\text{unc} = -d_\text{unc}/\tau + \log w
\end{equation}
where $d_\text{unc}$ are distances to unconditional negative samples. With probability $p_\text{uncond} = 0.1$, all conditions are replaced with a learned null token $\varnothing$.

\textbf{Inference.} Unlike standard two-pass CFG ($z = z_\text{uncond} + \alpha \cdot (z_\text{cond} - z_\text{uncond})$), our generator is trained to directly respond to embedded $\alpha$, enabling \emph{single-pass} conditional generation:
\begin{equation}
    \hat{z} = f_\theta(\varepsilon, c, \alpha)
\end{equation}
where $\alpha = 1$ recovers the base conditional model and $\alpha > 1$ amplifies property steering. This single-pass design avoids the cost of two forward passes and is consistent with how $\alpha$ modulates the drift field during training. A two-pass variant is also available but not used in the reported results.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Experiments}

\subsection*{Setup}

\textbf{Dataset.} ZINC250K~\cite{Irwin2012ZINC} (${\sim}250$K drug-like molecules). Properties: QED (drug-likeness), SA Score (synthetic accessibility), LogP (lipophilicity), MolWt (molecular weight). Reference statistics: QED $= 0.732 \pm 0.139$, MolWt $= 332.1 \pm 61.9$, LogP $= 2.46 \pm 1.43$, SA $= 3.05 \pm 0.84$.

\textbf{Metrics.} Validity (V), Uniqueness (U = unique/valid), Novelty (N = not in train set), VUN $=$ V$\times$U$\times$N. Spearman $\rho$ (rank correlation of target vs.\ actual property), MAE, slope (OLS regression from target to actual, ideal $= 1.0$). Internal diversity (IntDiv $= 1 - $ avg pairwise Tanimoto on Morgan fingerprints), scaffold diversity (ScafDiv = unique generic Murcko scaffolds / n\_valid). Note that V $= 100\%$ for all variants by construction (SELFIES); we report it for completeness but it does not differentiate methods.

\textbf{Evaluation Protocol.} Each variant is trained for 300 epochs on a single RTX 4090D. The best checkpoint is selected by a quality gate (VUN score). Final evaluation generates 10,000 molecules at each guidance scale $\alpha \in \{1.0, 1.5, 2.0, 3.0, 5.0\}$; note that $\alpha$ is trained on $[1, 4]$, so $\alpha{=}5.0$ represents moderate out-of-distribution extrapolation. For single-property conditional evaluation, target values are drawn from the training quantile distribution. For multi-property evaluation, 9 quantile bins per property define target values ($\sim$1,100 molecules per bin).

\subsection*{Ablation Study Design}

We evaluate 15 variants to dissect the mechanism of property control. Table~\ref{tab:variants} summarizes the design. All variants share the same architecture, dataset, and training protocol---only the ablated component differs.

\begin{table}[htbp]
\centering
\caption{\textbf{Ablation Variants.} All share the same DiT generator (20.2M params) and training protocol. Only the drift mechanism differs. The C-series uses externally trained LatentMAE feature extractors (plain: $R^2_\text{QED} = 0.52$; property-enhanced: $R^2_\text{QED} = 0.68$).}
\label{tab:variants}
\begin{tabular}{l l l}
\toprule
Variant & Category & Key ablation \\
\midrule
\textbf{Full} & Proposed & Reference: coupled decoder-$\varphi$ drift, multi-$\tau$, z-div \\
Single-$\tau$ & Proposed & $\tau = \{1.0\}$ only (vs.\ $\{0.5, 1.0, 2.0\}$) \\
No-Div & Proposed & $\lambda_\text{zdiv} = 0$ (no z-diversity regularization) \\
\midrule
Z-Drift & Feature space & No decoder $\varphi$; drift in z-space directly (256D) \\
Rand-$\varphi$ & Feature space & $\varphi =$ frozen random MLP (512D) \\
Stop-Grad & Gradient flow & Drift gradients blocked from generator \\
Decouple & Gradient flow & $\varphi$-oracle weights + z-gradient (decoupled) \\
Decouple+Z & Gradient flow & Decouple + z-space drift \\
\midrule
MLP-Head & Non-drift & No drift ($\lambda_\text{drift}=0$), MLP property regressor \\
Ridge-Head & Non-drift & No drift ($\lambda_\text{drift}=0$), ridge on $\varphi$ \\
\midrule
Trained-$\varphi$ & Trained $\varphi$ & LatentMAE plain as $\varphi$ (drift only) \\
Trained-$\varphi$-P & Trained $\varphi$ & LatentMAE prop-enhanced as $\varphi$ (drift only) \\
Trained-$\varphi$+Z & Trained $\varphi$ & LatentMAE plain + z-drift \\
$\varphi$+Dec & Trained $\varphi$ & LatentMAE plain + coupled decoder-$\varphi$ \\
$\varphi$-P+Dec & Trained $\varphi$ & LatentMAE prop + coupled decoder-$\varphi$ \\
\bottomrule
\end{tabular}
\end{table}

Note: MLP-Head and Ridge-Head use property prediction heads instead of drift ($\lambda_\text{drift} = 0$, $\lambda_\text{prop} = 1$). The C-series (Trained-$\varphi$ variants) tests the original Drifting Models recipe~\cite{corso2024drifting} of training a separate feature extractor, with five sub-variants probing the effect of $\varphi$ quality (plain vs.\ property-enhanced LatentMAE), auxiliary z-drift, and combining trained $\varphi$ with the decoder feature space.

\subsection*{QED-Conditional Generation}

Table~\ref{tab:qed_main} presents the main conditional generation results. Each experiment generates 10,000 molecules per $\alpha$ value; we report the best $\alpha$ for each variant. Only drift-based experiments ($\lambda_\text{prop} = 0$) are directly comparable.

\begin{table}[htbp]
\centering
\caption{\textbf{QED-Conditional Generation on ZINC250K.} All 15 variants ranked by Spearman $\rho$ at best $\alpha$. All variants achieve V $= 100$\%. Horizontal lines separate tier boundaries.}
\label{tab:qed_main}
\begin{tabular}{l l c c c c c c c}
\toprule
Variant & Drift Space & $\alpha$ & $\rho$ $\uparrow$ & slope & MAE $\downarrow$ & U (\%) & IntDiv & ScafDiv \\
\midrule
No-Div & Decoder-$\varphi$ & 5.0 & \textbf{0.510} & 0.828 & \textbf{0.194} & 74.1 & 0.892 & 0.459 \\
Single-$\tau$ & Decoder-$\varphi$ & 5.0 & 0.500 & 0.793 & 0.204 & 94.1 & 0.895 & 0.581 \\
\textbf{Full} & \textbf{Decoder-$\varphi$} & \textbf{5.0} & \textbf{0.493} & \textbf{0.796} & 0.200 & \textbf{94.7} & 0.894 & \textbf{0.588} \\
\midrule
MLP-Head$^\dagger$ & (no drift) & 1.0 & 0.449 & 0.495 & 0.393 & 97.3 & 0.876 & 0.790 \\
\midrule
Trained-$\varphi$-P & LatentMAE-prop & 5.0 & 0.329 & 0.499 & 0.259 & 94.8 & 0.892 & 0.483 \\
Trained-$\varphi$+Z & LatentMAE + z & 5.0 & 0.312 & 0.499 & 0.244 & 97.2 & 0.911 & 0.473 \\
Trained-$\varphi$ & LatentMAE-plain & 5.0 & 0.297 & 0.472 & 0.249 & 96.3 & 0.925 & 0.377 \\
$\varphi$-P+Dec & LatentMAE-prop + dec & 5.0 & 0.296 & 0.449 & 0.202 & 97.8 & 0.894 & 0.634 \\
Z-Drift & z-space (256D) & 5.0 & 0.286 & 0.461 & 0.245 & 98.5 & 0.913 & 0.574 \\
$\varphi$+Dec & LatentMAE + dec & 3.0 & 0.281 & 0.432 & 0.200 & 98.8 & 0.898 & 0.648 \\
Rand-$\varphi$ & Random MLP (512D) & 5.0 & 0.276 & 0.433 & 0.234 & 97.9 & 0.912 & 0.468 \\
\midrule
Decouple & Detached $\varphi$ & 3.0 & 0.031 & --- & 0.274 & 0.3 & 0.609 & $\approx$0 \\
Decouple+Z & Detached $\varphi$ + z & 3.0 & 0.019 & --- & 0.273 & 0.3 & 0.583 & $\approx$0 \\
Stop-Grad & Decoder-$\varphi$ (sg) & 3.0 & 0.011 & --- & 0.272 & 0.3 & 0.604 & $\approx$0 \\
Ridge-Head$^\dagger$ & (no drift) & 3.0 & 0.009 & --- & 0.273 & 0.3 & 0.607 & $\approx$0 \\
\bottomrule
\multicolumn{9}{l}{\footnotesize $^\dagger$ Non-drift baseline (property prediction head, $\lambda_\text{drift}=0$).} \\
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig2_qed_ablation.pdf}
\caption{\textbf{QED ablation control-diversity profile.} Bars show best-$\alpha$ Spearman $\rho$ and points show uniqueness for the 15 core variants. Decoder-$\varphi$ variants form the top control tier, while decoupled and stop-gradient controls collapse in uniqueness.}
\label{fig:qed_ablation}
\end{figure}

To separate single-run rankings from reproducibility, Figure~\ref{fig:qed_seed_ci}
and Table~\ref{tab:qed_3seed} summarize the three-seed QED aggregate for the
key variants. The final aggregate uses the canonical seed-42 run plus
publication replicate seeds 43--44.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.72\textwidth]{fig4_qed_seed_ci.pdf}
\caption{\textbf{Three-seed QED confidence intervals.} Bars show mean Spearman $\rho$ across seeds 42--44 for the main QED variants; error bars are two-sided 95\% Student-$t$ confidence intervals. Seed 42 uses the canonical final run and seeds 43--44 use the publication replicate runs.}
\label{fig:qed_seed_ci}
\end{figure}

\begin{table}[htbp]
\centering
\caption{\textbf{Three-seed QED aggregate for key variants.} Means and two-sided 95\% Student-$t$ confidence intervals are computed from the canonical seed-42 run plus publication replicate seeds 43--44.}
\label{tab:qed_3seed}
\IfFileExists{results/tables/tab_qed_3seed.tex}
{\input{results/tables/tab_qed_3seed.tex}}
{\input{../results/tables/tab_qed_3seed.tex}}
\end{table}

Four clear tiers emerge:
\begin{itemize}
    \item \textbf{Tier 1} ($\rho \geq 0.49$): Full, Single-$\tau$, No-Div---all use coupled decoder drift with gradient flow. No-Div achieves the highest $\rho{=}0.510$ but with severely degraded uniqueness (74.1\%).
    \item \textbf{Tier 2} ($\rho \approx 0.28$--$0.33$): Trained-$\varphi$ variants (C-series), Z-Drift, Rand-$\varphi$---whether using a well-trained LatentMAE ($R^2_\text{QED}{=}0.68$), a random MLP, or no $\varphi$ at all, these variants cluster tightly, indicating that without end-to-end gradient coupling, the quality of $\varphi$ is largely irrelevant.
    \item \textbf{Tier 3} (collapsed): Decouple, Decouple+Z, Stop-Grad, Ridge-Head---near-zero uniqueness, no property control.
    \item \textbf{Non-drift reference}: MLP-Head ($\rho = 0.449$) uses direct property regression rather than drift. Its placement between Tiers 1 and 2 provides a useful calibration point.
\end{itemize}

\subsection*{Per-Target QED Steering}

Table~\ref{tab:pertarget} shows the actual QED values for selected target bins at $\alpha{=}5.0$. The Full variant demonstrates monotonic actual QED increase from 0.341 to 0.673 as the target increases from 0.39 to 0.92, with an actual spread of 0.332 (vs.\ target range 0.53). This corresponds to ${\sim}63\%$ of the ideal spread, indicating meaningful but not perfect property control.

\begin{table}[htbp]
\centering
\caption{\textbf{Per-Target QED Steering} ($\alpha{=}5.0$, 500 molecules per bin). Actual QED mean for selected target quantiles. ``Spread'' = max(actual) $-$ min(actual).}
\label{tab:pertarget}
\begin{tabular}{l c c c c c c}
\toprule
Target QED & 0.39 & 0.66 & 0.77 & 0.85 & 0.92 & Spread \\
\midrule
Full & 0.341 & 0.455 & 0.649 & 0.654 & 0.673 & 0.332 \\
Single-$\tau$ & 0.333 & 0.456 & 0.623 & 0.640 & 0.676 & 0.342 \\
No-Div & 0.352 & 0.460 & 0.629 & 0.678 & 0.695 & 0.342 \\
Z-Drift & 0.330 & 0.499 & 0.548 & 0.561 & 0.584 & 0.254 \\
Rand-$\varphi$ & 0.343 & 0.501 & 0.550 & 0.567 & 0.577 & 0.234 \\
\bottomrule
\end{tabular}
\end{table}

Decoder-$\varphi$ variants (Full, Single-$\tau$, No-Div) achieve actual spreads of 0.33--0.34, while z-drift (Z-Drift) and random-$\varphi$ (Rand-$\varphi$) achieve only 0.23--0.25. This confirms that the semantic structure of the decoder's intermediate representation---not merely the presence of a drift field---drives effective property steering. In addition to rank correlation, we estimate Success@$\delta$ (fraction of molecules within $\pm\delta$ of target) using a Gaussian approximation from per-bin statistics. At $\delta{=}0.10$, Full achieves S@0.10 $= 0.274$ vs.\ Z-Drift's $0.216$ for QED; detailed results across all properties are in Appendix~\ref{appD}.

\subsection*{CFG Guidance Scale Trade-off}

Table~\ref{tab:alpha_sweep} shows how $\rho$ and uniqueness trade off as guidance scale $\alpha$ increases for the Full variant.

\begin{table}[htbp]
\centering
\caption{\textbf{CFG Guidance Scale Trade-off} for DriftingMol-Full (QED conditional). Higher $\alpha$ increases property control at the cost of uniqueness.}
\label{tab:alpha_sweep}
\begin{tabular}{c c c c c c c}
\toprule
$\alpha$ & $\rho$ $\uparrow$ & MAE $\downarrow$ & slope & U (\%) & VUN & IntDiv \\
\midrule
1.0 & 0.317 & 0.187 & 0.476 & 96.6 & 0.966 & 0.890 \\
1.5 & 0.401 & 0.181 & 0.646 & 95.9 & 0.959 & 0.894 \\
2.0 & 0.438 & 0.182 & 0.695 & 96.3 & 0.963 & 0.896 \\
3.0 & 0.473 & 0.187 & 0.763 & 95.3 & 0.953 & 0.893 \\
5.0 & 0.493 & 0.200 & 0.796 & 94.7 & 0.947 & 0.894 \\
\bottomrule
\end{tabular}
\end{table}

Uniqueness degrades gracefully from 96.6\% ($\alpha{=}1.0$) to 94.7\% ($\alpha{=}5.0$), while $\rho$ improves from 0.317 to 0.493. Internal diversity remains stable ($0.890$--$0.896$), indicating that stronger guidance compresses mode coverage slightly but does not compromise molecular diversity.

\subsection*{Unconditional Generation}

Table~\ref{tab:uncond} reports unconditional generation quality. This measures pure generation ability independent of property control.

\begin{table}[htbp]
\centering
\caption{\textbf{Unconditional Generation on ZINC250K.} Best $\alpha$ per variant (by VUN). All achieve V $= 100$\%, N $\geq 99.9$\%. Reference: QED $= 0.732$, MolWt $= 332.1$.}
\label{tab:uncond}
\begin{tabular}{l c c c c c c}
\toprule
Variant & $\alpha$ & U (\%) $\uparrow$ & VUN $\uparrow$ & QED & LogP & IntDiv \\
\midrule
$\varphi$+Dec & 1.0 & \textbf{98.7} & \textbf{0.987} & 0.567 & 2.62 & 0.896 \\
$\varphi$-P+Dec & 1.0 & 98.4 & 0.984 & 0.561 & 2.67 & 0.897 \\
Z-Drift & 1.0 & 98.2 & 0.982 & 0.536 & 2.64 & \textbf{0.905} \\
Trained-$\varphi$+Z & 1.0 & 98.1 & 0.980 & 0.539 & 2.64 & 0.901 \\
Trained-$\varphi$ & 1.0 & 97.4 & 0.974 & 0.501 & 2.45 & 0.911 \\
Stop-Grad & 1.0 & 97.0 & 0.970 & 0.559 & 3.10 & 0.904 \\
Rand-$\varphi$ & 1.0 & 96.8 & 0.968 & 0.544 & 2.60 & 0.903 \\
Trained-$\varphi$-P & 1.0 & 96.5 & 0.965 & 0.524 & 2.63 & 0.904 \\
\textbf{Full} & \textbf{1.0} & 96.1 & 0.960 & \textbf{0.588} & 2.70 & 0.889 \\
Single-$\tau$ & 1.0 & 95.5 & 0.955 & 0.583 & 2.75 & 0.887 \\
MLP-Head & 1.0 & 94.9 & 0.949 & 0.493 & 3.69 & 0.892 \\
\midrule
No-Div & 1.0 & 7.4 & 0.074 & 0.630 & 3.27 & 0.846 \\
Decouple & 1.0 & 0.9 & 0.009 & 0.483 & 2.58 & 0.693 \\
Decouple+Z & 1.0 & 1.1 & 0.011 & 0.484 & 2.58 & 0.690 \\
\bottomrule
\end{tabular}
\end{table}

Notably, No-Div collapses to 7.4\% uniqueness unconditionally despite achieving the highest $\rho$ conditionally (Table~\ref{tab:qed_main}). Decouple/Decouple+Z generate mostly tiny molecules that collapse to $<$2\% uniqueness. The Full variant achieves 96.1\% uniqueness with distributional statistics closest to ZINC (QED $= 0.588$). Among the trained-$\varphi$ variants, $\varphi$+Dec and $\varphi$-P+Dec achieve the highest unconditional quality (U $\geq 98.4\%$), showing that the decoder $\varphi$ component contributes to unconditional generation even when combined with LatentMAE---though this advantage does not transfer to conditional control (Table~\ref{tab:qed_main}).

\subsection*{Multi-Property Conditional Generation}

We extend the evaluation to simultaneous 4-property conditioning, where each generated molecule receives target values for QED, SA Score, LogP, and MolWt concurrently. The main multi-property results use the fair v2 protocol: conditioning bins are disabled and positive references are selected by the full normalized property vector rather than by QED alone. Table~\ref{tab:multi4} reports per-property Spearman $\rho$ for the key v2 variants.

\begin{table}[htbp]
\centering
\caption{\textbf{Fair Multi-Property Conditional Generation on ZINC250K.} Per-property Spearman $\rho$ when simultaneously conditioning on four properties under the v2 no-binning protocol. $\bar{\rho}$ = mean across properties. Best $\alpha$ per variant. All achieve V $= 100$\%.}
\label{tab:multi4}
\begin{tabular}{l c c c c c c c c}
\toprule
Variant & $\alpha$ & QED $\rho$ & SA $\rho$ & LogP $\rho$ & MolWt $\rho$ & $\bar{\rho}$ & Min U (\%) & Avg U (\%) \\
\midrule
Single-$\tau$ & 5.0 & \textbf{0.367} & 0.529 & \textbf{0.708} & 0.790 & \textbf{0.598} & 88.2 & 93.2 \\
No-Div & 5.0 & 0.315 & \textbf{0.530} & 0.682 & \textbf{0.793} & 0.580 & 72.8 & 77.4 \\
Full & 5.0 & 0.323 & 0.488 & 0.638 & 0.790 & 0.560 & \textbf{91.5} & 94.8 \\
\midrule
Z-Drift & 5.0 & 0.188 & 0.271 & 0.496 & 0.552 & 0.377 & 97.9 & 98.3 \\
Rand-$\varphi$ & 5.0 & 0.187 & 0.282 & 0.506 & 0.518 & 0.373 & 97.0 & 97.7 \\
\midrule
NoDrift & 1.5 & $-$0.005 & 0.024 & 0.003 & 0.009 & 0.008 & 98.0 & 98.2 \\
Stop-Grad & 2.0 & $-$0.016 & 0.007 & $-$0.036 & 0.015 & $-$0.008 & 97.6 & 97.8 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\textwidth]{fig3_multi4_v2.pdf}
\caption{\textbf{Fair multi-property control under the v2 no-binning protocol.} Heatmap cells show per-property Spearman $\rho$ for QED, SA Score, LogP, and MolWt; the side bars show min-bin uniqueness. Decoder-$\varphi$ variants provide the strongest average control, with Full preserving the best min-bin uniqueness among the top tier.}
\label{fig:multi4_v2}
\end{figure}

The tier structure from single-property evaluation persists under the fair protocol: decoder-$\varphi$ variants (Single-$\tau$, No-Div, Full) achieve $\bar{\rho}=0.560$--$0.598$, clearly above z-space and random-$\varphi$ controls ($\bar{\rho}\approx0.37$--$0.38$) and far above NoDrift or Stop-Grad controls ($|\bar{\rho}|<0.01$). Thus, simultaneous conditioning is not produced by generic correlations in the generator alone; it requires an active drift signal, and the decoder-coupled drift remains strongest.

\textbf{Property controllability.} Across the top-performing v2 variants, MolWt and LogP are the most controllable properties ($\rho\approx0.64$--$0.79$), SA Score is intermediate ($\rho\approx0.49$--$0.53$), and QED remains the hardest ($\rho\approx0.32$--$0.37$). MolWt's dominance is expected: molecular weight is strictly additive over atoms, making it highly amenable to continuous latent-space manipulation. QED combines several structural and physicochemical factors, so its target-to-actual mapping compresses more strongly when four properties are steered simultaneously.

\textbf{Legacy diagnostic protocol.} The older multi-property table used QED quantile bins while conditioning on four properties. We retain those runs only as diagnostics in the appendix because they mix a single-property grouping rule with a multi-property target. The v2 results in Table~\ref{tab:multi4} replace the legacy table for all main multi-property claims.

\textbf{Per-target steering.} Table~\ref{tab:multi4_pertarget} shows target$\to$actual mappings for the Full variant across all four properties.

\begin{table}[htbp]
\centering
\caption{\textbf{Multi-Property Per-Target Steering} for DriftingMol-Full ($\alpha{=}5.0$, $\sim$1,100 molecules per bin). Four selected quantile bins (Q1--Q4 from 9 bins) with target $\to$ actual mean values. All properties show monotonic actual trends.}
\label{tab:multi4_pertarget}
\begin{tabular}{l c c c c}
\toprule
Property & Q1 & Q2 & Q3 & Q4 \\
\midrule
QED & $0.46 \to 0.40$ & $0.65 \to 0.52$ & $0.81 \to 0.60$ & $0.91 \to 0.63$ \\
SA Score & $1.99 \to 3.65$ & $2.42 \to 3.98$ & $3.22 \to 4.32$ & $4.63 \to 5.54$ \\
LogP & $-0.13 \to 0.73$ & $1.58 \to 1.80$ & $3.13 \to 3.30$ & $4.54 \to 4.10$ \\
MolWt & $229 \to 112$ & $290 \to 234$ & $354 \to 353$ & $442 \to 410$ \\
\bottomrule
\end{tabular}
\end{table}

All four properties show monotonic actual trends in the correct direction. LogP demonstrates near-perfect tracking in the mid-range bins ($3.13 \to 3.30$) but compression at the low end ($-0.13 \to 0.73$). MolWt shows strong mid-range tracking ($354 \to 353$) but a large negative offset at the low end ($229 \to 112$), consistent with the unconditional distributional shift toward lighter molecules (Section Discussion). QED exhibits the most severe target-to-actual compression, with the full target range $[0.46, 0.91]$ compressed to actual $[0.40, 0.63]$.

\textbf{Single-property vs.\ multi-property.} Comparing with Table~\ref{tab:qed_main}, QED $\rho$ decreases from 0.493 (single-property) to 0.323 (fair multi-property) for the Full model. This is expected: when conditioning on four properties simultaneously, the drift field must balance competing steering signals, diluting per-property QED control. The mean $\bar{\rho}=0.560$ (Full) to $0.598$ (Single-$\tau$) demonstrates that the framework extends to multi-property settings, with the strongest gains on the properties whose latent directions are more directly encoded by structure.

\textbf{Uniqueness.} The No-Div uniqueness penalty is even more pronounced in fair multi-property conditioning: min-bin U $=72.8\%$ vs.\ 91.5\% for Full, and average U $=77.4\%$ vs.\ 94.8\%. This reinforces that z-diversity regularization becomes increasingly important as conditioning complexity grows.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Analysis}

\subsection*{Coupled Decoder Drift is the Core Mechanism}

The dominant factor for property controllability is the \emph{coupled} use of the VAE decoder as the drift feature space $\varphi$ (Table~\ref{tab:qed_main}). All Tier-1 variants (Full, Single-$\tau$, No-Div) achieve $\rho \geq 0.49$ by drifting in the decoder's 512D intermediate representation. Removing the decoder and drifting directly in z-space (Z-Drift: $\rho = 0.286$, $-42\%$), using a random frozen MLP as $\varphi$ (Rand-$\varphi$: $\rho = 0.276$, $-44\%$), or substituting a well-trained LatentMAE external feature extractor (Trained-$\varphi$-P: $\rho = 0.329$, $-33\%$) all degrade performance to Tier~2. This demonstrates that the decoder's learned semantic structure---combined with end-to-end gradient flow---is essential for effective drifting.

The tight clustering of Tier-2 variants ($\rho \in [0.28, 0.33]$) is striking: Z-Drift ($0.286$), Rand-$\varphi$ ($0.276$), and all five Trained-$\varphi$ variants ($0.28$--$0.33$) achieve statistically similar performance despite vastly different $\varphi$ quality. This indicates that without end-to-end gradient coupling, the drift field provides only a weak, geometry-independent steering signal.

\subsection*{Gradient Flow is a Necessary Condition}

Stop-Grad blocks drift gradients from reaching the generator ($\partial \mathcal{L}_\text{drift} / \partial \theta = 0$). The result is catastrophic: $\rho = 0.011$, U $= 0.3\%$. This proves that the drift field must update the generator's parameters to be effective. Without gradient flow, the generator cannot learn to place latents in semantically meaningful regions of $\varphi$-space.

The decoupled variants (Decouple, Decouple+Z) suffer a similar fate. By detaching $\varphi$'s computation from the generator's output, the drift computes on a ``frozen snapshot'' that does not guide the generator, leading to U $\approx 0.3\%$ and complete mode collapse.

\subsection*{Trained Feature Extractors Cannot Replace End-to-End Coupling}

The original Drifting Models framework~\cite{corso2024drifting} trains a separate feature extractor (e.g., SimCLR) as $\varphi$. A natural question is whether a sufficiently good trained $\varphi$ could match the coupled decoder $\varphi$. The five Trained-$\varphi$ variants (C-series) answer this definitively: \textbf{no}.

We trained two LatentMAE feature extractors on ZINC250K latent codes: a \emph{plain} version ($R^2_\text{QED} = 0.52$) and a \emph{property-enhanced} version with auxiliary property prediction ($R^2_\text{QED} = 0.68$). Despite substantial property-predictive power, the best trained-$\varphi$ variant (Trained-$\varphi$-P) achieves only $\rho = 0.329$---far below the coupled decoder ($\rho = 0.493$, $\Delta = -0.164$). The entire C-series clusters in $\rho \in [0.28, 0.33]$, indistinguishable from Z-Drift ($\rho = 0.286$) and Rand-$\varphi$ ($\rho = 0.276$).

Several sub-comparisons illuminate the mechanism:
\begin{itemize}
    \item \textbf{$\varphi$ quality matters marginally:} Trained-$\varphi$-P ($R^2{=}0.68$) outperforms Trained-$\varphi$ ($R^2{=}0.52$) by only $+0.032$ in $\rho$. A property-predictive feature space provides a modestly better drift manifold, but the improvement is small compared to the gap from the decoder ($\Delta = 0.164$).
    \item \textbf{Combining trained-$\varphi$ with decoder drift provides no synergy:} $\varphi$+Dec ($\rho = 0.281$) and $\varphi$-P+Dec ($\rho = 0.296$) perform no better than their non-decoder counterparts ($\rho = 0.297$, $0.329$). The LatentMAE component appears to interfere with rather than complement the decoder drift.
    \item \textbf{Adding z-drift helps marginally:} Trained-$\varphi$+Z ($\rho = 0.312$) gains $+0.015$ over Trained-$\varphi$ ($\rho = 0.297$), consistent with the z-drift baseline providing a weak auxiliary signal.
\end{itemize}

The critical insight is that the decoder $\varphi$'s advantage is not its feature quality but its \emph{differentiable coupling} to the generator. The Jacobian $\partial\varphi_\text{dec}/\partial z$ provides an adaptive gradient metric that a frozen external $\varphi$ cannot replicate. Even when the LatentMAE features \emph{predict} properties well, they cannot \emph{steer} the generator because the gradient $\partial\varphi_\text{MAE}/\partial z$ does not reflect the decoder's structural sensitivity.

\subsection*{z-Diversity Regularization Prevents Mode Collapse}

No-Div ($\lambda_\text{zdiv} = 0$) achieves the highest conditional $\rho = 0.510$ but at catastrophic cost: unconditional U drops to 7.4\% (vs.\ Full: 96.1\%). Without $\mathcal{L}_\text{zdiv}$, the generator concentrates latents in a small region, boosting property correlation by reducing variance but destroying diversity. The kNN repulsion term in $\mathcal{L}_\text{zdiv}$ ensures generated latents remain spread in z-space, making z-div regularization essential for practical use.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.72\textwidth]{fig5_zdiv_pareto.pdf}
\caption{\textbf{z-diversity control--diversity Pareto sweep.} Points sweep $\lambda_\text{zdiv}$ for the G4 QED setting and plot QED Spearman $\rho$ against uniqueness. The curve quantifies the regularization strength needed to retain molecular diversity while preserving conditional control.}
\label{fig:zdiv_pareto}
\end{figure}

\subsection*{Multi-Temperature Provides No Benefit}

Single-$\tau$ ($\tau{=}1.0$) achieves $\rho = 0.500$ vs.\ Full's $\rho = 0.493$ for single-property QED---a negligible difference. In fair multi-property conditioning, Single-$\tau$ achieves $\bar{\rho}=0.598$ vs.\ Full's $\bar{\rho}=0.560$ (Table~\ref{tab:multi4}), a consistent advantage that suggests multi-temperature aggregation ($\tau \in \{0.5, 1.0, 2.0\}$) provides no benefit and may introduce slight interference for this task. A single temperature suffices to capture the relevant structure in $\varphi$-space. Based on this finding, we recommend Single-$\tau$ as the default DriftingMol configuration when maximal control is desired, while Full remains preferable when preserving min-bin uniqueness is the primary criterion.

\subsection*{Why Decoder-$\varphi$ Provides Better Drift Geometry}

The effectiveness of decoder drift over alternatives ($\rho \geq 0.49$ vs.\ $\leq 0.33$) cannot be explained by feature quality alone. The Rand-$\varphi$ baseline ($\rho = 0.276$, 512D random MLP) and the Trained-$\varphi$-P ($\rho = 0.329$, LatentMAE with $R^2_\text{QED}{=}0.68$) demonstrate a $6\times$ range in feature quality yet only a $0.053$ difference in $\rho$. Meanwhile, the decoder $\varphi$ ($\rho = 0.493$) is $0.164$ above the best trained alternative, despite never being explicitly trained to predict properties.

The decoder's advantage stems from two interacting factors:

\textbf{(1) Task-relevant geometry.} The VAE decoder is trained to reconstruct molecules from latent codes; its hidden state $\varphi(z) = \operatorname{mean}_\ell h_\text{dec}^{(L)}(z)_\ell$ has learned to separate latent directions that lead to structurally distinct molecules. Because molecular properties are deterministic functions of structure, the decoder's structural separation induces property-relevant clustering in $\varphi$-space, creating well-defined attraction basins for the drift field.

\textbf{(2) Adaptive gradient metric.} Because gradients flow through $\varphi$ via the chain rule ($\partial \mathcal{L} / \partial\theta = \partial\mathcal{L}/\partial\varphi \cdot \partial\varphi/\partial z \cdot \partial z/\partial\theta$), the local Jacobian $\partial\varphi/\partial z$ acts as an adaptive metric that weights generator updates according to the decoder's sensitivity. A frozen external $\varphi$ (whether random, trained, or property-enhanced) cannot provide this adaptive coupling because its Jacobian does not reflect the decoder's structural knowledge. This explains why even property-predictive LatentMAE features fail: predicting properties is not sufficient; the gradient pathway must encode the decoder's structure-to-property sensitivity.

\subsection*{Why SELFIES Enable Property Control}

The choice of SELFIES is not merely for validity. In graph-based latent models, argmax discretization collapses continuous property signals: our preliminary graph-VAE experiments showed the $\varphi$-predicted property gap between groups was 0.29, but after argmax decoding the actual gap collapsed to 0.009 (${\sim}3\%$ retention). SELFIES decoding is deterministic and continuous-preserving, enabling the regression slope of 0.796 (Table~\ref{tab:alpha_sweep}) that would be impossible with graph decoders.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Discussion}

\subsection*{Property Control Ceiling and Practical Significance}

Our best $\rho = 0.493$ with slope $= 0.796$ indicates meaningful but \emph{not precise} property steering: the generated QED range (0.341--0.673) covers only 63\% of the target range (0.39--0.92). For context, a slope of 0.796 means that requesting a QED increase of 0.10 yields an actual increase of $\sim$0.08 on average---sufficient for coarse library design (e.g., enriching a virtual library toward high-QED compounds) but insufficient for precise property targeting (e.g., ``generate molecules with QED $= 0.85 \pm 0.02$''). This positions DriftingMol as a \emph{property-biased} generator rather than a \emph{property-precise} one.

The systematic attenuation has multiple contributing factors: (i)~the VAE's $\beta$-regularized latent space inherently compresses molecular diversity into a smooth manifold (the VAE achieves 82.3\% exact reconstruction, meaning $\sim$18\% of fine-grained structural variation is lost at encoding); (ii)~the 1-NFE constraint limits the generator's expressiveness compared to multi-step methods; and (iii)~the drift field's kernel-based matching is inherently soft, averaging over neighbor contributions rather than targeting exact property values. Potential improvements include auxiliary property regression loss during generator training, higher-capacity VAEs with lower $\beta$, and iterative refinement of generated latents.

\subsection*{Unconditional Distribution Shift}

The unconditional generation of the Full variant produces molecules with QED $= 0.588$ (ZINC reference: $0.732$)---a notable distributional shift toward lower-QED molecules. This shift is consistent across the well-functioning variants (Table~\ref{tab:uncond}). We attribute this to two factors: (i)~the VAE's KL regularization encourages latent codes near $\mathcal{N}(0,I)$, and the generator's noise-to-latent mapping may not perfectly cover the tails of the training distribution; (ii)~the drift field's preference for densely-populated regions of $\varphi$-space, which biases generation toward the mode of the underlying distribution. This distributional shift is a known limitation of single-step generators and represents a quality gap relative to multi-step methods that can iteratively refine samples toward the target distribution.

\subsection*{Novelty and Reconstruction}

All variants achieve Novelty $\geq 99.9\%$, meaning virtually no generated molecule matches any training molecule exactly. While this indicates strong generalization, it also reflects the VAE's imperfect reconstruction (82.3\% exact match): decoded molecules from nearby but distinct latent codes often differ from training molecules, inflating novelty. This is not a disadvantage per se---for drug discovery, novel molecules are desired---but it means Novelty $= 100\%$ should not be interpreted as evidence of exceptional generative diversity.

\subsection*{The Controllability-Diversity Trade-off}

A recurring theme is the tension between property control ($\rho$) and generation diversity (U, IntDiv). No-Div achieves $\rho = 0.510$ but U $= 74.1\%$; Full achieves $\rho = 0.493$ with U $= 94.7\%$. The z-div regularization provides a principled way to navigate this trade-off, analogous to the KL penalty in VAEs. Interestingly, the trained-$\varphi$ variants with decoder coupling ($\varphi$+Dec, $\varphi$-P+Dec) achieve the highest unconditional uniqueness ($>$98\%) but weak conditional control, suggesting that the separate LatentMAE feature space acts as a diversity-promoting regularizer that simultaneously dampens the decoder's steering signal.

\subsection*{Implications for the Drifting Models Paradigm}

Our findings have broader implications for the Drifting Models framework~\cite{corso2024drifting}. The original formulation trains an external feature extractor (e.g., SimCLR for images) as $\varphi$. Our results suggest that when the generation pipeline includes a decoder with a suitable intermediate representation, repurposing that decoder as $\varphi$ is strictly preferable to training a separate extractor---regardless of the extractor's quality. This is because the decoder provides both geometric structure \emph{and} gradient coupling, whereas an external extractor provides only the former. For image generation, this would correspond to using a frozen decoder (e.g., from a VQ-VAE) as $\varphi$ rather than a separately trained SimCLR model.

\subsection*{Limitations}

\textbf{Multi-seed confirmation.} The main tables report canonical seed-42 runs for the full ablation, while Table~\ref{tab:qed_3seed} reports the three-seed aggregate for the key QED variants. At the current publication checkpoint, Single-$\tau$, No-Div, and Full have complete 3/3 seed coverage, with mean $\rho$ values of 0.515, 0.513, and 0.512, respectively. G4 has 2/3 seeds collected and remains the only incomplete row, so final confidence-interval claims and Fig.~\ref{fig:qed_seed_ci} should be interpreted as gated until G4 seed 44 finishes. We note that the \emph{qualitative} tier structure is likely robust to seed variation given the large performance gaps ($\rho$ differences of $> 0.15$ between tiers).

\textbf{External baseline comparisons.} We focus on internal ablation to dissect the mechanism of coupled decoder drift. Direct comparisons with state-of-the-art conditional molecular generators (e.g., REINVENT~\cite{Olivecrona2017}, LIMO~\cite{Eckmann2022LIMO}, FREED~\cite{Yang2024FREED}) are needed to contextualize absolute performance. Most existing methods frame the task as property \emph{optimization} rather than \emph{targeting}, making direct comparison nontrivial.

\textbf{z-div hyperparameters.} The kNN margin $m = 3.0$ and $K = 5$ were selected via preliminary tuning but not subjected to full sensitivity analysis. The No-Div ablation demonstrates the qualitative necessity of z-div, but the optimal $(m, K)$ may differ across datasets.

\textbf{Multi-property evaluation.} The main paper uses the fair v2 multi-property protocol, which disables conditioning bins and selects positives by the full property vector. Legacy QED-binned multi-property runs are retained only as diagnostics because they can introduce property-dependent biases. Additionally, the 4 properties studied (QED, SA, LogP, MolWt) are all continuous scalars; extending to discrete constraints or multi-objective Pareto optimization requires further development.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Broader Impact}

Molecular generation technologies carry dual-use risks. While property-conditional generation can accelerate drug discovery and materials science by enabling rapid exploration of chemical space, the same capability could in principle be used to design harmful substances. We note that DriftingMol's property control is coarse (``property-biased'' rather than ``property-precise''), limiting its utility for precise molecular design without additional experimental validation. The ZINC250K dataset used in this work contains only known drug-like molecules, and the SELFIES-based system cannot generate molecules outside the expressivity of the SELFIES grammar. We encourage responsible use of molecular generation tools and adherence to ethical guidelines for chemical research.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Conclusion}

We presented DriftingMol, a two-stage framework for single-step property-conditional molecular generation via latent drifting models on SELFIES representations. Our key contribution is the \textit{coupled decoder drift} mechanism, where the frozen VAE decoder provides a semantically rich 512D feature space for the drift field while allowing gradients to flow back to the generator. On ZINC250K, DriftingMol achieves $>$94\% uniqueness (with 100\% validity guaranteed by SELFIES) and Spearman $\rho(\text{QED})$ up to $0.510$ with guidance scale $\alpha{=}5.0$.

A comprehensive 15-variant ablation study reveals clear mechanistic insights:
\begin{itemize}
    \item \textbf{Coupled decoder drift is the core mechanism}: $\rho \geq 0.49$ (decoder-$\varphi$) vs.\ $0.286$ (z-space) vs.\ $0.276$ (random-$\varphi$).
    \item \textbf{Gradient flow is necessary}: stop-gradient and decoupled variants collapse ($\rho < 0.03$, U $< 1\%$).
    \item \textbf{End-to-end coupling, not feature quality}: even a well-trained LatentMAE ($R^2_\text{QED}{=}0.68$) achieves only $\rho = 0.329$, confirming that gradient coupling through the decoder---not feature quality per se---is the key mechanism.
    \item \textbf{z-diversity regularization is essential}: without it, $\rho$ improves to $0.510$ but unconditional uniqueness drops to $7.4\%$.
    \item \textbf{Multi-temperature is dispensable}: Single-$\tau$ matches or exceeds the multi-$\tau$ Full model across all settings.
\end{itemize}

Extension to simultaneous 4-property conditioning achieves mean $\bar{\rho}$ up to $0.598$ (Single-$\tau$) under the fair no-binning protocol, confirming that the framework scales to multi-property settings. The Full variant gives a slightly lower $\bar{\rho}=0.560$ but stronger min-bin uniqueness (91.5\% vs.\ 88.2\%), making the control-diversity trade-off explicit.

These findings establish that coupled decoder drift with SELFIES validity, gradient-based generator training, and z-space diversity regularization forms a practical recipe for 1-NFE property-conditional molecular generation. The single-temperature variant (Single-$\tau$) provides the highest controllability ($\bar{\rho}=0.598$), whereas the Full model provides the stronger diversity-preserving default.

\begin{appendices}

\section{Hyperparameter Details}\label{appA}

\textbf{SELFIES VAE:} Encoder: Transformer encoder with attention pooling, hidden=512. Decoder: one-shot non-causal Transformer, hidden=512. Vocab=108, max\_len=64. Latent dim=256. $\beta = 0.01$ (fixed). Trained on ZINC250K ($N = 249{,}455$). Achieves 82.3\% exact reconstruction, 99.1\% token accuracy, and prior VUN $= 0.989$.

\textbf{DiT Generator (CFG):} 20.2M parameters. Noise dim: 64, hidden: 384, 8 layers, 8 heads, 16 tokens, MLP ratio: 4. Dropout: 0.1. Token noise std: 0.03. FiLM conditioning with learnable null token ($p_\text{uncond} = 0.1$).

\textbf{Training:} 300 epochs, Adam ($\text{lr} = 2 \times 10^{-4}$, weight decay $= 0.01$). Cosine schedule with 5-epoch warmup, min LR ratio $= 0.01$. EMA decay $= 0.999$. Gradient clip norm $= 1.0$. Single RTX 4090D per experiment.

\textbf{Drift Field:} $\lambda_\text{drift} = 1.0$, $\lambda_\text{zdiv} = 2.0$, $m_\text{zdiv} = 3.0$, $K_\text{zdiv} = 5$. Temperatures: $\{0.5, 1.0, 2.0\}$. Normalization mode: \texttt{xy}. Positive mode: hybrid (binning + kNN). $N_\text{pos} = 2048$, $N_\text{gen} = 32$, $N_\text{unc} = 32$, $N_\text{bins} = 20$ (quantile). The margin $m = 3.0$ was chosen via preliminary tuning: values $m < 2.0$ provided insufficient repulsion (uniqueness degraded), while $m > 5.0$ interfered with drift convergence. A full sensitivity analysis is deferred to future work.

\textbf{LatentMAE (for Trained-$\varphi$ variants):} 4-layer Transformer encoder (hidden=256, 4 heads), trained on ZINC250K latent codes with masked autoencoding (mask ratio 0.5). Two versions: plain ($R^2_\text{QED} = 0.52$, reconstruction-only loss) and property-enhanced ($R^2_\text{QED} = 0.68$, with auxiliary QED/SA/LogP/MolWt prediction heads, $\lambda_\text{prop} = 1.0$). Both trained for 200 epochs with Adam ($\text{lr} = 3 \times 10^{-4}$). The LatentMAE's 256D output serves as $\varphi$ for the C-series drift variants.

\textbf{Evaluation:} Best checkpoint by VUN quality gate. At each $\alpha \in \{1.0, 1.5, 2.0, 3.0, 5.0\}$: 10,000 molecules generated ($\alpha$ trained on $[1,4]$; $\alpha{=}5.0$ is out-of-distribution). Single-property conditional: target values from training distribution. Multi-property conditional: 9 quantile bins per property $\times$ $\sim$1,100 molecules.

\textbf{Computational Cost.} Table~\ref{tab:compute} summarizes training and
inference costs. A single experiment trains in ${\sim}5$--$6$ GPU-hours.
Inference throughput was measured on \texttt{cuda} with 20,000
samples, batch size 1,000, NFE $=1$, and two-pass CFG
disabled. The benchmark reports generator-only
throughput of $47{,}500$ mol/s and end-to-end SELFIES decoding throughput of
$4{,}600$ mol/s.

\begin{table}[htbp]
\centering
\caption{\textbf{Computational Cost} on a single RTX 4090D GPU. Training time is for 300 epochs. Inference throughput is measured by \texttt{scripts/benchmark\_inference.py} using 1-NFE generation.}
\label{tab:compute}
\begin{tabular}{l c c}
\toprule
 & Full & Single-$\tau$ \\
\midrule
Model params (generator) & 20.2M & 20.2M \\
Model params (VAE, frozen) & 32.1M & 32.1M \\
Training time (single-prop) & 5.9 GPU-hr & 5.3 GPU-hr \\
Training time (multi-prop) & 5.8 GPU-hr & 5.2 GPU-hr \\
Inference: generator only & \multicolumn{2}{c}{$47{,}500$ mol/s} \\
Inference: end-to-end & \multicolumn{2}{c}{$4{,}600$ mol/s} \\
Peak GPU memory & \multicolumn{2}{c}{1.5 GB} \\
\bottomrule
\end{tabular}
\end{table}

\section{Additional Unconditional Results}\label{appB}

\begin{table}[htbp]
\centering
\caption{Unconditional generation distributional statistics for all variants ($\alpha{=}1.0$). ZINC reference: QED $= 0.732 \pm 0.139$, SA $= 3.05 \pm 0.84$, LogP $= 2.46 \pm 1.43$, MolWt $= 332.1 \pm 61.9$.}
\label{tab:uncond_full}
\begin{tabular}{l c c c c c c}
\toprule
Variant & U (\%) & QED & SA & LogP & IntDiv \\
\midrule
$\varphi$+Dec & 98.7 & 0.567 & 4.16 & 2.62 & 0.896 \\
$\varphi$-P+Dec & 98.4 & 0.561 & 4.17 & 2.67 & 0.897 \\
Z-Drift & 98.2 & 0.536 & 4.24 & 2.64 & 0.905 \\
Trained-$\varphi$+Z & 98.1 & 0.539 & 4.14 & 2.64 & 0.901 \\
Trained-$\varphi$ & 97.4 & 0.501 & 4.13 & 2.45 & 0.911 \\
Stop-Grad & 97.0 & 0.559 & 4.31 & 3.10 & 0.904 \\
Rand-$\varphi$ & 96.8 & 0.544 & 3.94 & 2.60 & 0.903 \\
Trained-$\varphi$-P & 96.5 & 0.524 & 4.04 & 2.63 & 0.904 \\
\textbf{Full} & 96.1 & 0.588 & 4.06 & 2.70 & 0.889 \\
Single-$\tau$ & 95.5 & 0.583 & 4.11 & 2.75 & 0.887 \\
MLP-Head & 94.9 & 0.493 & 5.27 & 3.69 & 0.892 \\
\midrule
No-Div & 7.4 & 0.630 & 3.77 & 3.27 & 0.846 \\
Decouple & 0.9 & 0.483 & 2.14 & 2.58 & 0.693 \\
Decouple+Z & 1.1 & 0.484 & 2.15 & 2.58 & 0.690 \\
\bottomrule
\end{tabular}
\end{table}

Decouple/Decouple+Z generate extremely small molecules (SA $\approx 2.1$), indicating the decoupled drift mechanism causes mode collapse to simple structures. MLP-Head shows distributional shift toward harder-to-synthesize molecules (SA $= 5.27$ vs.\ ZINC $3.05$). Among trained-$\varphi$ variants, $\varphi$+Dec and $\varphi$-P+Dec achieve excellent unconditional diversity (U $> 98\%$, IntDiv $\approx 0.90$) despite their limited conditional control, suggesting the LatentMAE features promote latent space coverage.

\section{Full Alpha-Sweep Tables}\label{appC}

\begin{table}[htbp]
\centering
\caption{Full $\alpha$-sweep for selected variants --- QED conditional. Single-$\tau$ and Full show strong $\rho$ scaling; Trained-$\varphi$-P (best trained-$\varphi$) shows weaker scaling with a persistent gap.}
\label{tab:sweep_selected}
\begin{tabular}{l c c c c c c}
\toprule
& $\alpha$ & $\rho$ & MAE & slope & U (\%) & IntDiv \\
\midrule
\multirow{5}{*}{No-Div} & 1.0 & 0.315 & 0.178 & 0.478 & 75.2 & 0.889 \\
& 1.5 & 0.385 & 0.177 & 0.631 & 76.7 & 0.887 \\
& 2.0 & 0.438 & 0.174 & 0.716 & 76.0 & 0.887 \\
& 3.0 & 0.502 & 0.173 & 0.830 & 75.7 & 0.882 \\
& 5.0 & 0.510 & 0.194 & 0.828 & 74.1 & 0.892 \\
\midrule
\multirow{5}{*}{Z-Drift} & 1.0 & 0.128 & 0.242 & 0.189 & 98.0 & 0.912 \\
& 1.5 & 0.174 & 0.242 & 0.265 & 97.9 & 0.912 \\
& 2.0 & 0.205 & 0.241 & 0.309 & 97.9 & 0.911 \\
& 3.0 & 0.224 & 0.241 & 0.349 & 98.0 & 0.913 \\
& 5.0 & 0.286 & 0.245 & 0.461 & 98.5 & 0.913 \\
\midrule
\multirow{5}{*}{Trained-$\varphi$-P} & 1.0 & 0.171 & 0.251 & 0.243 & 96.9 & 0.912 \\
& 1.5 & 0.231 & 0.253 & 0.346 & 96.8 & 0.929 \\
& 2.0 & 0.259 & 0.251 & 0.383 & 96.9 & 0.931 \\
& 3.0 & 0.286 & 0.255 & 0.437 & 95.9 & 0.928 \\
& 5.0 & 0.329 & 0.259 & 0.499 & 94.8 & 0.892 \\
\midrule
\multirow{5}{*}{$\varphi$+Dec} & 1.0 & 0.222 & 0.206 & 0.333 & 98.7 & 0.897 \\
& 1.5 & 0.234 & 0.206 & 0.358 & 98.6 & 0.898 \\
& 2.0 & 0.267 & 0.201 & 0.415 & 98.9 & 0.896 \\
& 3.0 & 0.281 & 0.200 & 0.432 & 98.8 & 0.898 \\
& 5.0 & 0.280 & 0.201 & 0.432 & 98.8 & 0.898 \\
\bottomrule
\end{tabular}
\end{table}

The Trained-$\varphi$-P variant follows a similar scaling pattern to Z-Drift but at a modestly higher baseline (Trained-$\varphi$-P $\rho = 0.171$ at $\alpha{=}1.0$ vs.\ Z-Drift $0.128$), converging to $\rho = 0.329$ vs.\ $0.286$ at $\alpha{=}5.0$. The $\varphi$+Dec variant shows notably weak $\alpha$-scaling: $\rho$ plateaus at $\approx 0.28$ by $\alpha{=}3.0$, suggesting the LatentMAE component interferes with the decoder's guidance signal at high $\alpha$.

\section{Success-Rate Metrics}\label{appD}

To complement rank-correlation ($\rho$), we estimate the fraction of generated molecules falling within a tolerance $\delta$ of their target property value (Success@$\delta$). Estimates are based on a Gaussian approximation using per-bin mean and standard deviation from the formal evaluation of 10,000 molecules.

\begin{table}[htbp]
\centering
\caption{\textbf{Estimated Success@$\delta$ for QED-Conditional Generation} ($\alpha{=}5.0$). Fraction of molecules within $\pm\delta$ of target. Decoder-$\varphi$ variants (Full, Single-$\tau$) substantially outperform z-drift (Z-Drift).}
\label{tab:success_qed}
\begin{tabular}{l c c c c c}
\toprule
Variant & $\rho$ & MAE & S@0.05 & S@0.10 & S@0.15 \\
\midrule
Full & 0.493 & 0.200 & 0.137 & 0.274 & 0.408 \\
Single-$\tau$ & 0.500 & 0.204 & 0.132 & 0.265 & 0.397 \\
Z-Drift & 0.286 & 0.245 & 0.107 & 0.216 & 0.327 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{\textbf{Legacy Multi-Property Success@$\delta$ Diagnostic} for key variants (best $\alpha$). These estimates use the older QED-binned multi-property protocol and are retained only as diagnostic evidence. $\delta$ is chosen per property: QED $\pm 0.05$, SA $\pm 0.3$, LogP $\pm 0.5$, MolWt $\pm 20$.}
\label{tab:success_multi}
\begin{tabular}{l c c c c c}
\toprule
Variant & QED S@0.05 & SA S@0.3 & LogP S@0.5 & MolWt S@20 & $\bar{\rho}$ \\
\midrule
Single-$\tau$ & 0.143 & 0.091 & 0.294 & 0.188 & 0.387 \\
Full & 0.133 & 0.103 & 0.285 & 0.175 & 0.363 \\
Z-Drift & 0.115 & 0.091 & 0.243 & 0.199 & 0.073 \\
\bottomrule
\end{tabular}
\end{table}

Within this historical diagnostic protocol, the success rates show the same tier
structure as the legacy $\rho$ table: decoder-$\varphi$ variants (Full,
Single-$\tau$) consistently outperform z-drift (Z-Drift). However, the absolute
success rates are modest (e.g., S@0.05 $\approx 13\%$ for QED), consistent with
the ``property-biased'' rather than ``property-precise'' characterization
discussed in this paper. Fair no-binning success-rate estimates should be
regenerated from the active publication multi-property runs before being used as
main evidence.

\section{Figure Descriptions}\label{appE}

Figures are generated by:
\begin{itemize}
    \item \texttt{python scripts/plot\_main\_figure.py} for Fig.~\ref{fig:main}.
    \item \texttt{python scripts/plot\_result\_figures.py} for Fig.~\ref{fig:qed_ablation}, Fig.~\ref{fig:multi4_v2}, Fig.~\ref{fig:qed_seed_ci}, and Fig.~\ref{fig:zdiv_pareto}.
\end{itemize}

The z-diversity Pareto curve is emitted once at least two sweep points are
complete. The CI plot is intentionally gated on complete 3/3 seed coverage for
Full, Single-$\tau$, No-Div, and G4.

\end{appendices}


\bibliographystyle{sn-nature}

\begin{thebibliography}{25}

\bibitem{corso2024drifting}
Deng, M., Li, H., Li, T., Du, Y., \& He, K.
Generative Modeling via Drifting.
\textit{arXiv preprint arXiv:2602.04770} (2025).

\bibitem{Digress}
Vignac, C., et al.
DiGress: Discrete Denoising Diffusion for Graph Generation.
\textit{ICLR} (2023).

\bibitem{MoFlow}
Zang, C. \& Wang, F.
MoFlow: An Invertible Flow Model for Generating Molecular Graphs.
\textit{KDD} (2020).

\bibitem{JT-VAE}
Jin, W., Barzilay, R., \& Jaakkola, T.
Junction Tree Variational Autoencoder for Molecular Graph Generation.
\textit{ICML} (2018).

\bibitem{MolGAN}
De Cao, N. \& Kipf, T.
MolGAN: An Implicit Generative Model for Small Molecular Graphs.
\textit{ICML Workshop} (2018).

\bibitem{GraphVAE}
Simonovsky, M. \& Komodakis, N.
GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders.
\textit{ICANN} (2018).

\bibitem{GraphNVP}
Madhawa, K., et al.
GraphNVP: An Invertible Flow Model for Generating Molecular Graphs.
\textit{arXiv:1905.11600} (2019).

\bibitem{GDSS}
Jo, J., Lee, S., \& Hwang, S. J.
Score-based Generative Modeling of Graphs via SDEs.
\textit{ICML} (2022).

\bibitem{Irwin2012ZINC}
Irwin, J. J., Sterling, T., Mysinger, M. M., Bolstad, E. S., \& Coleman, R. G.
ZINC: A Free Tool to Discover Chemistry for Biology.
\textit{J.\ Chem.\ Inf.\ Model.}, 52(7), 1757--1768 (2012).

\bibitem{Krenn2020SELFIES}
Krenn, M., H\"ase, F., Nigam, A., Friederich, P., \& Aspuru-Guzik, A.
Self-Referencing Embedded Strings (SELFIES): A 100\% robust molecular string representation.
\textit{Machine Learning: Science and Technology}, 1(4), 045024 (2020).

\bibitem{Krenn2022SELFIES}
Krenn, M., et al.
SELFIES and the future of molecular string representations.
\textit{Patterns}, 3(10), 100588 (2022).

\bibitem{Ho2022CFG}
Ho, J. \& Salimans, T.
Classifier-Free Diffusion Guidance.
\textit{NeurIPS Workshop} (2022).

\bibitem{Peebles2023DiT}
Peebles, W. \& Xie, S.
Scalable Diffusion Models with Transformers.
\textit{ICCV} (2023).

\bibitem{Gomez2018}
G\'omez-Bombarelli, R., et al.
Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules.
\textit{ACS Central Science}, 4(2), 268--276 (2018).

\bibitem{Olivecrona2017}
Olivecrona, M., Blaschke, T., Engkvist, O., \& Chen, H.
Molecular De-Novo Design through Deep Reinforcement Learning.
\textit{J.\ Cheminformatics}, 9, 48 (2017).

\bibitem{Eckmann2022LIMO}
Eckmann, P., Sun, K., Zhao, B., Feng, M., Gilmer, M., \& Yu, R.
LIMO: Latent Inceptionism for Targeted Molecule Generation.
\textit{ICML} (2022).

\bibitem{Zhou2019MolDQN}
Zhou, Z., Kearnes, S., Li, L., Zare, R. N., \& Riley, P.
Optimization of Molecules via Deep Reinforcement Learning.
\textit{Scientific Reports}, 9, 10752 (2019).


\bibitem{Yang2024FREED}
Yang, J., et al.
FREED: Fragment-based Exploration with Energy-based Directed generation.
\textit{ICLR} (2024).

\bibitem{Lee2023MOOD}
Lee, S., Jo, J., \& Hwang, S. J.
Exploring Chemical Space with Score-based Out-of-distribution Generation.
\textit{ICML} (2023).

\bibitem{Lipman2023CFM}
Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., \& Le, M.
Flow Matching for Generative Modeling.
\textit{ICLR} (2023).

\end{thebibliography}

\end{document}
