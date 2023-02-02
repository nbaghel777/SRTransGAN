# SRTransGAN: SRTransGAN: Image Super-Resolution using Transformer based Generative Adversarial Network
[Neeraj Baghel](https://sites.google.com/view/nbaghel777) , [Satish Singh](https://cvbl.iiita.ac.in/sks/) and [Shiv Ram Dubey](https://profile.iiita.ac.in/srdubey/)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)]()
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)]()
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)]()
[![Summary](https://img.shields.io/badge/Summary-Slide-87CEEB)]()

#### News
<!--
- **April 4, 2022:** Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/swzamir/Restormer)
- **March 30, 2022:** Added Colab Demo. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C2818h7KnjNv4R1sabe14_AYL7lWhmu6?usp=sharing)
- **March 29, 2022:** Restormer is selected for an ORAL presentation at CVPR 2022 :dizzy:
- **March 10, 2022:** Training codes are released :fire:
- **March 3, 2022:** Paper accepted at CVPR 2022 :tada: 
 -->
- **Jan, 2023:** Testing codes and pre-trained models are released!
- **April, 2022:** Paper submitted

<hr />

> **Abstract:** *Image super-resolution aims to synthesize high-resolution image from a low-resolution image. 
It is an active area to overcome the resolution limitations in several applications like low-resolution object-recognition, medical image enhancement, etc. 
The generative adversarial network (GAN) based methods have been the state-of-the-art for image super-resolution by utilizing the convolutional neural networks (CNNs) based generator and discriminator networks. However, the CNNs are not able to exploit the global information very effectively in contrast to the transformers, which are the recent breakthrough in deep learning by exploiting the self-attention mechanism. Motivated from the success of transformers in language and vision applications, we propose a SRTransGAN for image super-resolution using transformer based GAN. Specifically, we propose a novel transformer-based encoder-decoder network as a generator to generate $2\times$ images and $4\times$ images. We design the discriminator network using vision transformer which uses the image as sequence of patches and hence useful for binary classification between synthesized and real high-resolution images. 
The proposed SRTransGAN outperforms the existing methods by 4.38\% on an average of PSNR and SSIM scores. We also analyze the saliency map to show the effectiveness of the proposed method. The code used in the paper will be publicly available at xxxxxxxxx.
<hr />

# Network Architecture
<img src = "https://github.com/nbaghel777/SRTransGAN/blob/main/ETSR-Generator.png"> 

# Training and Evaluation
% removed Manga109 results
\begin{table}[t]
% \centering
\caption{PSNR and SSIM comparision has been performed among diffrent SR methords on diffrent dataset and diffrent scale ($2\times$  and $4\times$  super-resolution)}
\label{tab:PSNR/SSIMx2x4}
\resizebox{\columnwidth}{!}{
\begin{tabular}{m{0.20\columnwidth}m{0.022\columnwidth}m{0.13\columnwidth}m{0.13\columnwidth}m{0.13\columnwidth}m{0.13\columnwidth}}
\hline
Method &Sca & Set5 & Set14 & BSD100 & Urban100 \\
&le& PSNR/SSIM & PSNR/SSIM & PSNR/SSIM & PSNR/SSIM \\
\hline
SRCNN \cite{SRCNNr1} & &  36.66/0.9542 & 32.45/0.9067 & 31.36/0.8879 & 29.50/0.8946 \\
FSRCNN \cite{FSRCNNr2} & &37.00/0.9558 & 32.63/0.9088 & 31.53/0.8920 & 29.88/0.9020 \\
VDSR \cite{VDSRr3} & &  37.53/0.9587 & 33.03/0.9124 & 31.90/0.8960 & 30.76/0.9140  \\
DRCN \cite{DRCN} & & 37.63/0.9588 & 33.04/0.9118 & 31.85/0.8942 & 30.75/0.9133  \\
LapSRN \cite{LapSRN}& & 37.52/0.9591 & 32.99/0.9124 & 31.80/0.8952 & 30.41/0.9103 \\
DRRN \cite{DRRN} & & 37.74/0.9591 & 33.23/0.9136 & 32.05/0.8973 & 31.23/0.9188 \\
MemNet \cite{Memnet} & & 37.78/0.9597 & 33.28/0.9142 & 32.08/0.8978 & 31.31/0.9195 \\
EDSR \cite{EDSR-baseline} & & 37.99/0.9604 & 33.57/0.9175 & 32.16/0.8994 & 31.98/0.9272  \\
SRMDNF \cite{SRMDNF} & & 37.79/0.9601 & 33.32/0.9159 & 32.05/0.8985 & 31.33/0.9204  \\
CARN \cite{CARN} & $2\times$ & 37.76/0.9590 & 33.52/0.9166 & 32.09/0.8978 & 31.92/0.9256  \\
IMDN \cite{IMDN} & & 38.00/0.9605 & 33.63/0.9177 & 32.19/0.8996 & 32.17/0.9283  \\
ESRT \cite{ESRT} & & 38.03/0.9600 & 33.75/0.9184 & 32.25/0.9001 & 32.58/0.9318  \\
RCAN \cite{RCAN} & & 38.27/0.9614 & 34.12/0.9216 & 32.41/0.9027 & \underline{33.34/0.9384}  \\
OISR \cite{OISR-RK3} & & 38.21/0.9612 & 33.94/0.9206 & 32.36/0.9019 & 33.03/0.9365  \\
RNAN \cite{RNAN} & & 38.17/0.9611 & 33.87/0.9207 & 32.32/0.9014 & 32.73/0.9340  \\
SAN \cite{SAN} & & 38.31/\underline{0.9620} & 34.07/0.9213 & 32.42/\underline{0.9028} & 33.10/0.9370 \\
IGNN \cite{IGNN} & & 38.24/0.9613 & 34.07/\underline{0.9217} & 32.41/0.9025 & 33.23/0.9383 \\
IPT \cite{IPT} &  & \underline{38.37}/$0.959^*$ & \underline{34.43}/$0.924^*$ & \underline{32.48}/$0.943^*$ & \textbf{33.76/0.9535}^*  \\
Proposed & & \textbf{43.862}/\textbf{0.986} & \textbf{36.162}/\textbf{0.944} & \textbf{36.977}/\textbf{0.957} & 31.935/0.933 \\
\hline
SRCNN \cite{SRCNNr1} & & 30.48/0.8628 & 27.50/0.7513 & 26.9/0.7101 & 24.52/0.7221 \\
FSRCNN \cite{FSRCNNr2} & & 30.72/0.8660 & 27.61/0.7550 & 26.98/0.7150 & 24.62/0.7280 \\
VDSR \cite{VDSRr3} & &  31.35/0.8838 & 28.01/0.7674 & 27.29/0.7251 & 25.18/0.7524  \\
DRCN \cite{DRCN} & & 31.53/0.8854 & 28.02/0.7670 & 27.23/0.7233 & 25.14/0.7510\\
LapSRN \cite{LapSRN} & & 31.54/0.8852 & 28.09/0.7700 & 27.32/0.7275 & 25.21/0.7562 \\
DRRN \cite{DRRN} & & 31.68/0.8888 & 28.21/0.7720 & 27.38/0.7284 & 25.44/0.7638  \\
MemNet \cite{Memnet} & & 31.74/0.8893 & 28.26/0.7723 & 27.40/0.7281 & 25.50/0.7630  \\
EDSR \cite{EDSR-baseline} & & 32.09/0.8938 & 28.58/0.7813 & 27.57/0.7357 & 26.04/0.7849  \\
SRMDNF \cite{SRMDNF} & & 31.96/0.8925 & 28.35/0.7787 & 27.49/0.7337 & 25.68/0.7731  \\
CARN \cite{CARN} & $4\times$ & 32.13/0.8937 & 28.60/0.7806 & 27.58/0.7349 & 26.07/0.7837  \\
IMDN \cite{IMDN} & &  32.21/0.8948 & 28.58/0.7811 & 27.56/0.7353 & 26.04/0.7838  \\
ESRT \cite{ESRT} & & 32.19/0.8947 & 28.69/0.7833 & 27.69/0.7379 & 26.39/0.7962  \\
RCAN \cite{RCAN} & & 32.63/0.9002 & 28.87/0.7889 & 27.77/\underline{0.7436} & 26.82/\underline{0.8087}  \\
OISR \cite{OISR-RK3} & & 32.53/0.8992 & 28.86/0.7878 & 27.75/0.7428 & 26.79/0.8068  \\
RNAN \cite{RNAN} & &  32.49/0.8982 & 28.83/0.7878 & 27.72/0.7421 & 26.61/0.8023 \\
SAN \cite{SAN} & &  \underline{32.64}/\underline{0.9003} & 28.92/0.7888 & 27.78/\underline{0.7436} & 26.79/0.8068 \\
IGNN \cite{IGNN} & & 32.57/0.8998 & 28.85/\underline{0.7891} & 27.77/0.7434 & \underline{26.84}/\textbf{0.8090} \\
IPT \cite{IPT} & & \underline{32.64}/$0.8260^*$ & \underline{29.01}/$0.6783^*$ & \underline{27.82}/$0.6800^*$ & \textbf{27.26}/$0.7952^*$ \\
Proposed & & \textbf{36.941}/\textbf{0.944} & \textbf{29.509}/\textbf{0.828} & \textbf{30.322}/\textbf{0.823} & 25.519/0.761  \\ \hline
\multicolumn{6}{c} {Here, * denotes the reproduced results from pre-train model.}
\end{tabular}}

\end{table}

# Results
<img src = "[https://github.com/nbaghel777/SRTransGAN/blob/main/PTSR-x2.jpg](https://github.com/nbaghel777/SRTransGAN/blob/main/Screenshot%20from%202023-02-02%2011-12-05.png)"> 

# Contact:
Should you have any question, please contact neerajbaghel@ieee.org
1) SRTransGAN: SRTransGAN: Image Super-Resolution using Transformer based Generative Adversarial Network

# Citation
If you use SRTransGAN, please consider citing:

@inproceedings{baghel2022srtransgan,
    title={SRTransGAN: SRTransGAN: Image Super-Resolution using Transformer based Generative Adversarial Network}, 
    author={Neeraj Baghel and Satish Singh and Shiv Ram Dubey},
    year={2022}
}

# Related Works (Also cite this): 
1) Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)
2) ViTGAN: Training GANs with Vision Transformers

