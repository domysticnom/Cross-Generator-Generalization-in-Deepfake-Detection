# Cross-Generator Generalization in Deepfake Detection

**IE7374: Generative AI, Northeastern University**  
**Project Proposal (Revised / Final)**  
**Dominic Rivas · Jonathan Jude Regalado · Lyxelis Rodriguez Navarro · Obinna Okonkwo · Sagar Ayare**  


## 1. Project Title and Description

This project sits in the **computer vision** application area. We train conventional convolutional neural network (CNN) deepfake detectors and characterize how well they generalize to manipulation methods they were never trained on. The vision framing is deliberate: deepfakes are produced by generative models (Generative Adversarial Networks and, increasingly, diffusion models), and detecting their visual output is the discriminative counterpart to the generative-AI techniques studied in this course.

To ground the project in generative modeling directly, not only in the detection of it, the pipeline **also implements a generative model of its own**. We use a pretrained **SimSwap** face-swap generator (Chen et al., 2020) to synthesize a fresh set of fake videos from real source footage, and treat those self-generated forgeries as a **fifth, self-produced "unseen generator"** alongside FaceForensics++'s four manipulation methods. SimSwap belongs to a generator family absent from FF++, a modern, subject-agnostic learned face swap, so it is a genuine out-of-distribution test rather than a re-run of a method the detectors already know. Crucially, this adds a generative component we implement and evaluate against while leaving the transfer-matrix and leave-one-out design fully intact; we *use* a pretrained generator rather than train one from scratch.

Concretely, we fine-tune two ImageNet-pretrained CNN architectures, **EfficientNet** and **XceptionNet**, on the FaceForensics++ (FF++) dataset, then measure **cross-generator generalization**: the degree to which a detector trained on a subset of FF++'s four manipulation methods still detects forgeries produced by a method held out of training. The novel emphasis of the project is methodological honesty. Rather than reporting a single in-distribution accuracy number, the metric most commonly published, we report the **seen-versus-unseen performance gap** as a transfer matrix across all four manipulation methods. A detector that scores ~96% on its training distribution but collapses toward chance on an unseen generator is a more informative and more honest result than a single headline number, and it directly characterizes the limits of standard CNN detectors.


## 2. Problem Statement

Deepfakes are AI-generated digital forgeries that are highly realistic and difficult to identify at a glance. Reporting on recent industry testing indicates that fewer than 1% of people can reliably distinguish deepfake content from genuine media (Zupan, 2025). The technology enables fraud, harassment, non-consensual imagery, and large-scale misinformation, and government and intergovernmental bodies have flagged it as a growing threat to information integrity (U.S. Government Accountability Office, 2024; Naffi, 2025). Critically, generation is outpacing detection: as synthesis methods diversify across generator families, detectors trained on yesterday's manipulations are repeatedly caught off guard by new ones.

This is fundamentally a generative-AI problem. Each new face-manipulation technique is a new generative model that imprints its own statistical fingerprint on the output, and a detector that has only ever seen a few such fingerprints has no guarantee of recognizing an unseen one. The specific, well-documented failure mode this project targets is that conventional CNN detectors achieve near-saturated accuracy when trained and tested on the **same** generator, yet degrade sharply, often toward random guessing, on generators held out of training (see Section 3). Despite this, cross-generator transfer is still inconsistently measured and frequently under-reported. **The core problem we address is that cross-generator generalization must be measured, not assumed.**


### Research Questions

1. **RQ1 (generalization gap).** Under a rotating leave-one-manipulation-out protocol on FF++, how large is the gap between a CNN detector's in-distribution performance and its performance on the held-out, unseen manipulation method?
2. **RQ2 (transfer structure).** Which manipulation methods transfer best and worst, and does the pattern align with generator family, face-swap versus reenactment, and computer-graphics versus learning-based synthesis?
3. **RQ3 (architecture sensitivity).** Do EfficientNet and XceptionNet differ in their cross-generator robustness, or is the generalization gap a property of the conventional CNN approach itself?
4. **RQ4 (self-generated generator).** When we synthesize fakes ourselves with a pretrained SimSwap face-swap generator, a generator family not present in FF++, do detectors trained only on FF++ methods still recognize them, and where does this self-generated set fall relative to the seen and held-out FF++ methods?
5. **RQ5 (cross-dataset, stretch).** Does the gap widen further under a full dataset shift, when an FF++-trained detector is evaluated on a never-seen DFDC subset?


## 3. Background

Modern deepfake generation was strongly enabled by Generative Adversarial Networks (Goodfellow et al., 2014), which train two neural networks against each other to produce realistic synthetic media. The term "deepfake" emerged around 2017, building on a longer lineage of computer-vision, computer-graphics, and media-forensics research; more recently, diffusion-based methods have pushed realism further, heightening the detection challenge.


### Existing detection approaches

The dominant detection paradigm treats forgery detection as supervised image classification. The FaceForensics++ benchmark established **XceptionNet** as the strong CNN baseline, reporting binary detection accuracy of 99.26% on raw video, 95.73% on lightly compressed (c23) video, and 81.00% on heavily compressed (c40) video (Rössler et al., 2019). Earlier, **MesoNet** introduced a deliberately shallow CNN targeting mesoscopic image properties (Afchar et al., 2018). **EfficientNet** (Tan & Le, 2019) subsequently became the empirically dominant backbone for this task: every top solution in Meta's DeepFake Detection Challenge (DFDC) used pretrained EfficientNets, and the first-place entry was an ensemble of seven EfficientNet-B7 models (Seferbekov, 2020). A second strand of work explicitly targets generalization: **Face X-Ray** detects the blending boundary common to face swaps rather than method-specific artifacts (Li, Bao, et al., 2020); **LipForensics** keys on semantic mouth-motion irregularities (Haliassos et al., 2021); and frequency-domain detectors exploit the systematic spectral artifacts left by GAN upsampling (Frank et al., 2020).


### The generalization gap (the opportunity)

A consistent and striking pattern runs through the literature: strong in-distribution detectors collapse on unseen generators and datasets. An FF++-trained Xception detector falls from 99.7 AUC on FF++ to 48.2 AUC on Celeb-DF, below the 50% chance line for a balanced AUC (Li, Yang, et al., 2020). Under a leave-one-manipulation-out protocol on FF++, an Xception baseline averages only 77.9% AUC on the held-out method and drops to 51.2% AUC on unseen FaceSwap, versus 97.1% for a generalization-oriented design (Haliassos et al., 2021). The effect extends across generator families: detectors trained on GAN imagery average ~97.4% AUROC on GANs but only ~78.6% on diffusion-generated images, catching just 26.3% of diffusion samples (Ricker et al., 2024). Even Face X-Ray, designed for transfer, reports 80.92 AUC on DFDC and 80.58 on Celeb-DF, recovering, but not closing, the drop (Li, Bao, et al., 2020).

The shape is always the same: 95-99% in-distribution decaying to roughly 48-80% on an unseen generator, frequently near chance for the hardest cases. Yet detectors are still most often summarized by a single in-distribution number, leaving cross-generator transfer under-characterized (Yan et al., 2023). **This project's opportunity is to measure that transfer directly** for conventional CNN detectors, using an explicit held-out-manipulation protocol, and to report the gap as a first-class result rather than a footnote.


## 4. Methodology


### Models and tools

We train two ImageNet-pretrained CNN classifiers and fine-tune them (transfer learning) rather than training from scratch, the standard, compute-efficient approach for this task and the recommended practice for the course:

- **XceptionNet** (Chollet, 2017): a depthwise-separable-convolution CNN; the canonical FF++ detector, chosen so our results are directly comparable to the founding benchmark.
- **EfficientNet** (Tan & Le, 2019): a compound-scaled CNN (e.g., B0/B4); the proven best-in-class deepfake backbone from the DFDC competition and a stronger, more modern comparison point.

Both are single-frame CNN classifiers, exactly the "conventional CNN detector" class the project sets out to characterize. Running both also adds a within-project axis: whether the generalization gap is architecture-specific or intrinsic to the approach (RQ3). The implementation stack is PyTorch with mixed-precision training, scikit-learn / torchmetrics for metrics, and version-controlled YAML configs with fixed seeds and checkpoint/resume so runs are reproducible and portable across the team's heterogeneous hardware.


### Generative component: a self-synthesized unseen generator

Alongside the two discriminative detectors, the project **implements a generative model** in the pipeline. We take a **pretrained SimSwap** face-swap generator (Chen et al., 2020), an efficient, subject-agnostic learned face swap that transfers a source identity onto a target face using InsightFace identity embeddings, and use it to synthesize a set of fake videos from FF++'s real ("pristine") source footage. This produces a **fifth manipulation category, entirely of our own making**, which the detectors never see during training. We *use* the pretrained generator off-the-shelf rather than training or fine-tuning a generator from scratch: building a generator is weeks of GAN/diffusion work outside this project's scope, whereas running a pretrained one is a bounded, reproducible step that still puts a real generative model, with its own artifacts and identity-blending fingerprint, inside the pipeline. Because SimSwap is a modern GAN-based swap, it sits in a **different generator family** from FF++'s DeepFakes (autoencoder swap), FaceSwap (computer-graphics swap), Face2Face and NeuralTextures (reenactment), making it a principled additional "unseen generator" for the same transfer question the rest of the project asks.


### Dataset

**Primary: FaceForensics++ (Rössler et al., 2019).** FF++ contains 1,000 real ("pristine") video sequences sourced from YouTube and 4,000 manipulated videos, 5,000 in total, created by applying four manipulation methods to each of the 1,000 originals. The four methods form a clean 2×2 design that directly supports our transfer hypotheses:

| Method | Manipulation type | Synthesis family |
| --- | --- | --- |
| DeepFakes | Identity face-swap | Learning-based (autoencoder) |
| FaceSwap | Identity face-swap | Computer-graphics-based |
| Face2Face | Facial reenactment | Computer-graphics-based |
| NeuralTextures | Facial reenactment | Learning-based (neural rendering) |

We use the **c23 (high-quality, H.264 QP 23)** release, roughly 10 GB of originals plus 10 GB of manipulated video (~20 GB combined), which is the most common reporting point and fits the team's storage and preprocessing budget. Videos are H.264-encoded MP4 centered on a single largely frontal face at varying resolution (≥480p). FF++ ships an official **720 / 140 / 140** train/validation/test split defined over the original source sequences, so the same identity never crosses splits and our splits are **identity-disjoint by construction**, the standard guard against the most common silent leakage bug in this domain. At the video level the data is fake-heavy (~1,000 real vs. ~4,000 fake, 1:4); we address this by constructing balanced per-method binary tasks (1,000 real vs. 1,000 fake) and using class-weighted sampling. The official source requires a signed Terms-of-Use request form; community Kaggle mirrors of the c23 set also exist (we will note provenance in the final report).

**Self-generated: SimSwap fakes (Chen et al., 2020).** We synthesize our own fake set by running the pretrained SimSwap generator on a sample of FF++ real source videos, swapping identities among them so the fakes share FF++'s capture conditions, resolution, and compression but carry a **new generator's fingerprint**. This set is produced by us, held out of all training, and passed through the identical face-crop pipeline as everything else. It becomes a fifth column in the evaluation, a self-produced unseen generator, and is the concrete generative component of the project (RQ4).

**Stretch: DFDC Preview (Dolhansky et al., 2019).** For an optional cross-dataset test we use the DFDC Preview subset (~5,000 clips from 66 actors, two undisclosed face-swap methods, in-the-wild capture), not the full ~470 GB DFDC set. Its generators, capture conditions, and compression are disjoint from FF++, making it a genuine unseen-generator stress test.


### Preprocessing

Following the community-standard pipeline (DeepfakeBench; Yan et al., 2023), we run face detection and alignment with dlib or MTCNN, crop with a ~1.3× margin around the face, resize to each model's native input (299×299 for Xception; 224-256 for EfficientNet), and normalize with ImageNet statistics to match the pretrained weights. We uniformly sample 16-32 frames per video and **cache the cropped faces to disk once** (as .npy/.pt with a manifest index), so preprocessing is run a single time and shared across teammates and all training runs, removing per-epoch video decoding, the project's binding time constraint. The self-generated SimSwap videos are cropped and cached through this same pipeline, so the fifth-column fakes are pixel-for-pixel comparable to the FF++ crops the detectors were trained on.


### Experimental design

The core protocol is a rotating **leave-one-manipulation-out (LOMO)** evaluation. For each of the four FF++ methods we train on the other three (plus real) and test on the held-out fourth, rotating the held-out method through all four. With two backbones this yields 4 × 2 = 8 cross-generator training runs, plus in-distribution baselines for reference. Results are assembled into a **transfer matrix** whose diagonal is in-distribution performance and whose off-diagonal is cross-generator performance. We note one honest methodological point: many published cross-manipulation numbers train on a single method, whereas our train-on-three design provides more source diversity and should therefore narrow the gap somewhat, we will state this explicitly so our numbers are not mis-compared (Section 5).

The **self-generated SimSwap set enters purely at test time**: every trained detector is additionally scored on it, adding a fifth "unseen generator" column to each detector's row in the transfer matrix without adding any training runs (the eight remain 2 backbones × 4 held-out FF++ folds). Because SimSwap is never in any training split, its column is unseen for all eight models by construction, the cleanest possible generalization test, since the generator is one we introduced ourselves and controls for FF++'s capture domain.


### Evaluation metrics

**AUC / AUROC** is the primary metric (threshold-independent and robust to class imbalance), reported at both frame and video level (video score = mean of frame probabilities). We also report accuracy, precision, recall, F1, and confusion matrices per split, at fixed c23 compression. Crucially, seen and unseen results are reported **separately**, with an explicit ΔAUC (seen − unseen) column; the unseen results are never folded into a single average.


### Analysis techniques

Beyond metrics, we analyze the transfer matrix for best/worst pairs and test the swap-versus-reenactment and graphics-versus-learning hypotheses, and we produce Grad-CAM overlays to visualize which spatial regions and artifacts each detector relies on for each manipulation method.


### Workflow and timeline

The project follows a sequential pipeline scoped so multiple members work in parallel (training folds and backbones are embarrassingly parallel). Indicative schedule:

| Week | Phase | Output |
| --- | --- | --- |
| 1 | Data acquisition & inventory | FF++ c23 on disk; verified per-method counts |
| 2 | EDA & identity/quality audit | Class balance, identity coverage, quality stats |
| 3 | Preprocessing pipeline | Cached face crops + manifest, portable across machines |
| 4 | Cross-generator split design | 4 LOMO manifests; identity-disjoint audit passes |
| 5-6 | CNN training | 8 trained models (2 backbones × 4 held-out folds) |
| 7 | Evaluation & metrics | Transfer matrix, seen-vs-unseen gap; (stretch) DFDC |
| 8 | Analysis & final report | Artifact analysis, Grad-CAM, written report |


## 5. Expected Outcomes

We will deliver two trained CNN deepfake detectors (EfficientNet and XceptionNet), a reproducible preprocessing-and-evaluation pipeline on FF++, and, as the project's central contribution, a **cross-generator transfer matrix** with an explicit seen-versus-unseen analysis.

Grounded in the literature, we anticipate the following, which also defines our evaluation targets. In-distribution performance should reach roughly **0.96 AUC** at c23 (cf. DeepfakeBench: Xception 0.9637, EfficientNet-B4 0.9567; Yan et al., 2023), consistent with the FF++ paper's 95.73% accuracy. On held-out (unseen) manipulations we expect performance to drop substantially, published cross-manipulation results fall to roughly **0.55-0.85 AUC**, a 15-40 AUC-point gap, with an Xception baseline as low as 51.2% on unseen FaceSwap (Haliassos et al., 2021). Because our LOMO protocol trains on three methods rather than one, we expect our held-out AUCs to land modestly higher than those single-source figures, while still exhibiting a clear and measurable gap.

We further expect a structured transfer pattern: **NeuralTextures** the hardest method to detect when held out (it is the subtlest, partial manipulation and the weakest cell even in-distribution), **FaceSwap** the weakest training source (its graphics-based artifacts transfer poorly to learned generators), and the **DeepFakes ↔ Face2Face** pair transferring comparatively well. For the optional cross-dataset stretch, an FF++-trained detector is expected to reach only ~0.70 AUC on DFDC (cf. DeepfakeBench Xception 0.7077; Yan et al., 2023). Honest negative findings, large gaps, methods that fail to transfer, are treated as successful outcomes, because the contribution is an accurate characterization of how standard CNN detectors generalize, not a single top-line accuracy.


## 6. Team Contribution

Responsibilities are distributed so that every stage, data preparation, preprocessing, modeling, evaluation, and reporting, has a clear primary owner, while the eight training runs (two backbones × four held-out folds) are shared across members. The Final Report is written collaboratively, with each member drafting the sections tied to their area and reviewing the others. The table below is the planned distribution; exact assignments will be confirmed at project kickoff.

| Member | Primary area | Key responsibilities |
| --- | --- | --- |
| Dominic Rivas | Coordination, evaluation & analysis | Manage timeline; define the cross-generator (LOMO) evaluation protocol; build the transfer matrix; lead spatial-artifact analysis and figures. |
| Jonathan Jude Regalado | Exploratory data analysis | Characterize FF++ (class balance, identity coverage, quality); audit identity overlap across manipulation methods to validate the splits. |
| Lyxelis Rodriguez Navarro | Preprocessing pipeline | Build frame-extraction, face-detection, cropping and normalization; manage the cached feature store shared across the team. |
| Obinna Okonkwo | Model implementation & training | Implement EfficientNet and XceptionNet detectors; run training and checkpointing; maintain reproducible configs across machines. |
| Sagar Ayare | Documentation, reporting & cross-dataset stretch | Lead literature review and report writing/editing; own reproducibility documentation; drive the optional DFDC cross-dataset evaluation. |


## References

1. Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: A compact facial video forgery detection network. IEEE International Workshop on Information Forensics and Security (WIFS). https://arxiv.org/abs/1809.00888
2. Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1251-1258. https://arxiv.org/abs/1610.02357
3. Dolhansky, B., Howes, R., Pflaum, B., Baram, N., & Ferrer, C. C. (2019). The Deepfake Detection Challenge (DFDC) Preview dataset. https://arxiv.org/abs/1910.08854
4. Dolhansky, B., Bitton, J., Pflaum, B., Lu, J., Howes, R., Wang, M., & Ferrer, C. C. (2020). The DeepFake Detection Challenge (DFDC) dataset. https://arxiv.org/abs/2006.07397
5. Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D., & Holz, T. (2020). Leveraging frequency analysis for deep fake image recognition. Proceedings of the 37th International Conference on Machine Learning (ICML), 3247-3258. https://arxiv.org/abs/2003.08685
6. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27, 2672-2680. https://arxiv.org/abs/1406.2661
7. Haliassos, A., Vougioukas, K., Petridis, S., & Pantic, M. (2021). Lips don't lie: A generalisable and robust approach to face forgery detection (LipForensics). Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5039-5049. https://arxiv.org/abs/2012.07657
8. Li, L., Bao, J., Zhang, T., Yang, H., Chen, D., Wen, F., & Guo, B. (2020). Face X-Ray for more general face forgery detection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5001-5010. https://arxiv.org/abs/1912.13458
9. Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). Celeb-DF: A large-scale challenging dataset for DeepFake forensics. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 3207-3216. https://arxiv.org/abs/1909.12962
10. Naffi, N. (2025). Deepfakes and the crisis of knowing. UNESCO. https://www.unesco.org/en/articles/deepfakes-and-crisis-knowing
11. Ricker, J., Damm, S., Holz, T., & Fischer, A. (2024). Towards the detection of diffusion model deepfakes. Proceedings of the 19th International Conference on Computer Vision Theory and Applications (VISAPP). https://arxiv.org/abs/2210.14571
12. Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 1-11. https://arxiv.org/abs/1901.08971
13. Seferbekov, S. (2020). DFDC winning solution (1st place, DeepFake Detection Challenge). GitHub. https://github.com/selimsef/dfdc_deepfake_challenge
14. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. Proceedings of the 36th International Conference on Machine Learning (ICML), 6105-6114. https://arxiv.org/abs/1905.11946
15. U.S. Government Accountability Office. (2024). Science & tech spotlight: Combating deepfakes (GAO-24-107292). https://www.gao.gov/products/gao-24-107292
16. Yan, Z., Zhang, Y., Yuan, X., Lyu, S., & Wu, B. (2023). DeepfakeBench: A comprehensive benchmark of deepfake detection. Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track. https://arxiv.org/abs/2307.01426
17. Zupan, C. (2025). (Deep)fake news: Recent data reveals gaps between perception and reality. Mimecast. https://www.mimecast.com/blog/deepfake-news-recent-data-reveals-gaps-between-perception-and-reality/

