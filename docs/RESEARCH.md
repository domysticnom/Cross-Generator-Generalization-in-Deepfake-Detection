# Research and method selection

> Items marked **[team]** need numbers filled in from our own runs or from the cited
> papers before the final report. Do not quote figures we have not verified.

## 1. Objectives

The task is binary image classification (real vs fake) on cropped face frames, framed as a
**generalization** study rather than an accuracy contest.

Concretely we answer three questions:

1. Which FF++ manipulation methods are hardest for a CNN detector to recognise when unseen?
2. How far does detection accuracy fall when a detector trained on three methods is tested
   on the fourth?
3. Does the architecture (EfficientNet vs XceptionNet) change cross-generator behaviour, or
   do both degrade similarly?

Success is a populated transfer matrix with seen and unseen reported separately, not a
single top-line score.

## 2. Literature review

**FaceForensics++ (Rössler et al., ICCV 2019).** Defines the corpus and the four
manipulation methods we use, and have established XceptionNet as the reference detector for the
benchmark. 
- Key takeaway: detectors do well in-distribution, which is why an in-distribution number alone is not an interesting result.

**Xception (Chollet, CVPR 2017).** Depthwise-separable convolutions; the canonical FF++
detection baseline (our baseline as well), so including it keeps our numbers comparable to published work.

**EfficientNet (Tan & Le, ICML 2019).** Compound scaling of depth/width/resolution, giving
markedly better accuracy per Floating Point Operation (FLOP) than earlier CNNs.

**DFDC (Dolhansky et al., 2020).** A different corpus with different generators and capture
conditions; reserved as an optional cross-dataset stretch test.
- Although we have yet to receive any response from Meta in relation to accessing teh DFDC dataset.

**Generative background (Goodfellow et al., 2014).** GAN-based synthesis is what makes
modern face manipulation possible, and newer generator families keep appearing — which is
precisely why generalization to an unseen generator, not in-distribution accuracy, is the
question worth asking.

## 3. Benchmarking the candidate backbones

Both candidates are ImageNet-pretrained and available in `timm`, so both are adopted via
transfer learning rather than trained from scratch.

| | EfficientNet-B0 | EfficientNet-B4 | Xception |
|---|---|---|---|
| Parameters | ~5.3 M | ~19 M | ~22.9 M |
| Native input | 224² | 380² (we run 224²) | 299² |
| Pretrained in `timm` | yes | yes | yes |
| Relative train cost | lowest | moderate | highest |
| Role here | quick trials | reported EfficientNet run | FF++ reference baseline |
| In-distribution AUC | **[team]** | **[team]** | **[team]** |
| Unseen AUC | **[team]** | **[team]** | **[team]** |

Selection rationale against the criteria:

>(B0 - B4 refers to EfficientNet model variants (aka "scaling levels") from the EfficientNet paper by Tan & Le (2019). EfficientNet was designed as a family of models that use the same architecture but are scaled up or down systematically. These span from EfficientNet-B0 ... EfficientNet-B7 the largest original variant). 

- **Accuracy / performance.** Xception is the published FF++ baseline; EfficientNet matches
  or beats comparable CNNs at lower cost. Filling the two AUC rows above from our own runs
  is what turns this from a claim based on literature into our result.
- **Computational efficiency.** EfficientNet-B4  reaches useful accuracy at roughly the
  parameter budget of Xception, and B0 is small enough for laptop-scale sanity runs.
- **Scalability.** The B0→B4 family lets us trade cost for accuracy without changing any
  code, since the backbone is a config field.
- **Pretrained availability.** Both load ImageNet weights from `timm` through the same
  `build_model` call, so the two architectures share one pipeline and any difference in
  cross-generator behaviour is attributable to the architecture rather than the setup.

**Rejected alternatives.** Training a detector from scratch (not enough data or compute for
the timeline); video/temporal models such as 3D CNNs or transformers (our unit of analysis
is the frame, and video-level scores are obtained by averaging frame scores instead).

## 4. Preliminary experiments

Small-scale checks run before committing to the full protocol.

**Face detection viability.** `data/preprocess.py` logs a per-clip detection rate to
`data/manifests/detection_log.csv` and prints a real-vs-fake summary. A subset run reached
~100% detection with comparable real and fake rates, confirming that cropping does not
silently bias one class. This matters: a detector that only sees faces it could crop would
otherwise inherit the crop step's own bias.

**Pipeline feasibility on a subset.** A reduced run (a few clips per method, 4 frames each)
was taken end to end — preprocess → splits → train → evaluate — producing a well-formed
results JSON. Purpose was to validate plumbing and runtime, not accuracy; the subset has
too few test identities for its numbers to mean anything.

**Device portability.** `resolve_device()` selects CUDA, then Apple MPS, then CPU, so the
same code runs on the GPU boxes and on laptops. Verified on both.

**[team] To add:** the full-scale in-distribution baseline for one fold, which sets the
upper-bound reference the unseen numbers are measured against.

## 5. What this justifies

The protocol that follows from the above: ImageNet-pretrained EfficientNet and XceptionNet,
fine-tuned identically, evaluated leave-one-manipulation-out across the four FF++ methods,
scored with ROC-AUC at frame and video level, reported seen-vs-unseen separately.

Implementation details are in [`MODEL.md`](MODEL.md); the module contracts are in
[`INTERFACES.md`](INTERFACES.md).

## References

- Chollet, F. (2017). *Xception: Deep learning with depthwise separable convolutions.* CVPR.
- Dolhansky, B., et al. (2020). *The DeepFake Detection Challenge (DFDC) dataset.* Meta AI.
- Goodfellow, I., et al. (2014). *Generative adversarial nets.* NeurIPS.
- Rössler, A., et al. (2019). *FaceForensics++: Learning to detect manipulated facial
  images.* ICCV.
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking model scaling for convolutional
  neural networks.* ICML.
