# Dataset Access

How this project acquires its datasets, which route the team chose, the license
terms we operate under, and the layout every teammate's copy must end up in. The
primary corpus (FaceForensics++ c23) is used now. DFDC is documented here as a
deferred, scope-gated stretch and is not acquired in Phase 1.

## FaceForensics++ c23 (primary, used now)

FaceForensics++ (FF++) is the shared training corpus. We use the **c23** (HQ,
H.264 constant-rate-quantization 23, "visually lossless") compression level only.
Other compressions (raw/c0, c40) and the later FaceShifter method are out of
scope and are not downloaded or normalized.

### Expected contents (c23)

| Item | Value |
|------|-------|
| Real (pristine) clips | 1000 |
| Manipulated clips | 4 methods x 1000 = 4000 |
| Total clips | 5000 |
| Methods | DeepFakes, Face2Face, FaceSwap, NeuralTextures |
| Compression | c23 (H.264, QP 23) |
| Container | H.264-encoded MP4 |
| Approx size | ~10 GB per category, ~20 GB combined |
| Official split | 720 train / 140 val / 140 test, by source sequence |

The four methods form a clean 2x2: swap = {DeepFakes (learned), FaceSwap
(graphics)}, reenactment = {Face2Face (graphics), NeuralTextures (learned)}. The
same 1000 source sequences feed all four methods, and the official split is keyed
on source-sequence id, so identities stay disjoint across train/val/test.

### License

FF++ is released under the **FaceForensics Terms of Use** (research /
non-commercial). The repository code is MIT, but the *videos* are governed by the
Terms of Use. Do not redistribute the videos, and keep them git-ignored (see
`data/README.md`). The official Google-form route is the license-clean path; the
community Kaggle mirror is a convenience re-upload and does not replace agreeing
to the Terms of Use.

### Route A: official (Google form + download script)

1. Request access via the FaceForensics Google form
   (https://github.com/ondyari/FaceForensics). Approval has lead time, so request
   early.
2. On approval you receive `faceforensics_download_v4.py`. Run it to pull the
   c23 originals and the four manipulation methods into a source tree.
3. Normalize that tree into the repo layout:

   ```
   python data/download_ffpp.py --route official --source-dir <downloaded_ffpp_root>
   ```

Pros: license-clean, complete, canonical. Cons: approval lead time.

### Route B: community Kaggle mirror

A community re-upload of the c23 set exists on Kaggle
(`xdxd003/ff-c23`; an alternate is `hungle3401/faceforensics`). This needs an
authenticated Kaggle CLI (`KAGGLE_USERNAME` / `KAGGLE_KEY` from
kaggle.com -> Account -> Create New API Token).

```
python data/download_ffpp.py --route kaggle --kaggle-dataset xdxd003/ff-c23
```

**Integrity caveat.** Kaggle mirrors are unofficial re-uploads. They may be
incomplete, subset a different compression, or reshape directories. Treat the
mirror as untrusted input: the inventory step (plan 01-02) enforces exactly 1000
clips per method and an ffprobe decode check before the corpus is trusted. Prefer
Route A when the approval timeline allows.

### Selected route

**Route B (Kaggle mirror `xdxd003/ff-c23`) is the team's working default** for
speed, with the inventory integrity gate (plan 01-02) standing in for the
license-clean guarantees of Route A. Any teammate who has official access should
prefer Route A; both routes normalize to the identical layout below, so the two
are interchangeable downstream.

### Native FF++ layout (as downloaded)

```
<ffpp_root>/
  original_sequences/youtube/c23/videos/*.mp4          -> real
  manipulated_sequences/Deepfakes/c23/videos/*.mp4     -> DeepFakes
  manipulated_sequences/Face2Face/c23/videos/*.mp4     -> Face2Face
  manipulated_sequences/FaceSwap/c23/videos/*.mp4      -> FaceSwap
  manipulated_sequences/NeuralTextures/c23/videos/*.mp4 -> NeuralTextures
```

### Normalized layout (produced by download_ffpp.py)

```
data/raw/
  real/*.mp4
  DeepFakes/*.mp4
  Face2Face/*.mp4
  FaceSwap/*.mp4
  NeuralTextures/*.mp4
```

`download_ffpp.py` only relocates videos into `data/raw/<method>/`; it never
renames them. The native filename stem is preserved verbatim (for example
`033_097.mp4` stays `033_097.mp4`) because `data/preprocess.py` derives
`clip_id` from the filename stem and `source_id` from the part before the first
underscore, and the Phase 4 identity-disjoint split audit keys on `source_id`.
Renaming a file here would silently corrupt splits two phases downstream. Only the
parent directory changes.

## DFDC (deferred, scope-gated Phase 7 stretch)

**DEFERRED. Not acquired in Phase 1.** DFDC is the optional cross-dataset
"unseen generator" test. It is documented here only so the access path is
identified; nothing below is downloaded during Phase 1.

- **Use the DFDC Preview subset only** (~5000 clips, 66 actors, two undisclosed
  face-swap methods). Its generators are disjoint from the FF++ four, and its
  in-the-wild capture and degradation profile make it a genuine unseen test.
- **The full DFDC set (~470 GB) is out of scope.** Download and face-cropping
  time, not GPU, is the binding constraint for a semester. Only a subset would
  ever be used.
- **Access path:** accept the Kaggle competition rules for the DeepFake Detection
  Challenge, then pull the Preview via the Kaggle API
  (`KAGGLE_USERNAME` / `KAGGLE_KEY`, same credentials as the FF++ mirror route).
- **License:** governed by the Kaggle competition / DFDC dataset terms; research
  use, no redistribution.

This section satisfies ROADMAP Phase 1 Success Criterion 3 (the cross-dataset
subset selection and access path are identified) without pulling DFDC into
Phase 1 scope.

## Provenance

`data/download_ffpp.py` appends a record of every acquisition run (route, source
id or source directory, UTC timestamp, and per-method counts) to
`data/PROVENANCE.md`, so each teammate's copy is auditable and comparable. The
raw videos and `data/PROVENANCE.md` are git-ignored; only code and small index
files are committed. The inventory step (plan 01-02) produces the committed
`data/manifests/ff_inventory.csv`, which is the integrity gate that must pass
before the corpus is trusted for EDA, preprocessing, and the eight training runs.
