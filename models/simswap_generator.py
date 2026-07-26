import contextlib
import os
import sys

import cv2
import numpy as np
import torch


@contextlib.contextmanager
def _simswap_models(simswap_repo):
    """Temporarily make `models` resolve to SimSwap's package instead of ours.

    Both this project and the SimSwap repo ship a top-level `models` package. Ours
    is already in sys.modules by the time this file runs, so SimSwap's internal
    imports (`models.models`, and the relative `from .fs_model import ...` that
    create_model performs lazily) would otherwise resolve into our package and fail
    with ModuleNotFoundError.

    Everything imported while shadowed keeps working afterwards -- the classes and
    the constructed model object hold their own references; only the sys.modules
    entry is restored.
    """
    ours = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith("models.")}
    for k in ours:
        del sys.modules[k]
    if simswap_repo not in sys.path:
        sys.path.insert(0, simswap_repo)
    try:
        yield
    finally:
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[k]
        sys.modules.update(ours)


class SimSwapGenerator:
    # SimSwap's own download-weights.sh points at a OneDrive copy of the legacy
    # "antelope" insightface pack; that URL is dead (404). "antelopev2" is the same
    # family and insightface auto-downloads it from its GitHub releases. Only the
    # detector + 5-point landmarks are used here (the identity embedding comes from
    # SimSwap's own arcface_checkpoint.tar), so the pack choice only has to provide
    # bbox + kps.
    DEFAULT_FACE_PACK = "antelopev2"

    def __init__(self, weights_dir, simswap_repo, device=None, crop_size=224,
                 face_pack=DEFAULT_FACE_PACK):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # Set compute device to GPU if available and requested, otherwise default to CPU
        self.crop_size = crop_size  # Define the pixel dimensions for cropping face images

        # NAME COLLISION: this project has its own top-level `models` package (the
        # one this file lives in) and so does SimSwap. Simply putting the SimSwap
        # repo on sys.path is not enough -- `models` is already in sys.modules, so
        # `models.models` resolves into OUR package and fails. _simswap_models()
        # below shadows `models` with SimSwap's for as long as we need it.
        if simswap_repo not in sys.path:
            sys.path.insert(0, simswap_repo)

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "insightface is required for face detection/alignment. "
                "pip install insightface onnxruntime-gpu"
            ) from e

        # Only the detector is needed: we use bbox + 5-point kps for alignment, and
        # the identity embedding comes from SimSwap's own ArcFace. Restricting to
        # 'detection' skips loading glintr100 (260 MB) and 1k3d68 (143 MB).
        #
        # Provider choice: onnxruntime-gpu builds are pinned to a CUDA major version
        # and will hard-fail to load if it does not match what is installed (e.g.
        # a CUDA-13 build looking for cublasLt64_13.dll next to a CUDA-12 torch).
        # Probe CUDA and fall back to CPU rather than dying or spamming errors.
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            try:
                import onnxruntime as _ort
                if "CUDAExecutionProvider" in _ort.get_available_providers():
                    _ort.InferenceSession  # noqa: B018 - presence check only
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                pass

        self.face_app = FaceAnalysis(name=face_pack, root=weights_dir,
                                     allowed_modules=["detection"],
                                     providers=providers)
        self.face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(320, 320))

        # bound to a differently-named local first: inside a class body,
        # `crop_size = crop_size` makes the name local to that body, so the
        # right-hand side reads an unbound local and raises NameError. Names that
        # are only READ in a class body (weights_dir, self) resolve fine.
        _crop_size = crop_size

        # Stand-in for SimSwap's argparse options object. Every attribute below is
        # actually read by fsModel.initialize / BaseModel.initialize / create_model
        # on the inference path -- omitting any of them raises AttributeError.
        class _Opt:
            name = "people"
            gpu_ids = "0" if self.device == "cuda" else "-1"
            checkpoints_dir = weights_dir          # netG loads from <dir>/<name>/latest_net_G.pth
            isTrain = False
            Arc_path = os.path.join(weights_dir, "arcface_checkpoint.tar")
            crop_size = _crop_size
            which_epoch = "latest"                 # -> "latest_net_G.pth"
            resize_or_crop = "none"
            verbose = False
            load_pretrain = ""
            continue_train = False
            fp16 = False

        # SimSwap's arcface_checkpoint.tar is a PICKLED nn.Module, not a state dict
        # (fs_model.py does `self.netArc = torch.load(...)` and uses the object
        # directly). torch >= 2.6 defaults torch.load to weights_only=True, which
        # refuses to unpickle a module and breaks model construction. Restore the
        # old default just for the duration of create_model, then put it back so we
        # are not loosening unpickling globally for the rest of the process.
        _orig_load = torch.load

        def _load_full(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)

        torch.load = _load_full
        try:
            # create_model() itself does `from .fs_model import fsModel` lazily at
            # CALL time, so SimSwap's `models` package has to still be the one in
            # sys.modules while we call it -- not just while we import it.
            with _simswap_models(simswap_repo):
                import importlib
                create_model = importlib.import_module("models.models").create_model
                self.model = create_model(_Opt())
        finally:
            torch.load = _orig_load

# Transition the model parameters into evaluation mode to disable layers like dropout
        self.model.eval()

    def _align(self, img_bgr):
        faces = self.face_app.get(img_bgr)  # Detect all human faces present within the input BGR image frame
        if not faces:
            return None, None

         # Select the single largest face found in the image based on bounding box area
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Normalize, crop, and generate the transformation matrix for the chosen face
        aligned, m = insightface_align(img_bgr, face, self.crop_size)
        return aligned, m

# Preprocesses an aligned BGR face crop, resizes it to 112x112, and extracts an L2-normalized identity embedding vector using the ArcFace network.
    # SimSwap's ArcFace branch expects IMAGENET normalisation, not [-1,1]. See
    # test_one_image.py: transformer_Arcface = ToTensor() + Normalize(mean, std).
    _ARC_MEAN = (0.485, 0.456, 0.406)
    _ARC_STD = (0.229, 0.224, 0.225)

    def _identity_embedding(self, aligned_source_bgr):
        img = cv2.cvtColor(aligned_source_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
        mean = torch.tensor(self._ARC_MEAN).view(3, 1, 1)
        std = torch.tensor(self._ARC_STD).view(3, 1, 1)
        img = (img - mean) / std
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.model.netArc(torch.nn.functional.interpolate(img, size=(112, 112)))
            latent = latent / latent.norm(dim=1, keepdim=True)
        return latent

# Detect and align faces from both the source image and target frame, returning None if a face is missing in either image.
    def prepare_source(self, source_img_bgr):
        """Align + embed one source identity, for reuse across many target frames.

        A swap pair holds the source constant over every sampled frame of the target
        clip, so doing this per frame repeats an identical face detection (CPU-bound)
        and ArcFace pass 20x per pair for no benefit. Returns None if no face.
        """
        src_aligned, _ = self._align(source_img_bgr)
        if src_aligned is None:
            return None
        return self._identity_embedding(src_aligned)

    def swap(self, source_img_bgr, target_frame_bgr, source_latent=None):
        # pass source_latent from prepare_source() to skip re-detecting the source
        if source_latent is None:
            source_latent = self.prepare_source(source_img_bgr)
            if source_latent is None:
                return None
        latent = source_latent

        tgt_aligned, tgt_m = self._align(target_frame_bgr)
        if tgt_aligned is None:
            return None

        # The target ("att") branch takes a plain [0,1] tensor -- ToTensor() with the
        # Normalize line commented out in SimSwap's test_one_image.py. Mapping it to
        # [-1,1] here is what produced a flat grey patch instead of a face.
        tgt = cv2.cvtColor(tgt_aligned, cv2.COLOR_BGR2RGB)
        tgt = torch.from_numpy(tgt).permute(2, 0, 1).float().div(255)
        tgt = tgt.unsqueeze(0).to(self.device)

# Feed target face and source embedding into the generator network without tracking gradients
        with torch.no_grad():
            swapped = self.model.netG(tgt, latent)
        # netG's output lives in the same [0,1] space as its input: the reference
        # does `output * 255` with no de-normalisation step.
        swapped = swapped.squeeze(0).clamp(0, 1).mul(255)
        swapped = swapped.permute(1, 2, 0).byte().cpu().numpy()
        swapped_bgr = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)

        return paste_back(swapped_bgr, target_frame_bgr, tgt_m)


# InsightFace to normalize, rotate, and crop a face based on facial landmarks, returning the aligned face and its transformation matrix.
def insightface_align(img_bgr, face, crop_size):
    from insightface.utils import face_align
    aligned, m = face_align.norm_crop2(img_bgr, face.kps, crop_size)
    return aligned, m

# Blend the swapped face patch back into the original frame using an inverted transformation matrix and a Gaussian-blurred edge mask.
def paste_back(swapped_crop_bgr, original_frame_bgr, m):
    h, w = original_frame_bgr.shape[:2]
    inv_m = cv2.invertAffineTransform(m)
    warped = cv2.warpAffine(swapped_crop_bgr, inv_m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    mask = np.ones(swapped_crop_bgr.shape[:2], dtype=np.float32)
    mask = cv2.warpAffine(mask, inv_m, (w, h))
    mask = cv2.GaussianBlur(mask, (15, 15), 0)[..., None]

    out = warped.astype(np.float32) * mask + original_frame_bgr.astype(np.float32) * (1 - mask)
    return out.astype(np.uint8)
