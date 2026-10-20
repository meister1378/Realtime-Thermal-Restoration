"""PPFN-small-direct-v1 학습 스크립트 (실시간 복원 경로, 4.26M 파라미터).

이 폴더(정리/)의 구조:
    model/model.py      Restormer_PPFN 아키텍처 -- train/infer 공용, 한 곳에만 존재
    train/train.py       (이 파일) 학습 전용. 체크포인트를 저장하는 것 말고는
                          infer 쪽 코드를 전혀 참조하지 않음
    infer/infer.py        추론 전용. 학습 코드를 전혀 참조하지 않음 -- 체크포인트
                          파일(.pth) 하나로만 train과 연결됨
    infer/data_io.py       raw16 정규화 + 렌즈 왜곡 보정 (추론에서만 필요)
    prompts/              모델이 요구하는 prompt 임베딩(type7.npy, multi2.npy) --
                          HM-TIR 원 저장소 그대로, train/infer 공용
    checkpoints/           학습 결과 .pth 저장 위치

random init(랜덤 초기화)에서 HM-TIR 공개 데이터셋(NeurIPS 2025, PPFN 논문 데이터)
하나만으로 noise+motion blur+illumination 합성 열화를 학습한다. 이 프로젝트가 직접
촬영한 열화상 데이터는 전혀 필요 없다 -- 다운로드 링크는 이 폴더의 README.md 참고.

사용법:
    python train.py --clean-data-root /path/to/HM-TIR-parent --out-dir ../checkpoints
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'model'))
from model import Restormer_PPFN  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--n-steps', type=int, default=30000)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr-min', type=float, default=1e-6)
parser.add_argument('--patch', type=int, default=256)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--lambda-grad', type=float, default=5.0)
parser.add_argument('--eval-every', type=int, default=500)
parser.add_argument('--eval-n-samples', type=int, default=64)
parser.add_argument('--log-every', type=int, default=50)
parser.add_argument('--out-dir', default=os.path.join(_HERE, '..', 'checkpoints'))
parser.add_argument('--clean-data-root', default='./data',
                     help='parent of an "HM-TIR/imgs/*.png" folder (the public HM-TIR clean corpus)')
parser.add_argument('--prompts-dir', default=os.path.join(_HERE, '..', 'prompts'))
# architecture -- these defaults ARE ppfn_small_direct_v1; only change them
# to train a deliberately different variant (e.g. the 9.43M "medium" config
# used a 2/3-scale-up: dim=32, num_blocks=[3,4,4,6], num_refinement_blocks=3)
parser.add_argument('--dim', type=int, default=24)
parser.add_argument('--num-blocks', type=int, nargs=4, default=[2, 3, 3, 4])
parser.add_argument('--num-refinement-blocks', type=int, default=2)
# degradation knobs -- defaults are the validated recipe; do not change
# these unless you mean to train a genuinely different checkpoint
parser.add_argument('--stripe-scale', type=float, default=1.0)
parser.add_argument('--noise-kernel-size', type=int, default=5)
parser.add_argument('--noise-kernel-sigma', type=float, default=0.7)
parser.add_argument('--blur-strength-min', type=float, default=0.15)
parser.add_argument('--illum-strength-min', type=float, default=0.0)
parser.add_argument('--illum-none-frac', type=float, default=0.5)
parser.add_argument('--resume-from', default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(args.out_dir, exist_ok=True)

model = Restormer_PPFN(
    inp_channels=1, out_channels=1, dim=args.dim, num_blocks=args.num_blocks,
    num_refinement_blocks=args.num_refinement_blocks, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
).to(device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'model parameters: {n_params:.2f}M (dim={args.dim}, num_blocks={args.num_blocks}, '
      f'num_refinement_blocks={args.num_refinement_blocks})', flush=True)

if args.resume_from is not None:
    model.load_state_dict(torch.load(args.resume_from, map_location=device))
    print(f'warm-started from {args.resume_from}')
else:
    print('random init (no warm-start)')

model.train()

TYPEP = torch.from_numpy(np.load(os.path.join(args.prompts_dir, 'type7.npy'))[:3].astype(np.float32)).to(device)
MULTIP = torch.from_numpy(np.load(os.path.join(args.prompts_dir, 'multi2.npy')).astype(np.float32)).to(device)
TYPE_DIC = {"contrast": 0, "blur": 1, "noise": 2}
SCE_DIC = {"single": 0, "composited": 1}


def run_student(x01, batch_size):
    """Single-pass, noise-prompt call -- the real-time inference convention
    (see infer.py). Unlike PPFN-full's blur->noise 2-stage cascade, this
    model was trained to do noise+blur+illumination removal in one forward
    pass under this one prompt."""
    tp = TYPEP[TYPE_DIC['noise']].unsqueeze(0).expand(batch_size, -1, -1)
    mp = MULTIP[SCE_DIC['composited']].unsqueeze(0).expand(batch_size, -1)
    return model([x01, tp, mp])


class CharbonnierLoss(torch.nn.Module):
    """Smooth L1 variant: sqrt((pred-target)^2 + eps^2)."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()


criterion = CharbonnierLoss()
opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.9))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.n_steps, eta_min=args.lr_min)


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()


def psnr(pred, target):
    mse = ((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).mean().clamp_min(1e-10)
    return (-10 * torch.log10(mse)).item()


# --- synthetic degradation recipe: noise (fixed-pattern + stripe + optical
# nonuniformity) + motion blur + illumination(brightness/contrast), each
# independently "sometimes zero" so the model also sees mild/no degradation ---

def nonuniformity_optical(image, strength=1.0):
    h, w = image.shape
    idx_h = np.expand_dims(np.arange(1, h + 1), 1)
    idx_w = np.expand_dims(np.arange(1, w + 1), 0)
    delta = np.random.randint(15, 55 + 1) * strength
    ch, cw = np.random.randint(h), np.random.randint(w)
    p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
    p /= np.max(p)
    noise = np.cos(p * np.pi / 2) ** 4
    if random.random() < 0.5:
        image = np.clip(image.astype('float32') + noise.astype('float32') * delta, 0, 255)
    else:
        image = np.clip(image.astype('float32') + (1 - noise.astype('float32')) * delta, 0, 255)
    return image


def stripe_noise_2d(image, strength=1.0):
    h, w = image.shape
    lo, hi = 0.03 * args.stripe_scale, 0.07 * args.stripe_scale
    add_mag = 3 * args.stripe_scale
    g_col = np.random.randn(1, w) * (np.random.uniform(lo, hi) * strength)
    b_col = np.random.randn(1, w) * (np.random.rand() * add_mag * strength)
    g_row = np.random.randn(h, 1) * (np.random.uniform(lo, hi) * strength)
    b_row = np.random.randn(h, 1) * (np.random.rand() * add_mag * strength)
    noise = image * g_col + b_col + image * g_row + b_row
    return np.clip(image.astype('float32') + noise.astype('float32'), 0, 255)


def correlated_2d_noise(shape, std):
    white = np.random.randn(*shape).astype(np.float32)
    k = cv2.getGaussianKernel(args.noise_kernel_size, args.noise_kernel_sigma)
    kernel2d = k @ k.T
    blurred = cv2.filter2D(white, -1, kernel2d, borderType=cv2.BORDER_REFLECT)
    blurred = blurred / (blurred.std() + 1e-8)
    return blurred * std


def sample_noise_strength():
    r = random.random()
    if r < 0.2:
        return 0.0
    elif r < 0.6:
        return random.uniform(0.15, 0.4)
    else:
        return 1.0


def strong_motion_blur(img_u8, strength=1.0):
    if random.random() < 0.5:
        length = max(3, int(np.random.uniform(20, 90) * strength))
        if length % 2 == 0:
            length += 1
        angle = np.random.uniform(0, 360)
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        center = (length / 2 - 0.5, length / 2 - 0.5)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (length, length))
        kernel = kernel / (kernel.sum() + 1e-8)
        return cv2.filter2D(img_u8, -1, kernel, borderType=cv2.BORDER_REFLECT)
    else:
        sigma = max(0.5, np.random.uniform(3, 12) * strength)
        ksize = int(sigma * 6) | 1
        return cv2.GaussianBlur(img_u8, (ksize, ksize), sigma)


def sample_blur_strength():
    r = random.random()
    if r < 0.3:
        return 0.0
    else:
        return random.uniform(args.blur_strength_min, 1.0)


def illumination_degrade(img_u8, strength=1.0):
    x = img_u8.astype('float32')
    brightness_factor = np.random.uniform(0.2, 0.4) * strength
    contrast_factor = 1.0 - np.random.uniform(0.2, 0.8) * strength
    mean = x.mean()
    x = mean + (x - mean) * contrast_factor
    x = x * (1 + brightness_factor)
    return np.clip(x, 0, 255).astype('uint8')


def sample_illum_strength():
    r = random.random()
    if r < args.illum_none_frac:
        return 0.0
    else:
        return random.uniform(args.illum_strength_min, 1.0)


def synth_degrade(img_u8, noise_strength=1.0, blur_strength=0.0, illum_strength=0.0):
    x = img_u8
    if illum_strength > 0:
        x = illumination_degrade(x, illum_strength)
    if blur_strength > 0:
        x = strong_motion_blur(x, blur_strength)
    x = x.astype('float32')
    if noise_strength <= 0:
        return x.astype('uint8')
    x = nonuniformity_optical(x, noise_strength)
    x = stripe_noise_2d(x, noise_strength)
    x = x + correlated_2d_noise(x.shape, np.random.uniform(5, 15) * noise_strength)
    return np.clip(x, 0, 255).astype('uint8')


def load_gray(p):
    return np.array(Image.open(p).convert('L'), dtype=np.uint8)


print('loading HM-TIR clean corpus (ONLY data source)...')
clean_paths = sorted(glob.glob(f'{args.clean_data_root}/HM-TIR/imgs/*.png'))
if not clean_paths:
    raise SystemExit(
        f'no images found under {args.clean_data_root}/HM-TIR/imgs/ -- download the HM-TIR dataset '
        f'first (see README.md in this folder for the link) and point --clean-data-root at its parent.'
    )
random.shuffle(clean_paths)
n_val = int(len(clean_paths) * 0.2)
clean_val_paths, clean_train_paths = clean_paths[:n_val], clean_paths[n_val:]
print(f'  clean train images: {len(clean_train_paths)}  val: {len(clean_val_paths)} (from {len(clean_paths)} total)')


def sample_patch(pool):
    img = load_gray(pool[np.random.randint(len(pool))])
    h, w = img.shape
    if h < args.patch or w < args.patch:
        img = np.array(Image.fromarray(img).resize((max(w, args.patch), max(h, args.patch))))
        h, w = img.shape
    y0 = np.random.randint(0, h - args.patch + 1)
    x0 = np.random.randint(0, w - args.patch + 1)
    return img[y0:y0 + args.patch, x0:x0 + args.patch]


def sample_batch(pool, batch_size):
    xs, ys = [], []
    for _ in range(batch_size):
        clean = sample_patch(pool)
        degraded = synth_degrade(clean, noise_strength=sample_noise_strength(), blur_strength=sample_blur_strength(),
                                  illum_strength=sample_illum_strength())
        xs.append(degraded.astype(np.float32) / 255.0)
        ys.append(clean.astype(np.float32) / 255.0)
    x = torch.from_numpy(np.stack(xs)).unsqueeze(1).to(device)
    y = torch.from_numpy(np.stack(ys)).unsqueeze(1).to(device)
    return x, y


@torch.no_grad()
def run_validation():
    model.eval()
    psnrs = []
    for _ in range(args.eval_n_samples):
        x, y = sample_batch(clean_val_paths, 1)
        pred = run_student(x, 1).clamp(0, 1)
        psnrs.append(psnr(pred, y))
    model.train()
    return float(np.mean(psnrs))


print(f'training ({n_params:.2f}M, random init, noise+blur+illum): {args.n_steps} steps, patch={args.patch}, '
      f'batch={args.batch}, data={len(clean_train_paths)} train images')
loss_ema = None
best_val_psnr = 0.0
for step in range(args.n_steps):
    x, y = sample_batch(clean_train_paths, args.batch)

    opt.zero_grad()
    pred = run_student(x, args.batch)
    recon_loss = criterion(pred, y)
    grad = gradient_loss(pred, y)
    loss = recon_loss + args.lambda_grad * grad
    loss.backward()
    if all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
    else:
        print(f'step {step}: SKIPPED non-finite gradient', flush=True)
    sched.step()

    loss_val = loss.item()
    loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
    if step % args.log_every == 0:
        with torch.no_grad():
            train_psnr = psnr(pred, y)
        print(f'step {step}/{args.n_steps}: loss={loss_val:.4f} (ema={loss_ema:.4f}) recon={recon_loss.item():.4f} '
              f'grad={grad.item():.4f} train_PSNR={train_psnr:.2f} lr={sched.get_last_lr()[0]:.2e}', flush=True)

    if step % args.eval_every == 0 and step > 0:
        val_psnr = run_validation()
        print(f'  [VAL step {step}] PSNR={val_psnr:.4f} (best={best_val_psnr:.4f})', flush=True)
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), f'{args.out_dir}/ppfn_small_direct_bestPSNR.pth')

    if step % 2000 == 0 and step > 0:
        torch.save(model.state_dict(), f'{args.out_dir}/ppfn_small_direct_step{step}.pth')

torch.save(model.state_dict(), f'{args.out_dir}/ppfn_small_direct_final.pth')
print(f'done. saved final model to {args.out_dir}/ppfn_small_direct_final.pth')
