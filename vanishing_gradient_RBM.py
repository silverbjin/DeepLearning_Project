import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# -----------------------------
# 0) 유틸
# -----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sample_bernoulli(p, rng):
    return (rng.random(p.shape) < p).astype(np.float32)

def show_grid(images, title, nrow=4, ncol=8, img_shape=(8, 8)):
    plt.figure(figsize=(ncol * 1.2, nrow * 1.2))
    for i in range(nrow * ncol):
        ax = plt.subplot(nrow, ncol, i + 1)
        ax.imshow(images[i].reshape(img_shape), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 1) RBM (Bernoulli-Bernoulli) - CD-1
# -----------------------------
class RBM:
    def __init__(self, n_vis, n_hid, lr=0.1, k=1, seed=0):
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.lr = lr
        self.k = k
        self.rng = np.random.default_rng(seed)

        # 작은 랜덤 초기화
        self.W = 0.01 * self.rng.standard_normal((n_vis, n_hid)).astype(np.float32)
        self.a = np.zeros((n_vis,), dtype=np.float32)  # visible bias
        self.b = np.zeros((n_hid,), dtype=np.float32)  # hidden bias

    def v_to_h_prob(self, v):
        return sigmoid(v @ self.W + self.b)

    def h_to_v_prob(self, h):
        return sigmoid(h @ self.W.T + self.a)

    def gibbs_step(self, v0):
        # v -> h -> v
        h_prob = self.v_to_h_prob(v0)
        h = sample_bernoulli(h_prob, self.rng)
        v_prob = self.h_to_v_prob(h)
        v = sample_bernoulli(v_prob, self.rng)
        return v, v_prob, h, h_prob

    def cd1_update(self, v_data):
        # Positive phase
        h0_prob = self.v_to_h_prob(v_data)
        h0 = sample_bernoulli(h0_prob, self.rng)

        # Negative phase (k-step Gibbs; 여기서는 k=1 기본)
        v = v_data.copy()
        for _ in range(self.k):
            v, v_prob, h, h_prob = self.gibbs_step(v)

        # Gradient (approx.)
        # dW ~ <v h>_data - <v h>_model
        pos = v_data.T @ h0_prob
        neg = v_prob.T @ h_prob

        batch_size = v_data.shape[0]
        self.W += self.lr * (pos - neg) / batch_size
        self.a += self.lr * np.mean(v_data - v_prob, axis=0)
        self.b += self.lr * np.mean(h0_prob - h_prob, axis=0)

        # reconstruction error(모니터링용)
        recon = v_prob
        err = np.mean((v_data - recon) ** 2)
        return err

    def reconstruct(self, v):
        h_prob = self.v_to_h_prob(v)
        v_prob = self.h_to_v_prob(h_prob)  # 확률 기반 재구성
        return v_prob

    def sample(self, n_samples=32, n_gibbs=200):
        v = sample_bernoulli(np.full((n_samples, self.n_vis), 0.5, dtype=np.float32), self.rng)
        for _ in range(n_gibbs):
            v, _, _, _ = self.gibbs_step(v)
        return v

# -----------------------------
# 2) 데이터 준비 (digits: 8x8)
# -----------------------------
digits = load_digits()
X = digits.data.astype(np.float32)  # (N, 64), 값 범위 0~16
X = X / 16.0                        # 0~1 로 정규화
# RBM은 Bernoulli를 가정하므로 확률로 보고 샘플링해서 이진화하면 데모가 더 "RBM스럽게" 보입니다.
rng = np.random.default_rng(123)
X_bin = (rng.random(X.shape) < X).astype(np.float32)

X_train, X_test = train_test_split(X_bin, test_size=0.2, random_state=42)

# -----------------------------
# 3) 학습
# -----------------------------
rbm = RBM(n_vis=64, n_hid=64, lr=0.15, k=1, seed=1)

batch_size = 128
epochs = 30
errs = []

for ep in range(1, epochs + 1):
    # 셔플
    idx = rng.permutation(len(X_train))
    X_train_shuf = X_train[idx]

    ep_err = 0.0
    n_batches = 0
    for i in range(0, len(X_train_shuf), batch_size):
        batch = X_train_shuf[i:i+batch_size]
        if len(batch) < 2:
            continue
        ep_err += rbm.cd1_update(batch)
        n_batches += 1

    ep_err /= max(1, n_batches)
    errs.append(ep_err)
    if ep % 5 == 0 or ep == 1:
        print(f"Epoch {ep:02d} | recon MSE: {ep_err:.4f}")

# -----------------------------
# 4) 시각화 ① 학습 곡선
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(np.arange(1, epochs+1), errs)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction MSE")
plt.title("RBM (CD-1) training curve")
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------
# 5) 시각화 ② '숨은 유닛이 보는 패턴' (가중치 필터)
#    W[:, j]를 8x8로 펼치면 '특징 검출기'처럼 보입니다.
# -----------------------------
n_show = 32
W = rbm.W  # (64, hidden)
# 보기 좋게 스케일 정규화
W_vis = W.copy()
W_vis = (W_vis - W_vis.min()) / (W_vis.max() - W_vis.min() + 1e-8)
show_grid(W_vis.T[:n_show], title="Hidden unit filters (W columns reshaped to 8x8)", nrow=4, ncol=8)

# -----------------------------
# 6) 시각화 ③ 원본 vs 재구성
# -----------------------------
n_compare = 32
orig = X_test[:n_compare]
recon = rbm.reconstruct(orig)

show_grid(orig,  title="Original (test) binaries", nrow=4, ncol=8)
show_grid(recon, title="Reconstruction (probabilities)", nrow=4, ncol=8)

# -----------------------------
# 7) 시각화 ④ Gibbs sampling으로 생성된 샘플
# -----------------------------
samples = rbm.sample(n_samples=32, n_gibbs=300)
show_grid(samples, title="Samples generated by RBM (Gibbs chain)", nrow=4, ncol=8)
