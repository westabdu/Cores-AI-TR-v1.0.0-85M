# Cores-AI TR - 152M Asil Model - GTX 1650 (4GB VRAM)
# Faz B: Asil egitim, gradient checkpointing SART
# Kullanim: python train.py config/train_cores_152m.py
# Devam:    python train.py config/train_cores_152m.py --init_from=resume

out_dir = 'out-cores-152m'
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = True
wandb_log = False

# Veri
dataset = 'turkce'

# Model - ~152M parametre (16 layer, 12 head, 768 dim)
# NOT: 12 layer + bias=False = ~123M. 152M icin 16 layer gerekli.
n_layer = 16
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Batch - VRAM SINIRINDA
batch_size = 1                          # Minimum
block_size = 256                        # VRAM icin kisitli (512 sigmaz)
gradient_accumulation_steps = 80        # Efektif batch = 1 x 80 = 80

# Optimizer - LR biraz dusuruldu (152M icin 6e-4 yuksek)
learning_rate = 5e-4
max_iters = 150000                      # ~3B token gecisi (~10 epoch)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 150000
min_lr = 5e-5                           # learning_rate / 10

# Sistem - GTX 1650
device = 'cuda'
dtype = 'float16'                       # bfloat16 desteklenmiyor!
compile = False                         # Turing (sm_75) sorunlu

# Gradient checkpointing - 152M ICIN ZORUNLU
use_gradient_checkpointing = True
