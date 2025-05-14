import random, shutil, os, pathlib

# ---------- 配置一下自己的路径 ----------
SRC_TRAIN = r"D:/Pycharm/WaterMark/HiDDeN/datasets/coco2014/train_class"
SRC_VAL   = r"D:/Pycharm/WaterMark/HiDDeN/datasets/coco2014/val_class"
DST_ROOT  = r"D:/Pycharm/WaterMark/HiDDeN"
N_TRAIN   = 10000
N_VAL     = 1000
# ---------------------------------------

# 1) 创建目标目录
os.makedirs(os.path.join(DST_ROOT, "train", "train_class"), exist_ok=True)
os.makedirs(os.path.join(DST_ROOT, "val",   "val_class"),   exist_ok=True)

# 2) 收集文件列表
train_files = list(pathlib.Path(SRC_TRAIN).glob("*.jpg"))
val_files   = list(pathlib.Path(SRC_VAL).glob("*.jpg"))

assert len(train_files) >= N_TRAIN and len(val_files) >= N_VAL, "源文件不足"

# 3) 随机采样（不放回）
sel_train = random.sample(train_files, N_TRAIN)
sel_val   = random.sample(val_files,   N_VAL)

# 4) 拷贝
for f in sel_train:
    shutil.copy2(f, os.path.join(DST_ROOT, "train", "train_class", f.name))

for f in sel_val:
    shutil.copy2(f, os.path.join(DST_ROOT, "val", "val_class", f.name))

print("Done!  train:", len(sel_train), " val:", len(sel_val))
