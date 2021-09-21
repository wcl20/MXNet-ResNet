import argparse
import json
import logging
import mxnet as mx
import os
from config import config
from core.nn import ResNet


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", required=True, help="Name of checkpoints directory")
parser.add_argument("--prefix", required=True, help="Name of model")
parser.add_argument("-s", "--start", type=int, default=0, help="Epoch to restart training.")
args = parser.parse_args()

os.makedirs(os.path.sep.join(["output", args.checkpoints, "logs"]), exist_ok=True)
logging.basicConfig(level=logging.DEBUG, filename=os.path.sep.join(["output", args.checkpoints, "logs", f"training_{args.start}.log"]), filemode="w")

# Load means
means = json.loads(open(config.MEAN_PATH).read())
batch_size = config.BATCH_SIZE * config.NUM_DEVICES

# Training image iterator
train_iter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 224, 224),
    batch_size=batch_size,
    rand_crop=True,
    rand_mirror=True,
    rotate=15,
    max_shear_ratio=0.1,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
    preprocess_threads=config.NUM_DEVICES * 2
)

# Validation image iterator
valid_iter = mx.io.ImageRecordIter(
    path_imgrec=config.VALID_MX_REC,
    data_shape=(3, 224, 224),
    batch_size=batch_size,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

# Optimizer
optimizer = mx.optimizer.SGD(learning_rate=1e-3, momentum=0.9, wd=0.0001, rescale_grad=1.0 / batch_size)

# Checkpoints
os.makedirs(os.path.sep.join(["output", args.checkpoints]), exist_ok=True)
checkpoints_path = os.path.sep.join(["output", args.checkpoints, args.prefix])

if args.start <= 0:
    # Build model
    print("[INFO] Building model ...")
    model = ResNet.build(config.NUM_CLASSES, stages=[3, 4, 6, 3], filters=[64, 256, 512, 1024, 2048])
    arg_params = None
    aux_params = None
else:
    model = mx.model.FeedForward.load(checkpoints_path, args.start)
    arg_params = model.arg_params
    aux_params = model.aux_params
    model = model.symbol

# Compile model
model = mx.model.FeedForward(
    ctx=[mx.gpu(i) for i in range(config.NUM_DEVICES)],
    symbol=model,
    initializer=mx.initializer.MSRAPrelu(),
    arg_params=arg_params,
    aux_params=aux_params,
    optimizer=optimizer,
    num_epoch=50,
    begin_epoch=args.start
)

# Callbacks
batch_end_callbacks = [mx.callback.Speedometer(batch_size, 250)]
epoch_end_callbacks = [mx.callback.do_checkpoint(checkpoints_path)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

# Train model
model.fit(
    X=train_iter,
    eval_data=valid_iter,
    eval_metric=metrics,
    batch_end_callback=batch_end_callbacks,
    epoch_end_callback=epoch_end_callbacks
)
