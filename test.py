import argparse
import json
import mxnet as mx
import os
from config import config

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="Name of checkpoints directory")
    parser.add_argument("--prefix", required=True, help="Name of model")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch to load")
    args = parser.parse_args()

    # Load means
    means = json.loads(open(config.MEAN_PATH).read())

    # Test image iterator
    test_iter = mx.io.ImageRecordIter(
        path_imgrec=config.TEST_MX_REC,
        data_shape=(3, 224, 224),
        batch_size=config.BATCH_SIZE,
        mean_r=means["R"],
        mean_g=means["G"],
        mean_b=means["B"]
    )

    # Load model
    checkpoints_path = os.path.sep.join(["output", args.checkpoints, args.prefix])
    model = mx.model.FeedForward.load(checkpoints_path, args.epoch)

    # Compile model
    model = mx.model.FeedForward(
        ctx=[mx.gpu(0)],
        symbol=model.symbol,
        arg_params=model.arg_params,
        aux_params=model.aux_params
    )

    metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5)]
    rank1, rank5 = model.score(test_iter, eval_metric=metrics)

    print(f"[INFO] Rank 1 Accuracy: {rank1 * 100:.2f}")
    print(f"[INFO] Rank 5 Accuracy: {rank5 * 100:.2f}")




if __name__ == '__main__':
    main()
