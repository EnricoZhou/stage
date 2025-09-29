import tensorflow as tf
import numpy as np
from PeerRead.dataset.dataset import make_buzzy_based_simulated_labeler, make_unprocessed_PeerRead_dataset, compose, make_extra_feature_cleaning

def write_example(writer, bow, index, treatment, outcome, beta1, y0, y1):
    features = {
        "bow": tf.train.Feature(float_list=tf.train.FloatList(value=bow.tolist())),
        "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
        "treatment": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(treatment)])),
        "outcome": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(outcome)])),
        "y0": tf.train.Feature(float_list=tf.train.FloatList(value=[float(y0)])),
        "y1": tf.train.Feature(float_list=tf.train.FloatList(value=[float(y1)])),
        "confounding_level": tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(beta1).encode()])),
        "beta1": tf.train.Feature(float_list=tf.train.FloatList(value=[float(beta1)])),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def make_nn_tfrecord(output_path="../dat/PeerRead/proc/peerread_all_betas_bow.tfrecord",
                     base_file="../dat/PeerRead/proc/arxiv-all.tf_record",
                     vocab_size=5000):
    # Parser di base (raw PeerRead abstracts)
    dataset = make_unprocessed_PeerRead_dataset(base_file, seq_length=250)
    dataset = dataset.map(make_extra_feature_cleaning(), num_parallel_calls=tf.data.AUTOTUNE)

    def to_bow(token_ids, vocab_size=5000):
        bow = np.zeros(vocab_size, dtype=np.float32)
        for tid in token_ids:
            if tid < vocab_size and tid > 0:  # ignora padding e fuori vocabolario
                bow[tid] += 1.0
        return bow

    betas = [1.0, 5.0, 25.0]

    with tf.io.TFRecordWriter(output_path) as writer:
        for beta in betas:
            labeler = make_buzzy_based_simulated_labeler(treat_strength=0.25, 
                                                         con_strength=beta,
                                                         noise_level=1.0,
                                                         setting="simple",
                                                         seed=42)
            ds_beta = dataset.map(labeler, num_parallel_calls=tf.data.AUTOTUNE)
            for ex in ds_beta:
                token_ids = ex["token_ids"].numpy()   # se il parser li include
                bow = to_bow(token_ids, vocab_size)
                write_example(writer,
                            bow=bow,
                            index=ex["index"].numpy(),
                            treatment=ex["treatment"].numpy(),
                            outcome=ex["outcome"].numpy(),
                            beta1=beta,
                            y0=ex["y0"].numpy(),
                            y1=ex["y1"].numpy())

    print(f"✅ File unico scritto in {output_path}")

if __name__ == "__main__":
    make_nn_tfrecord()