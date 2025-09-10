import tensorflow as tf

'''raw_dataset = tf.python_io.tf_record_iterator("../dat/PeerRead/proc/arxiv-all.tf_record")
print(tf.train.Example.FromString(raw_dataset))'''
'''# Descrivi la struttura del TFRecord
feature_description = {
    "text": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "outcome": tf.io.FixedLenFeature([], tf.float32),
}

def _parse_function(proto):
    return tf.io.parse_single_example(proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)

for elem in parsed_dataset.take(5):  # Mostra i primi 5
    print(elem["text"].numpy().decode())
    print("Treatment:", elem["label"].numpy())
    print("Outcome:", elem["outcome"].numpy())

import tensorflow as tf

for example in tf.python_io.tf_record_iterator("data/foobar.tfrecord"):
    print(tf.train.Example.FromString(example))'''


# funziona
'''raw_dataset = tf.data.TFRecordDataset("../dat/PeerRead/proc/arxiv-all.tf_record")

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

import tensorflow as tf'''

# Percorso originale
input_path = "../dat/PeerRead/proc/arxiv-all.tf_record"
# Percorso nuovo file TFRecord
output_path = "copia-arxiv.tf_record"

raw_dataset = tf.data.TFRecordDataset(input_path)

with tf.io.TFRecordWriter(output_path) as writer:
    for raw_record in raw_dataset:  # Itera su tutti i record originali
        writer.write(raw_record.numpy())  # Scrivi direttamente il contenuto

print(f"✅ Copiato in {output_path}")
