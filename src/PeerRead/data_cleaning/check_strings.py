# conta le feature di tipo stringa.
import tensorflow as tf

path = "../dat/PeerRead/proc/arxiv-all-final.tf_record"
raw_dataset = tf.data.TFRecordDataset(path)

count_strings = 0
for i, raw_record in enumerate(raw_dataset):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    for k, v in example.features.feature.items():
        if v.bytes_list.value:  # trovata stringa
            print(f"[Esempio {i}] ⚠️ String feature trovata:", k, v.bytes_list.value[:1])
            count_strings += 1

print("Totale feature stringa trovate:", count_strings)
