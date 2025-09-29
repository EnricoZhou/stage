import tensorflow as tf

input_file = "../dat/PeerRead/proc/arxiv-all.tf_record"
output_file = "../dat/PeerRead/proc/arxiv-small.tf_record"

n_examples = 100  # quante righe vuoi

writer = tf.io.TFRecordWriter(output_file)
count = 0

for record in tf.compat.v1.io.tf_record_iterator(input_file):
    writer.write(record)
    count += 1
    if count >= n_examples:
        break

writer.close()
print(f"Creato {output_file} con {count} esempi")