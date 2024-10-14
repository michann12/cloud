def your_map_function(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(your_map_function, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(your_map_function, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
test_dataset = test_dataset.batch(64)


