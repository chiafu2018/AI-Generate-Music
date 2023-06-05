import tensorflow as tf

# Check GPU

print(tf.__version__)
gpu_devices = tf.config.list_physical_devices('GPU')

if len(gpu_devices) > 0:
    print("The useful GPUs are:")
    for device in gpu_devices:
        print(device)
else:
    print("No useful GPU")


