import tensorflow as tf
import yaml

class Neural_Network:
    """
    Neural network functions
    Attributes:
    - optimizer
    """
    def __init__(self):
        self.cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.cfg['learning_rate'])

    # The loss function
    def loss(self, model, x, y, training):
        print(model)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        y_ = model(x, training=training)
        return loss_object(y_true=y, y_pred=y_)
    
    # Gradients and loss
    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
            # central_server.epoch_loss_avg.update_state(loss_value)
            # central_server.epoch_accuracy.update_state(targets, central_server.model(inputs, training=True))
        return tape.gradient(loss_value, model.trainable_variables)

    # Function used to aggregate gradient values into one
    def accumulate_gradients(self, dest, step_gradients):
        if dest.accumulative_gradients is None:
            dest.accumulative_gradients = [self.flat_gradients(g) for g in step_gradients]
        else:
            for i, g in enumerate(step_gradients):
                dest.accumulative_gradients[i] += self.flat_gradients(g) 

    # Function used to aggregate gradient values into one
    def accumulate_gradients_itr(self, step_gradients):
        accumulative_gradients = []
        for x in step_gradients:
            if not accumulative_gradients:
                accumulative_gradients = [self.flat_gradients(g) for g in x]
            else:
                for i, g in enumerate(x):
                    accumulative_gradients[i] += self.flat_gradients(g) 
        return accumulative_gradients

    # Helper function for accumulate_gradients()
    def flat_gradients(self, grads_or_idx_slices):
        if type(grads_or_idx_slices) == tf.IndexedSlices:
            return tf.scatter_nd(
                tf.expand_dims(grads_or_idx_slices.indices, 1),
                grads_or_idx_slices.values,
                grads_or_idx_slices.dense_shape
            )
        return grads_or_idx_slices
