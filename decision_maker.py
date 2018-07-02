import os
import numpy as np
import tensorflow as tf
import multiprocessing


class DecisionMaker:
    def __init__(self, state_space, number_of_actions: int, model_dir: str, learning_rate: float=0.001):
        self._model_name = "DeepQN"
        self.__learning_rate = learning_rate
        self.__conv_filter_count = [32, 64, 64]
        self.__conv_filter_size = [8, 4, 3]
        self.__conv_stride = [4, 2, 1]
        self.__fully_connected_units = [512]
        self.__number_of_actions = number_of_actions
        self.__state_space = state_space
        self.__decision_process_started = False
        self.__prediction_function = None
        model_dir = os.path.join(model_dir, "models", self._model_name)
        os.makedirs(model_dir, exist_ok=True)
        self.__model = tf.estimator.Estimator(model_fn=self._model_fn,
                                              model_dir=model_dir)

        # When a decision for a state is needed to be taken, the state will be stored in the `__needed_to_predict`
        # variable and then used in the `input_generation_for_prediction` function to provide the input for the model,
        # then the model will make the decision.
        self.__need_to_predict = None
        self.__provided_input = multiprocessing.Semaphore(0)

    def __del__(self):
        self.__terminate_decision_process()

    def reset(self):
        self.__terminate_decision_process()

    def _model_fn(self, features, labels, mode):
        with tf.variable_scope("Decision_Making"):
            net = features
            for layer_index, (filter_count, filter_size, stride) in enumerate(zip(self.__conv_filter_count,
                                                                                  self.__conv_filter_size,
                                                                                  self.__conv_stride)):
                net = tf.layers.conv2d(inputs=net,
                                       filters=filter_count,
                                       kernel_size=filter_size,
                                       strides=stride,
                                       padding='valid',
                                       name="conv_{}".format(layer_index))
            net = tf.layers.flatten(inputs=net)
            for layer_index, units in enumerate(self.__fully_connected_units):
                net = tf.layers.dense(inputs=net,
                                      units=units,
                                      activation=tf.nn.relu,
                                      name="dens_".format(layer_index))
            output = tf.layers.dense(inputs=net,
                                     units=self.__number_of_actions,
                                     activation=None,
                                     name="output")
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'selected_action': tf.argmax(output, axis=1),
                    'estimated_q_value': tf.reduce_max(output, axis=1)
                }
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
            committed_action = tf.one_hot(indices=labels['committed_action'],
                                          depth=self.__number_of_actions,
                                          dtype=tf.float32)
            output = tf.multiply(output, committed_action)
            loss = tf.losses.mean_squared_error(labels=labels['q_value'], predictions=output)
            tf.summary.scalar('loss', loss)
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.__learning_rate)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def __input_generator_for_prediction(self):
        while True:
            self.__provided_input.acquire()
            if self.__need_to_predict is None:
                return
            yield self.__need_to_predict

    def __get_features_structure(self):
        features_type = tf.float32
        features_shape = tf.TensorShape([None, *self.__state_space.shape])
        return features_type, features_shape

    def __get_labels_structure(self):
        labels_type = {'q_value': tf.float32, 'committed_action': tf.uint8}
        labels_shape = {'q_value': tf.TensorShape([None, self.__number_of_actions]),
                        'committed_action': tf.TensorShape([None])}
        return labels_type, labels_shape

    def __start_decision_process(self):
        if self.__decision_process_started:
            return
        self.__decision_process_started = True
        self.__provided_input = multiprocessing.Semaphore(0)
        features_type, features_shape = self.__get_features_structure()
        input_generator = tf.data.Dataset.from_generator(self.__input_generator_for_prediction,
                                                         features_type,
                                                         features_shape)
        self.__prediction_function = self.__model.predict(input_fn=lambda: tf.data.Dataset.from_generator(
            self.__input_generator_for_prediction, features_type, features_shape))

    def __terminate_decision_process(self):
        if not self.__decision_process_started:
            return
        self.__decision_process_started = False
        self.__prediction_function = None
        self.__need_to_predict = None

    def making_decision(self, state: np.ndarray=None):
        """
        Making decision about which action should be performed in the given `state`.
        If `state` be None, it means that decision procedure should be over.

        :param state: The state which the decision should be take for it.
        :return: The action which should be perform in the given state.
        """
        if state is None:
            self.__terminate_decision_process()
            return

        if not self.__decision_process_started:
            self.__start_decision_process()
        self.__need_to_predict = np.array([state])
        self.__provided_input.release()
        return next(self.__prediction_function)['selected_action']

    def train(self, input_generator):
        features_type, features_shape = self.__get_features_structure()
        labels_type, labels_shape = self.__get_labels_structure()
        types = (features_type, labels_type)
        shapes = (features_shape, labels_shape)
        self.__model.train(input_fn=lambda: tf.data.Dataset.from_generator(input_generator,
                                                                           types,
                                                                           shapes))

