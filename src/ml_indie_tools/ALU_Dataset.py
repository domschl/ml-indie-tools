import sys
import os
import logging
import random
import numpy as np


class ALU_Dataset:
    """Generate training data for all ALU operations

    The ALU takes two integers and applies one of the supported
    model_ops. E.g. `op1=123, op2=100, op='-' -> result 23`

    The net is supposed to learn to 'calculate' the results for
    arbitrary op1, op2 (positive integers, `0..2**bit_count - 1`) and
    the twelve supported ops:
    `["+", "-", "*", "/", "%", "AND", "OR", "XOR", ">", "<", "=", "!="]`

    :param bit_count: number of bits for each of the two operands, default 31 (mult uses 15 bits)
    :param pre_weight: if True, the model_dis will be reweighted to generate samples for 'difficult' ops
    """

    def __init__(self, bit_count=31, pre_weight=False):
        self.log = logging.getLogger("Datasets")
        self.model_ops = [
            "+",
            "-",
            "*",
            "/",
            "%",
            "AND",
            "OR",
            "XOR",
            ">",
            "<",
            "=",
            "!=",
        ]
        self.model_is_boolean = [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
        ]
        # Probabilites for creating a sample for each of the ops, (Will be
        # reweighted on checks to generate for samples for 'difficult' ops):
        self.model_dis = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        model_dis_w = [19, 12, 110, 15, 36, 10, 10, 10, 10, 10, 10, 10]
        self.model_funcs = [
            self._add_smpl,
            self._diff_smpl,
            self._mult_smpl,
            self._div_smpl,
            self._mod_smpl,
            self._and_smpl,
            self._bor_smpl,
            self._xor_smpl,
            self._greater_smpl,
            self._lesser_smpl,
            self._eq_smpl,
            self._neq_smpl,
        ]
        self.bit_count = bit_count
        self.op_count = len(self.model_ops)
        if self.bit_count + 1 > self.op_count:
            self.embedding_size = self.bit_count + 1
        else:
            self.embedding_size = self.op_count
        self.pre_weight = pre_weight
        self.all_bits_one = 2**self.bit_count - 1
        self.true_vect = self.all_bits_one
        self.false_vect = 0
        self.input_size = (self.bit_count + 1) * 2 + len(self.model_ops)
        self.output_size = 32
        if pre_weight is True:
            self.model_dis = model_dis_w

    @staticmethod
    def _int_to_binary_vect(num_int, num_bits=8):
        """get a binary encoded vector of n of bit-lenght nm

        :param num_int: integer to encoded
        :param num_bits: number of bits to use for positional_encoding
        :return: binary vector of length num_bits
        """
        num_vect = np.zeros(num_bits, dtype=np.float32)
        for i in range(0, num_bits):
            if num_int & (2**i) != 0:
                num_vect[i] = 1.0
        return num_vect

    @staticmethod
    def _int_to_onehot_vect(num_int, num_bits):
        """get a one-hot encoded vector of n of bit-lenght nm

        :param num_int: integer to encoded
        :param num_bits: number of bits to use for positional_encoding
        :return: one-hot vector of length num_bits
        """
        num_vect = np.zeros(num_bits, dtype=np.float32)
        num_vect[num_int] = 1.0
        return num_vect

    @staticmethod
    def _get_random_bits(bits):
        """get bits random int 0...2**bits-1

        :param bits: number of bits to uses
        :return: random int `0...2**bits-1`
        """
        return random.randint(0, 2**bits - 1)

    def op_string_to_index(self, op_string):
        """transform op_string (e.g. '+' -> 0) into corresponding index

        :param op_string: string of op to transform
        :return: index of op_string
        """
        for i in range(0, len(self.model_ops)):
            if self.model_ops[i] == op_string:
                return i
        return -1

    def get_data_point(
        self,
        equal_distrib=False,
        valid_ops=None,
        vector=False,
        positional_encoding=False,
    ):
        """Get a random example for on ALU operation for training

        :param equal_distrib: if False, more 'difficult' ops will be generated more often.
        :param valid_ops: if not None, only the ops in valid_ops will be used
        :param vector: if True, the result will be returned as an embedded encoded vector
        :param positional_encoding: if True, the result will be returned as an embedded encoded vector with additional bits for positional positional_encoding

        """
        # result = -1
        op1 = self._get_random_bits(self.bit_count)
        op2 = self._get_random_bits(self.bit_count)
        if valid_ops is not None and len(valid_ops) == 0:
            valid_ops = None
        if valid_ops is not None:
            if equal_distrib is False:
                self.log.warning(
                    "Op restriction via valid_ops forces equal_distrib=True"
                )
                equal_distrib = True
            for op in valid_ops:
                if op not in self.model_ops:
                    self.log.warning(
                        f"Cannot restrict valid_ops to {op}, unknown operation, ignoring all valid_ops"
                    )
                    valid_ops = None
                    break

        if equal_distrib or valid_ops is not None:
            if valid_ops is None:
                op_index = random.randint(0, len(self.model_ops) - 1)
            else:
                if len(valid_ops) == 1:
                    op_index = 0
                else:
                    op_index = random.randint(0, len(valid_ops) - 1)
                op_index = self.model_ops.index(valid_ops[op_index])
        else:  # make 'difficult' ops more present in training samples:
            rx = 0
            for md in self.model_dis:
                rx += md
            rrx = random.randint(0, rx)
            rx = 0
            op_index = 0
            for op_index in range(0, len(self.model_ops)):
                rx += self.model_dis[op_index]
                if rx > rrx:
                    break
        return self._encode_op(
            op1, op2, op_index, vector=vector, positional_suffix=positional_encoding
        )

    def _generatorenerator(self, samples=20000, equal_distrib=False, valid_ops=None):
        while True:
            x, Y = self.create_training_data(
                samples=samples,
                valid_ops=valid_ops,
                equal_distrib=equal_distrib,
                title=None,
            )
            yield x, Y

    def _encode_op(self, op1, op2, op_index, vector=False, positional_suffix=False):
        """turn two ints and operation into training data"""
        op1, op2, result = self.model_funcs[op_index](op1, op2)
        if self.model_is_boolean[op_index] is True:
            if result == self.false_vect:

                str_result = "True"
            else:
                str_result = "undefined"
        else:
            str_result = result
        sym = f"{op1} {self.model_ops[op_index]} {op2} = {str_result}"
        if vector is True:
            if positional_suffix is True:
                sz = self.embedding_size + 3
            else:
                sz = self.embedding_size
            v1 = self._int_to_binary_vect(op1, num_bits=sz)
            v2 = self._int_to_onehot_vect(op_index, num_bits=sz)
            v3 = self._int_to_binary_vect(op2, num_bits=sz)
            if positional_suffix is True:
                v1[-3] = 1.0
                v2[-2] = 1.0
                v3[-1] = 1.0
            inp = np.array([v1, v2, v3], dtype=np.float32)
        else:
            inp = np.concatenate(
                [
                    self._int_to_binary_vect(op1, num_bits=self.bit_count + 1),
                    self._int_to_onehot_vect(op_index, num_bits=len(self.model_ops)),
                    self._int_to_binary_vect(op2, num_bits=self.bit_count + 1),
                ]
            )

        oup = self._int_to_binary_vect(result, num_bits=self.output_size)
        return inp, oup, result, op_index, sym

    @staticmethod
    def _add_smpl(op1, op2):
        """addition training example"""
        result = op1 + op2
        return op1, op2, result

    @staticmethod
    def _diff_smpl(op1, op2):
        """subtraction training example"""
        if op2 > op1:
            op2, op1 = op1, op2
        result = op1 - op2
        return op1, op2, result

    def _mult_smpl(self, op1, op2):
        """multiplication training example"""
        modul = 2 ** (self.bit_count // 2) - 1
        op1 = op1 % modul
        op2 = op2 % modul
        result = op1 * op2
        return op1, op2, result

    def _div_smpl(self, op1, op2):
        """integer division training example"""
        while op2 == 0:
            op2 = self._get_random_bits(self.bit_count)
        if op1 < op2 and random.randint(0, 2) != 0:
            if op1 != 0:
                op1, op2 = op2, op1
        result = op1 // op2
        return op1, op2, result

    def _mod_smpl(self, op1, op2):
        """modulo (remainder) training example"""
        while op2 == 0:
            op2 = self._get_random_bits(self.bit_count)
        if op1 < op2 and random.randint(0, 2) != 0:
            if op1 != 0:
                op1, op2 = op2, op1
        result = op1 % op2
        return op1, op2, result

    @staticmethod
    def _and_smpl(op1, op2):
        """bitwise AND training example"""
        result = op1 & op2
        return op1, op2, result

    @staticmethod
    def _bor_smpl(op1, op2):
        """bitwise OR training example"""
        result = op1 | op2
        return op1, op2, result

    @staticmethod
    def _xor_smpl(op1, op2):
        """bitwise XOR training example"""
        result = op1 ^ op2
        return op1, op2, result

    def _greater_smpl(self, op1, op2):
        """integer comparisation > training example"""
        if op1 > op2:
            result = self.true_vect
        else:
            result = self.false_vect
        return op1, op2, result

    def _lesser_smpl(self, op1, op2):
        """integer comparisation < training example"""
        if op1 < op2:
            result = self.true_vect
        else:
            result = self.false_vect
        return op1, op2, result

    def _eq_smpl(self, op1, op2):
        """integer comparisation == training example"""
        if random.randint(0, 1) == 0:  # create more cases
            op2 = op1
        if op1 == op2:
            result = self.true_vect
        else:
            result = self.false_vect
        return op1, op2, result

    def _neq_smpl(self, op1, op2):
        """integer comparisation != training example"""
        if random.randint(0, 1) == 0:  # create more cases
            op2 = op1
        if op1 != op2:
            result = self.true_vect
        else:
            result = self.false_vect
        return op1, op2, result

    def create_data_point(
        self, op1, op2, op_string, vector=False, positional_suffix=False
    ):
        """create training data from given ints op1, op2 and op_string"""
        op_index = self.op_string_to_index(op_string)
        if op_index == -1:
            print(f"Invalid operation {op_string}")
            return np.array([]), np.array([]), -1, -1, None
        return self._encode_op(op1, op2, op_index, vector, positional_suffix)

    def create_training_data(
        self,
        samples=10000,
        valid_ops=None,
        equal_distrib=False,
        verbose=True,
        title=None,
    ):
        """create a number of training samples"""
        x, y, _, _, _ = self.get_data_point()
        dpx = np.zeros((samples, len(x)), dtype=np.float32)
        dpy = np.zeros((samples, len(y)), dtype=np.float32)
        if verbose is True:
            if title is None:
                print(f"Creating {samples} data points (. = 1000 progress)")
            else:
                print(f"{title}: Creating {samples} data points (. = 1000 progress)")

        for i in range(0, samples):
            if verbose is True:
                if i % 100000 == 0:
                    print(f"{i:>10} ", end="")
            if (i + 1) % 1000 == 0:
                if verbose is True:
                    print(".", end="")
                    sys.stdout.flush()
                    if (i + 1) % 100000 == 0:
                        print()
            if valid_ops is None:
                x, y, _, _, _ = self.get_data_point(equal_distrib=equal_distrib)
            else:
                x, y, _, _, _ = self.get_data_point(
                    equal_distrib=True, valid_ops=valid_ops
                )
            dpx[i, :] = x
            dpy[i, :] = y
        if verbose is True:
            print()
        return dpx, dpy

    def create_vector_training_data(
        self,
        samples=10000,
        valid_ops=None,
        equal_distrib=False,
        verbose=True,
        title=None,
        positional_encoding=True,
    ):
        """create a number of training samples"""
        x, y, _, _, _ = self.get_data_point()
        if positional_encoding is True:
            sz = self.embedding_size + 3
        else:
            sz = self.embedding_size
        dpx = np.zeros((samples, 3, sz), dtype=np.float32)
        dpy = np.zeros((samples, len(y)), dtype=np.float32)
        if verbose is True:
            if title is None:
                print(f"Creating {samples} data points (. = 1000 progress)")
            else:
                print(f"{title}: Creating {samples} data points (. = 1000 progress)")

        for i in range(0, samples):
            if verbose is True:
                if i % 100000 == 0:
                    print(f"{i:>10} ", end="")
            if (i + 1) % 1000 == 0:
                if verbose is True:
                    print(".", end="")
                    sys.stdout.flush()
                    if (i + 1) % 100000 == 0:
                        print()
            if valid_ops is None:
                x, y, _, _, _ = self.get_data_point(
                    equal_distrib=equal_distrib,
                    vector=True,
                    positional_encoding=positional_encoding,
                )
            else:
                x, y, _, _, _ = self.get_data_point(
                    equal_distrib=True,
                    valid_ops=valid_ops,
                    vector=True,
                    positional_encoding=positional_encoding,
                )
            dpx[i, :, :] = x
            dpy[i, :] = y
        if verbose is True:
            print()
        return dpx, dpy

    if "tensorflow" in sys.modules:

        def create_dataset(
            self,
            samples=10000,
            batch_size=2000,
            vector=False,
            positional_encoding=True,
            is_training=True,
            valid_ops=None,
            name=None,
            cache_path=None,
            use_cache=True,
            regenerate_cached_data=False,
            for_tpu=False,
        ):
            import tensorflow as tf

            is_loaded = False
            if use_cache is True and cache_path is None:
                print("can't use cache if no cache_path is given, disabling cache!")
                use_cache = False
            if use_cache is True:
                if valid_ops is not None:
                    infix = "_"
                    for vo in valid_ops:
                        if vo == "*":  # Prevent poison-filenames
                            vo = "MULT"
                        if vo == "/":
                            vo = "DIV"
                        if vo == "%":
                            vo = "MOD"
                        if vo == "<":
                            vo = "LT"
                        if vo == ">":
                            vo = "GT"
                        if vo == "=":
                            vo = "EQ"
                        if vo == "!=":
                            vo = "NE"
                        infix += vo
                else:
                    infix = ""
                if vector is True:
                    infix += "_VECT"
                if positional_encoding is True:
                    infix += "_PE"
                cache_file_x = os.path.join(
                    cache_path, f"{name}_{infix}_{self.bit_count}_{samples}_x.npy"
                )
                cache_file_Y = os.path.join(
                    cache_path, f"{name}_{infix}_{self.bit_count}_{samples}_Y.npy"
                )
            else:
                cache_file_x = None
                cache_file_Y = None
            if (
                use_cache is True
                and regenerate_cached_data is False
                and os.path.exists(cache_file_x)
                and os.path.exists(cache_file_Y)
            ):
                try:
                    x = np.load(cache_file_x, allow_pickle=True)
                    Y = np.load(cache_file_Y, allow_pickle=True)
                    if len(x) == samples:
                        is_loaded = True
                        self.log.debug(f"Data {name} loaded from cache")
                    else:
                        self.log.info(
                            f"Sample count has changed from {len(x)} to {samples}, regenerating {name} data..."
                        )
                except Exception as e:
                    self.log.error(
                        f"Something went wrong when loading {cache_file_x}, {cache_file_Y}: {e}"
                    )
            if is_loaded is False:
                if vector is True:
                    x, Y = self.create_vector_training_data(
                        samples=samples,
                        valid_ops=valid_ops,
                        title=name,
                        positional_encoding=positional_encoding,
                    )
                else:
                    x, Y = self.create_training_data(
                        samples=samples, valid_ops=valid_ops, title=name
                    )
                if use_cache is True:
                    self.log.debug(
                        f"Writing data-cache {cache_file_x}, {cache_file_Y}..."
                    )
                    np.save(cache_file_x, x, allow_pickle=True)
                    np.save(cache_file_Y, Y, allow_pickle=True)
                    self.log.debug("x, Y, done.")
            shuffle_buffer = 10000
            dataset = tf.data.Dataset.from_tensor_slices((x, Y)).cache()
            if is_training is True:
                dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
                if for_tpu is True:
                    dataset = dataset.repeat()  # Mandatory for Keras TPU for now
            dataset = dataset.batch(
                batch_size, drop_remainder=True
            )  # drop_remainder is important on TPU, batch size must be fixed
            dataset = dataset.prefetch(
                -1
            )  # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
            return dataset

        def create_dataset_from_generator(self, valid_ops=None):
            import tensorflow as tf

            dataset = tf.data.Dataset.from_generator(
                self._generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, self.input_size), dtype=np.float32),
                    tf.TensorSpec(shape=(None, self.output_size), dtype=np.float32),
                ),
            )
            return dataset

        def get_datasets(
            self,
            pre_weight=True,
            samples=100000,
            validation_samples=10000,
            batch_size=2000,
            vector=False,
            positional_encoding=True,
            valid_ops=None,
            cache_path=None,
            use_cache=True,
            regenerate_cached_data=False,
            for_tpu=False,
        ):
            train = self.create_dataset(
                samples=samples,
                batch_size=batch_size,
                is_training=True,
                vector=vector,
                positional_encoding=positional_encoding,
                valid_ops=valid_ops,
                name="train",
                cache_path=cache_path,
                use_cache=use_cache,
                regenerate_cached_data=regenerate_cached_data,
                for_tpu=for_tpu,
            )
            val = self.create_dataset(
                samples=validation_samples,
                batch_size=batch_size,
                vector=vector,
                positional_encoding=positional_encoding,
                is_training=False,
                valid_ops=valid_ops,
                name="validation",
                cache_path=cache_path,
                use_cache=use_cache,
                regenerate_cached_data=regenerate_cached_data,
            )
            return train, val

    else:

        def get_datasets(
            self,
            pre_weight=True,
            samples=100000,
            validation_samples=10000,
            batch_size=2000,
            vector=False,
            positional_encoding=True,
            valid_ops=None,
            cache_path=None,
            use_cache=True,
            regenerate_cached_data=False,
        ):
            print("No tensorflow, no datasets!")
            return None, None

    def decode_results(self, result_int_vects):
        """take an array of 32-float results from neural net and convert to ints"""
        result_vect_ints = []
        for vect in result_int_vects:
            if len(vect) != self.output_size:
                print(f"Ignoring unexpected vector of length {len(vect)}")
            else:
                int_result = 0
                for i in range(0, self.output_size):
                    if vect[i] > 0.5:
                        int_result += 2**i
                result_vect_ints.append(int_result)
        return result_vect_ints

    def check_results(
        self,
        model,
        samples=1000,
        vector=False,
        positional_encoding=True,
        valid_ops=None,
        verbose=False,
    ):
        """Run a number of tests on trained model"""
        ok = 0
        err = 0
        operr = [0] * len(self.model_ops)
        opok = [0] * len(self.model_ops)
        for _ in range(0, samples):
            x, _, z, op, s = self.get_data_point(
                equal_distrib=True,
                vector=vector,
                positional_encoding=positional_encoding,
                valid_ops=valid_ops,
            )
            res = self.decode_results(model.predict(np.array([x])))
            if res[0] == z:
                ok += 1
                opok[op] += 1
                r = "OK"
            else:
                err += 1
                operr[op] += 1
                r = "Error"
            if verbose is True:
                if self.model_is_boolean[op] is True:
                    if res[0] == self.false_vect:
                        str_result = "False"
                    elif res[0] == self.true_vect:
                        str_result = "True"
                    else:
                        str_result = "undefined"
                else:
                    str_result = res[0]
                if res[0] == z:
                    print(f"{s} == {str_result}: {r}")
                else:
                    print(f"{s} != {str_result}: {r}")
                    if self.model_is_boolean[op] is False:
                        print(bin(res[0]))
                        print(bin(z))
        opsum = ok + err
        if opsum == 0:
            opsum = 1
        print(f"Ok: {ok}, Error: {err} -> {ok/opsum*100.0}%")
        print("")
        for i in range(0, len(self.model_ops)):
            opsumi = opok[i] + operr[i]
            if opsumi == 0:
                continue
            # modify the distribution of training-data generated to favour
            # ops with bad test results, so that more training data is
            # generated on difficult cases:
            self.model_dis[i] = int(operr[i] / opsumi * 100) + 10
            print(f"OP{self.model_ops[i]}: Ok: {opok[i]}, Error: {operr[i]}", end="")
            print(f" -> {opok[i]/opsumi*100.0}%")
        if valid_ops is None:
            print("Change probability for ops in new training data:")
            print(f"Ops:     {self.model_ops}")
            print(f"Weights: {self.model_dis}")
        return ok / opsum
