'''Tools to configure ML environment for Tensorflow, Pytorch or JAX and 
optional notebook/colab environment'''

import os
import sys
import logging
import subprocess

class MLEnv():
    """ Initialize platform and accelerator. 
    
    This checks initialization and available accelerator hardware for different ml platforms.
    At return, the following variables are set: `self.is_tensorflow`, `self.is_pytorch`, `self.is_jax`,
    indicating that the ml environment is available for Tensorflow, Pytorch or JAX respectively if `True`.
    `self.is_notebook` and `self.is_colab` indicate if the environment is a notebook or colab environment.
    `self.is_gpu` indicates if the environment is a GPU environment, `self.is_tpu` indicates if the 
    environment is a TPU environment, and `self.is_cpu` that no accelerator is available.
    
    The logger `MLEnv` provdides details about the hardware and ml environment.

    :param platform: Known platforms are: `'tf'` (tensorflow), `'pt'` (pytorch), and `'jax'`
    :param accelerator: known accelerators are: `'fastest'` (pick best available hardware), `'cpu'`, `'gpu'`, `'tpu'`.
    :param old_disable_eager: default 'False', on True, old v1 compatibility layer is used to disable eager mode. According to rumors that might in resulting old codepaths being used?
    """

    def __init__(self, platform='tf', accelerator='fastest', old_disable_eager=False):
        self.log = logging.getLogger('MLEnv')
        self.known_platforms = ['tf', 'pt', 'jax']
        self.known_accelerators = ['cpu', 'gpu', 'tpu', 'fastest']
        if platform not in self.known_platforms:
            self.log.error(f"Platform {platform} is not known, please check spelling.")
            return
        if accelerator not in self.known_accelerators:
            self.log.error(f"Accelerator {accelerator} is not known, please check spelling.")
            return
        self.os_type = None  #: Operating system type, e.g. `'Linux'`, `'Darwin'`
        self.py_version = None  #: Python version, e.g. `'3.7.3'`
        self.is_conda = False  #: `True` if running in a conda environment
        self.is_tensorflow = False  #: `True` if running on Tensorflow
        self.tf_version = None  #: Tensorflow version, e.g. `'2.7.0'`
        self.is_pytorch = False  #: `True` if running on Pytorch
        self.pt_version = None  #: Pytorch version, e.g. `'1.6.0'`
        self.is_jax = False  #: `True` if running on Jax
        self.jax_version = None  #: Jax version, e.g. `'0.1.0'`
        self.is_cpu = False  #: `True` if no accelerator is available
        self.is_gpu = False  #: `True` if a GPU is is available
        self.is_tpu = False  #: `True` if a TPU is is available
        self.tpu_type = None  #: TPU type, e.g. `'TPU v2'`
        self.gpu_type = None  #: GPU type, e.g. `'Tesla V100'`
        self.gpu_memory = None  #: GPU memory for NVidia cards as provided by `nvidia-smi`
        self.is_notebook = False  #: `True` if running in a notebook
        self.is_colab = False  #: `True` if running in a colab notebook
        self.tpu_strategy = None
        if platform == 'tf':
            try:
                import tensorflow as tf
                self.is_tensorflow = True
                self.tf_version = tf.__version__
            except ImportError as e:
                self.log.error(f"Tensorflow not available: {e}")
                return
            try:
                from tensorflow.python.profiler import profiler_client
                self.tf_prof = True
            except:
                self.tf_prof = False
            self.log.debug(f"Tensorflow version: {tf.__version__}")
            if accelerator == 'tpu' or accelerator == 'fastest':
                try:  # XXX This fails in non-eager mode! Switch back?
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
                    tpc=tpu.cluster_spec().as_dict()['worker']
                    self.log.debug(f'Running on TPU {tpc}')
                    self.is_tpu = True
                except ValueError:
                    tpu = None
                    if accelerator!= 'fastest':
                        self.log.debug("No TPU available")
                if self.is_tpu is True:
                    # Connect_to_cluster requires eager mode, so we can't call it after having switched to non-eager mode
                    if tf.executing_eagerly() is True:
                        # Typically, this is the case when running for the first time, afterwards, we went to non-eager mode
                        tf.config.experimental_connect_to_cluster(tpu)
                    tf.tpu.experimental.initialize_tpu_system(tpu)
                    self.tpu_strategy = tf.distribute.TPUStrategy(tpu)
                    self.tpu_num_nodes = len(self.tpu_strategy.extended.worker_devices)
                    tpu_profile_service_address = os.environ['COLAB_TPU_ADDR'].replace('8470', '8466')
                    tpu_type = f"TPU, {self.tpu_num_nodes} nodes"
                    if self.tf_prof is True:
                        from tensorflow.python.profiler import profiler_client
                        state=profiler_client.monitor(tpu_profile_service_address, 100, 2)
                        if 'TPU v2' in state:
                            tpu_type=tpu_type+' v2 (8GB)'  # that's what you currently get on Colab    
                            self.log.info("You got old TPU v2 which is limited to 8GB Ram.")
                    self.tpu_type = tpu_type
                    self.log.debug("TPU strategy available")
                    if old_disable_eager is True:
                        self.log.debug("Switching to non-eager mode")
                        tf.compat.v1.disable_eager_execution()
                        self.log.debug("TPU: eager execution disabled using old compat.v1 API!")
            if self.is_tpu is False:
                if accelerator == 'gpu' or accelerator == 'fastest':
                    try:
                        tf.config.experimental.list_physical_devices('GPU')
                        self.is_gpu = True
                    except RuntimeError as e:
                        self.log.debug(f"GPU not available: {e}")
                        self.is_gpu = False
                    if self.is_gpu is True:
                        try:
                            self.gpu_type=tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0])['device_name']
                        except Exception as e:
                            self.log.warning(f"Could not get GPU type: {e}")
                        try: 
                            card = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
                            if len(card)>=8:
                                self.gpu_memory=card[9][33:54].strip()
                            else:
                                self.log.warning(f"Could not get GPU type, unexpected output from nvidia-smi, lines={len(card)}, content={card}")
                        except Exception as e:
                            self.log.debug(f"Failed to determine GPU memory {e}")
                        self.log.debug("GPU available")
            if self.is_gpu is False: 
                self.log.info("No GPU or TPU available, this is going to be very slow!")
        if platform == 'jax':
            try:
                import jax
                self.is_jax = True
                self.jax_version = jax.__version__
            except ImportError:
                self.log.debug("Jax not available")
            if self.is_jax is True:
                if accelerator == 'tpu' or accelerator == 'fastest':
                    try:
                        import jax.tools.colab_tpu as tpu
                        jax.tools.colab_tpu.setup_tpu()
                        self.is_tpu = True
                        jd=jax.devices()
                        self.tpu_type = f"TPU, {len(jd)} nodes"
                        self.log.debug(f"JAX TPU detected: {jd}")
                    except:
                        if accelerator != 'fastest':
                            self.log.debug("JAX TPU not detected.")
                            return
                if accelerator == 'gpu' or accelerator == 'fastest':
                    try:
                        jd=jax.devices()[0]
                        gpu_device_names = ['Tesla', 'GTX', 'Nvidia']  # who knows?
                        for gpu_device_name in gpu_device_names:
                            if gpu_device_name in jd.device_kind:
                                self.is_gpu = True
                                self.log.debug(f"JAX GPU: {jd.device_kind} detected.")
                                self.gpu_type = jd.device_kind
                                break
                        if self.is_gpu is False:
                            self.log.debug("JAX GPU not available.")
                        else:
                            try:  # Full speed ahead, captain!
                                card = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
                                if len(card)>=8:
                                    self.gpu_memory=card[9][33:54].strip()
                                else:
                                    self.log.warning(f"Could not get GPU type, unexpected output from nvidia-smi, lines={len(card)}, content={card}")
                            except Exception as e:
                                self.log.debug(f"Failed to determine GPU memory {e}")
                    except:
                        if accelerator != 'fastest':
                            self.log.debug("JAX GPU not available.")
                            return
                if accelerator == 'cpu' or accelerator == 'fastest':
                    try:
                        jd=jax.devices()[0]
                        cpu_device_names = ['CPU', 'cpu']  
                        for cpu_device_name in cpu_device_names:
                            if cpu_device_name in jd.device_kind:
                                self.is_cpu = True
                                self.log.debug(f"JAX CPU: {jd.device_kind} detected.")
                                break
                        if self.is_cpu is False:
                            self.log.debug("JAX CPU not available.")
                    except:
                        self.log.error("No JAX CPU available.")
                        return
        if platform == 'pt':
            try:
                import torch
                self.is_pytorch = True
                self.pt_version = torch.__version__
            except ImportError:
                self.log.error("Pytorch not available.")
                return
            if self.is_pytorch is True:
                if accelerator == 'tpu' or accelerator == 'fastest':
                    tpu_env=False
                    try:
                        assert os.environ['COLAB_TPU_ADDR']
                        tpu_env = True
                    except:
                        self.log.debug("Pytorch TPU instance not detected.")
                    if tpu_env is True:
                        try:
                            import torch
                            if '1.9.' not in torch.__version__:
                                self.log.warning("Pytorch version probably not supported with TPUs. Try (as of 12/2021): ")
                                self.log.warning("!pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl")
                            import torch_xla.core.xla_model as xm
                            self.is_tpu = True
                            self.log.debug("Pytorch TPU detected.")
                        except:
                            self.log.error("Pytorch TPU would be available, but failed to\
                                    import torch_xla.core.xla_model.")
                            if accelerator != 'fastest':
                                return
                if accelerator == 'gpu' or accelerator == 'fastest':
                    try:
                        import torch.cuda
                        if torch.cuda.is_available():
                            self.is_gpu = True
                            self.gpu_type = torch.cuda.get_device_name(0)
                            self.log.debug(f"Pytorch GPU {self.gpu_type} detected.")
                            try:  # Full speed ahead, captain!
                                card = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
                                if len(card)>=8:
                                    self.gpu_memory=card[9][33:54].strip()
                                else:
                                    self.log.warning(f"Could not get GPU type, unexpected output from nvidia-smi, lines={len(card)}, content={card}")
                            except Exception as e:
                                self.log.debug(f"Failed to determine GPU memory {e}")
                        else:
                            self.log.debug("Pytorch GPU not available.")
                    except:
                        if accelerator != 'fastest':
                            self.log.error("Pytorch GPU not available.")
                            return
                if accelerator == 'cpu' or accelerator == 'fastest':
                    self.is_cpu = True
                    self.log.debug("Pytorch CPU detected.")
                else:
                    self.log.error("No Pytorch CPU accelerator available.")
                    return
        self.flush_timer = 0
        self.flush_timeout = 180
        self._check_osenv()
        self._check_notebook_type()
        self.desc = self.describe()
        self.log.info(self.desc)

    def _check_osenv(self):
        os_type = sys.platform
        self.os_type = os_type[0].upper()+os_type[1:]
        self.py_version = sys.version.split(' ')[0]
        if 'conda' in sys.version:
            self.is_conda = True
        else:
            self.is_conda = False

    def _check_notebook_type(self):
        """ Internal function, use :func:`describe` instead

        Note: for colab notebooks and tensorflow environemts, this function
        will load tensorboard.
        """
        try:
            if 'IPKernelApp' in get_ipython().config:
                self.is_notebook = True
                self.log.debug("You are on a Jupyter instance.")
        except NameError:
            self.is_notebook = False
            self.log.debug("You are not on a Jupyter instance.")
        if self.is_notebook is True:
            try: # Colab instance?
                from google.colab import drive
                self.is_colab = True
                if self.is_tensorflow is True:
                    get_ipython().run_line_magic('load_ext', 'tensorboard')
                    try:
                        get_ipython().run_line_magic('tensorflow_version', '2.x')
                    except:
                        pass
                self.log.debug("You are on a Colab instance.")
            except: # Not? ignore.
                self.is_colab = False
                self.log.debug("You are not on a Colab instance, so no Google Drive access is possible.")
        return self.is_notebook, self.is_colab

    def describe_osenv(self):
        desc=f"OS: {self.os_type}, Python: {self.py_version}"
        if self.is_conda:
            desc += " (Conda)"
        if self.is_notebook:
            if self.is_colab:
                desc += ", Colab Jupyter Notebook"
            else:
                desc += ", Jupyter Notebook"
        return desc

    def describe_mlenv(self):
        if self.is_tensorflow is True:
            desc = f"Tensorflow: {self.tf_version}"
        elif self.is_pytorch is True:
            desc = f"Pytorch: {self.pt_version}"
        elif self.is_jax is True:
            desc = f"JAX: {self.jax_version}"
        else:
            desc = "(no-ml-platform) "
        if self.is_tpu is True:
            desc += f", TPU: {self.tpu_type}"
        if self.is_gpu is True:
            desc += f", GPU: {self.gpu_type}"
            if self.gpu_memory is not None:
                desc += f" ({self.gpu_memory})"
        if self.is_cpu is True:
            desc += f", CPU"
        return desc

    def describe(self):
        """ Prints a description of the machine environment.

        Returns:
            str: description of the machine environment.
        """
        return self.describe_osenv()+" "+self.describe_mlenv()

    def mount_gdrive(self, mount_point="/content/drive", root_path="/content/drive/My Drive"):
        if self.is_colab is True:
            from google.colab import drive
            self.log.info("You will now be asked to authenticate Google Drive access in order to store training data (cache) and model state.")
            self.log.info("Changes will only happen within Google Drive directory `My Drive/Colab Notebooks/<project-name>`.")
            if not os.path.exists(root_path):
                drive.mount(mount_point)
                return True, root_path
            if not os.path.exists(root_path):
                self.log.error(f"Something went wrong with Google Drive access. Cannot save model to {root_path}")
                return False, '.'
            else:
                return True, root_path
        else:
            self.log.error("You are not on a Colab instance, so no Google Drive access is possible.")
            return False, '.'

    def init_paths(self, project_name=None, model_name=None):
        """ Initializes the paths for the project.

        Depending on if this is a Colab environment or not, persistent data will be stored in either
        `project_path='/content/drive/My Drive/Colab Notebooks/<project_name>'` or `project_path='.'`.

        If Google drive access is not available, data will be stored in `project_path='.'`. This data
        is lost, once the Colab session is closed.

        .. code-block:: python

            project_path/data  # training data (cache)
            project_path/model[/<model_name>]  # model state, weights, etc.
            .logs  # log files

        Note that log_path is always local, since Colab Google drive caching prevents useful logs to Google drive.

        :param project_name: name of the project. Only used for Colab environments. Is always current directory for non-Colab environments.
        :param model_name: name of the model. Optional name for model subdirectory to allow support for multiple models.
        :return: (root_path, project_path, model_path, data_path, log_path)
        """
        self.has_persistence = True
        self.root_path = None
        self.project_path = None
        self.model_path=None
        self.data_path=None
        self.log_path = "./logs"
        if self.is_colab:
            self.has_persistence, self.root_path = self.mount_gdrive()
        else:
            self.root_path='.'

        self.log.debug(f"Root path: {self.root_path}")
        if self.is_colab and self.has_persistence:
            self.project_path=os.path.join(self.root_path,f"Colab Notebooks/{project_name}")
        else:
            self.project_path=self.root_path
        if model_name is not None:
            self.model_path=os.path.join(self.project_path,f"model/{model_name}")
        else:
            self.model_path=os.path.join(self.project_path, "model")
        self.data_path=os.path.join(self.project_path,'data')
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if self.has_persistence is False:
            self.log.error("No persistent storage available. Cannot save data to Google Drive.")
        return self.root_path, self.project_path, self.model_path, self.data_path, self.log_path

