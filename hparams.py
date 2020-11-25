import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs= 3000,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54322",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=["speaker_embedding.weight"],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='/home/drobo/hironishi/work/research/data_manage/tacotron2_dataset/csj_dataset/txts/train_all_f0.txt',
        validation_files='/home/drobo/hironishi/work/research/data_manage/tacotron2_dataset/csj_dataset/txts/test_all_f0.txt',
        text_cleaners=['transliteration_cleaners'],
        lf0s_meanvar_files = '/home/drobo/hironishi/work/research/data_manage/tacotron2_dataset/csj_dataset/lf0s_meanvar.pickle',
        lf0s_file = '/home/drobo/hironishi/work/research/data_manage/tacotron2_dataset/csj_dataset/lf0s.pickle',
        pleasantness_meanvar_files = '/home/drobo/hironishi/work/research/data_manage/tacotron2_dataset/uudb_dataset/pleasantness_meanvar.pickle',
        arousal_meanvar_files =  '/home/drobo/hironishi/work/research/data_manage/tacotron2_dataset/uudb_dataset/arousal_meanvar.pickle',
                                          
        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=16000,
        filter_length= 1024,
        hop_length=200,
        win_length=800,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim = 512,
        #追加
        using_paralin=False,
        preusing_paralin=False,
        
        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Speaker embedding
        n_speakers= 326,
        speaker_embedding_dim= 326,
        
        # hparams.prespeaker_embedding_dim は事前学習の話者が一人の場合は０
        prespeaker_embedding_dim= 32,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.1,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size= 32,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
