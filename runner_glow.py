import os

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs import GlowTTSConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs import BaseDatasetConfig, CharactersConfig

# init_training: Initialize and setup the training environment.
# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from TTS.trainer import init_training, Trainer, TrainingArgs

lj_dataset_config = BaseDatasetConfig(name="ljspeech",
                                   meta_file_train="metadata_abridged.csv",
                                   path="/home/arvind/data/LJSpeech-1.1/")
btc_dataset_config = BaseDatasetConfig(name="btcsessions",
                                   meta_file_train="metadata.tsv",
                                   path="/home/arvind/data/btcsessions/")
latam_dataset_config = BaseDatasetConfig(name="latam",
                                   meta_file_train="metadata.tsv",
                                   path="/home/arvind/data/LatinAmericanTTS/")

characters = CharactersConfig(
    pad = "_",
    eos = "~",
    bos = "^",
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz¡£¿ÁÉÚàáâèéêíñóúü",
    punctuations = ":;?!\"$&'(),-./[\]_’“” ",
    phonemes = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧɚ˞ɫ",
    unique = True
)

# we use the same path as this script as our training folder.
output_path = "/home/arvind/code/output/"

# Configure the model. Every config class inherits the BaseTTSConfig to have all the fields defined for the Trainer.
config = GlowTTSConfig(
    run_name="glow-btcsessions",
    characters=characters, 
    batch_size=32,
    batch_group_size=4,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=100,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    datasets=[latam_dataset_config, btc_dataset_config],
   
    # Model parameters
    hidden_channels_enc=192,
    hidden_channels_dec=192,
    hidden_channels_dp=256,
    use_encoder_prenet=True,
    encoder_type="rel_pos_transformer",
    encoder_params={
        "kernel_size": 3,
	"dropout_p": 0.1,
	"num_layers": 6,
	"num_heads": 2,
	"hidden_channels_ffn": 768,
	"input_length": None 
    },

    min_seq_len=6,
    max_seq_len=180,
    
    # multi-speaker
    #num_speakers=171,
    use_speaker_embedding=True,
    use_d_vector_file=True,
    d_vector_file="/home/arvind/code/output/btcsessions/speakerembeddings/speakers.json", 
    d_vector_dim=256, 

    # testing
    test_sentences=["si alguien tuviera una ventaja injusta y tuviera mucha computación cuántica poder, bueno",
    "Y utilizando el sistema tal como está construido y beneficiándose de él. Entonces, y yo diría que alguien que bloquea estos fondos y desvía fondos a otras personas es en realidad el ladrón.",
    "Solo unos pocos males evidentes todavía exigían un remedio.",
    "The hieroglyph has a yellow fish",
    "trece, un punto tres billones."],
    
    # TENSORBOARD and LOGGING
    tb_plot_step=100,
    save_step=5000,
    checkpoint=True,
    keep_all_best=False,
    keep_after=10000,
    tb_model_param_stats=False
)

# Take the config and the default Trainer arguments, setup the training environment and override the existing
# config values from the terminal. So you can do the following.
# >>> python train.py --coqpit.batch_size 128
args, config, output_path, _, _, _= init_training(TrainingArgs(), config)

# Initiate the Trainer.
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training etc.
trainer = Trainer(args, config, output_path)

# And kick it 🚀
trainer.fit()
