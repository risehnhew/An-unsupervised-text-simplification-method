
from model.encoder import RNNEncoder
from model.decoder import RNNAttentionDecoder
from model.generator import *
from trainer.translator import Translator
from utils.similarity_scorer import sentence_stats
from model.discriminator import CNN
import wordvecs
import random
import traceback
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import argparse
import numpy as np
import sys
import time
import math
random.seed(7)
torch.manual_seed(7)
from trainer.trainer import Trainer
from trainer.enctrainer import EncTrainer
from trainer.mgantrainer import MganTrainer
from trainer.sim_trainer2 import SimTrainer
from utils.logger import Logger
from trainer.validator import Validator
from loadmodel import loadmodel

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '23457'
#
#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()


def main_train():
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    np.random.seed(7)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Build argument parser
    parser = argparse.ArgumentParser(description='Train a neural machine translation model')

    # Training corpus
    corpora_group = parser.add_argument_group('training corpora', 'Corpora related arguments; specify either monolingual or parallel training corpora (or both)')
    corpora_group.add_argument('--src', default="../../tsdata/fkdifficpart-2m",help='the source language monolingual corpus')
    corpora_group.add_argument('--trg', default="../../tsdata/fkeasypart-2m", help='the target language monolingual corpus')
    corpora_group.add_argument('--src2trg', default=None, metavar=('SRC', 'TRG'), nargs=2, help='the source-to-target parallel corpus')
    corpora_group.add_argument('--trg2src', default=None, metavar=('TRG', 'SRC'), nargs=2, help='the target-to-source parallel corpus')
    corpora_group.add_argument('--max_sentence_length', type=int, default=40, help='the maximum sentence length for training (defaults to 50)')
    corpora_group.add_argument('--cache', type=int, default=4000, help='the cache size (in sentences) for corpus reading (defaults to 1000000)')
    corpora_group.add_argument('--cache_parallel', type=int, default=None, help='the cache size (in sentences) for parallel corpus reading')
    corpora_group.add_argument('--unsup',action='store_true',default=False,help='if true then supervised data is not used')
    # Embeddings/vocabulary
    embedding_group = parser.add_argument_group('embeddings', 'Embedding related arguments; either give pre-trained cross-lingual embeddings, or a vocabulary and embedding dimensionality to randomly initialize them')
    embedding_group.add_argument('--src_embeddings', help='the source language word embeddings')
    embedding_group.add_argument('--trg_embeddings', help='the target language word embeddings')
    embedding_group.add_argument('--com_embeddings', help='embeddings trained on the whole corpus')
    embedding_group.add_argument('--use_catembedds',action='store_true',default=False, help='use (common wordvecs,specific wordvecs) in embeddings')
    # embedding_group.add_argument('--use_comembedds',action='store_true',default=False, help='use common embeddings only')
    embedding_group.add_argument('--src_vocabulary', help='the source language vocabulary')
    embedding_group.add_argument('--trg_vocabulary', help='the target language vocabulary')
    embedding_group.add_argument('--embedding_size', type=int, default=0, help='the word embedding size')
    embedding_group.add_argument('--cutoff', type=int, default=50000, help='cutoff vocabulary to the given size')
    embedding_group.add_argument('--learn_encoder_embeddings', action='store_true', help='learn the encoder embeddings instead of using the pre-trained ones')
    embedding_group.add_argument('--fixed_decoder_embeddings', action='store_true', help='use fixed embeddings in the decoder instead of learning them from scratch')
    embedding_group.add_argument('--fixed_generator', action='store_true', help='use fixed embeddings in the output softmax instead of learning it from scratch')
    embedding_group.add_argument('--learn_genembedds', action='store_true',default=False, help='learn the embeddings used at softmax when fixed_gen flag is True')
    embedding_group.add_argument('--learn_dec_scratch', action='store_true',default=False, help='do not initialize dec embedds before learning')
    embedding_group.add_argument('--cat_embedds', help='the source,target language word embeddings and vocabulary')
    embedding_group.add_argument('--finetune_encembedds', action='store_true',default=False, help='learn encoder embeddings after initializing')
    embedding_group.add_argument('--diff_embeddings',action='store_true',default=False,help=\
        'source and target embeddings are different.(in this case using backtranslatino is useless)')
    # Architecture
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--layers', type=int, default=2, help='the number of encoder/decoder layers (defaults to 2)')
    architecture_group.add_argument('--hidden', type=int, default=600, help='the number of dimensions for the hidden layer (defaults to 600)')
    architecture_group.add_argument('--disable_bidirectional', action='store_true', help='use a single direction encoder')
    architecture_group.add_argument('--disable_denoising', action='store_true', help='disable random swaps')
    architecture_group.add_argument('--disable_backtranslation', action='store_true', help='disable backtranslation')
    architecture_group.add_argument('--denoising_steps', type=int, default=1,help='no of steps of denoising before backtranslation starts')
    architecture_group.add_argument('--backtranslation_steps',type=int,default=1,help='no.of steps of backtranslation right after denoising')
    architecture_group.add_argument('--immediate_consecutive',action='store_true',default=True,help='after den+back steps immediately do the denoising and back consecutively')
    architecture_group.add_argument('--max_cosine',action='store_true',default=False,help='maximizes cosine between attended embedding and predicted embedding')
    architecture_group.add_argument('--addn_noise',action='store_true',default=False,help='repeat and drop noise added to almost every translator')
    architecture_group.add_argument('--denoi_enc_loss',action='store_true',default=False,help='while denoising encoder representations of same sentence should be same')
    architecture_group.add_argument('--enable_cross_alignment',action='store_true',default=False,help='use 2 discriminators for cross alignment')
    architecture_group.add_argument('--enable_enc_alignment',action='store_true',default=False,help='use a discriminator at encoder site')
    architecture_group.add_argument('--disable_autoencoder',action='store_true',default=False,help='disable use of autoenc Trainer')
    architecture_group.add_argument('--enable_mgan',action='store_true',default=False,help='enable training the mgan ')
    architecture_group.add_argument('--startfrom',type=int,default=1,help='what step to start from')
    architecture_group.add_argument('--disable_advcompl',action='store_true',default=True,help='disable the discriminator and classifier for complicating pipeline')
    architecture_group.add_argument('--sim_Loss', action='store_true', default=False,
                                    help='similarity_loss')
    # architecture_group.add_argument('--tune_src2trg',action='store_true',default=False,help='only tune src2trg pipeline using supervised data')
    # Optimization
    optimization_group = parser.add_argument_group('optimization', 'Optimization related arguments')
    optimization_group.add_argument('--batch', type=int, default=36, help='the batch size (defaults to 50)')
    optimization_group.add_argument('--learning_rate', type=float, default=0.00012, help='the global learning rate (defaults to 0.0002)')
    optimization_group.add_argument('--dropout', metavar='PROB', type=float, default=0.2, help='dropout probability for the encoder/decoder (defaults to 0.3)')
    optimization_group.add_argument('--param_init', metavar='RANGE', type=float, default=0.1, help='uniform initialization in the specified range (defaults to 0.1,  0 for module specific default initialization)')
    optimization_group.add_argument('--iterations', type=int, default=200000, help='the number of training iterations (defaults to 300000)')
    optimization_group.add_argument('--penalty_tuning', nargs=3, type=float, default=(0.5,0.5,0), help='penalty = w1*(1-docsimil)+w2*(1-treesimil)+w3')
    optimization_group.add_argument('--cosinealpha',type=float,default=1.0,help='scaling for cosine_similarity')
    # Model saving
    saving_group = parser.add_argument_group('model saving', 'Arguments for saving the trained model')
    saving_group.add_argument('--save', metavar='PREFIX', help='save models with the given prefix')
    saving_group.add_argument('--save_interval', type=int, default=200, help='save intermediate models at this interval')
    saving_group.add_argument('--load_model',default=None,help='prefix to load the model')
    saving_group.add_argument('--save_small',action='store_true',default=False,help='only saves src2trg')
    saving_group.add_argument('--start_save', type=int, default=0, help="Start saving models from ")
    saving_group.add_argument('--stop_save', type=int, default=11000, help="Stop saving models from ")
    # Logging/validation
    logging_group = parser.add_argument_group('logging', 'Logging and validation arguments')
    logging_group.add_argument('--log_interval', type=int, default=100, help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--validation', nargs='+', default=(), help='use parallel corpora for validation')
    logging_group.add_argument('--validation_directions', nargs='+', default=['src2src', 'trg2trg', 'src2trg', 'trg2src'], help='validation directions')
    logging_group.add_argument('--validation_output', metavar='PREFIX', help='output validation translations with the given prefix')
    logging_group.add_argument('--validation_beam_size', type=int, default=0, help='use beam search for validation')


    classifier_group = parser.add_argument_group('discriminators', 'arguments required for initialization and training of discriminators')
    classifier_group.add_argument('--embedd_dim',type=int,default=600,help='hidden embeddings will be passed to discriminator')
    classifier_group.add_argument('--class_num',type=int,default=2,help='no. of classes for the classifier')
    classifier_group.add_argument('--kernel_num',type=int,default=128, help='no. of filters used')
    classifier_group.add_argument('--kernel_sizes',type=int,nargs='+',default=(1,2,3,4,5),help='filter sizes used by classifier')
    classifier_group.add_argument('--rho',type=float,default=1.0,help='loss += rho*lossadv')
    classifier_group.add_argument('--disclr',type=float,default=0.0005,help='learning rate for discriminator')
    classifier_group.add_argument('--startadv',type=int,default=6000,help='start adversarial loss on discriminator after startadv steps')
    classifier_group.add_argument('--periodic',action='store_true',default=False,help='turns adversarial losses on and off periodically after every args.startadv steps')
    classifier_group.add_argument('--noclassf',action='store_true',default=False,help='if true then classifier will not be used.')
    classifier_group.add_argument('--phase',action='store_true',default=False,help='if true then training starts with adversarial loss')
    classifier_group.add_argument('--detach_encoder',action='store_true',default=False,help='if true contexts will be detached from encoder before going through adversarial loss')
    classifier_group.add_argument('--noclasssim',action='store_true',default=False,help='if true then classifier loss will be masked for simplf pipeline')
    classifier_group.add_argument('--advdenoi',action='store_true',default=False,help='use denoising even in the adversarial and classifier losses')
    classifier_group.add_argument('--singleclassf',action='store_true',default=True,help='stronger penalty as a classifier loss')
    classifier_group.add_argument('--oldclassf',action='store_true',default=False,help='use old classifier')
    classifier_group.add_argument('--nodisc',action='store_true',default=False,help='dont use disc')
    classifier_group.add_argument('--add_control',action='store_true',default=False,help='adding control if true')
    classifier_group.add_argument('--control_num',type=int,default=1,help='limit for control variate.')
    classifier_group.add_argument('--easyprefix',type=str,default='fkeasypart-2m',help='easy datafile prefix')
    classifier_group.add_argument('--difficprefix',type=str,default='fkdifficpart-2m',help='diffic datafile prefix')
    # Other
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
    parser.add_argument('--use_multi_gpu', default=False, action='store_true', help='use multiple GPUs')

    # Parse arguments
    args = parser.parse_args()
    #PRINTED ARGS
    print(args)

    def add_optimizer(module, directions=(), no_init=False, lr=None):
        if args.param_init != 0.0 and not no_init:
            for param in module.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        optimizer = torch.optim.Adam(module.parameters(), lr=args.learning_rate if lr is None else lr)
        for direction in directions:
            direction.append(optimizer)
        return optimizer
    if args.unsup and args.load_model is None:
        args.src2trg=None
        args.trg2src=None
    # Validate arguments
    if args.src_embeddings is None and args.src_vocabulary is None or args.trg_embeddings is None and args.trg_vocabulary is None:
        print('Either an embedding or a vocabulary file must be provided')
        sys.exit(-1)
    if (args.src_embeddings is None or args.trg_embeddings is None) and (not args.learn_encoder_embeddings or args.fixed_decoder_embeddings or args.fixed_generator):
        print('Either provide pre-trained word embeddings or set to learn the encoder/decoder embeddings and generator')
        sys.exit(-1)
    if args.src_embeddings is None and args.trg_embeddings is None and args.embedding_size == 0:
        print('Either provide pre-trained word embeddings or the embedding size')
        sys.exit(-1)
    if len(args.validation) % 2 != 0:
        print('--validation should have an even number of arguments (one pair for each validation set)')
        sys.exit(-1)

    # Select device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def multi_gpu(model, multi_gpu1 = args.use_multi_gpu, device=device):
        # if multi_gpu1:
        #
        #     model = nn.DataParallel(model)
        model.to(device)
        return model

    # Create optimizer lists
    src2src_optimizers = []
    trg2trg_optimizers = []
    src2trg_optimizers = []
    trg2src_optimizers = []
    disc_optimizers = []
    enc_optimizers = []
    simdec_optimizers = []
    comdec_optimizers = []
    # Method to create a module optimizer and add it to the given lists


    # Load word embeddings---------------------------------------------------------------------------
    src_words = trg_words = src_embeddings = trg_embeddings = src_dictionary = trg_dictionary = None
    embedding_size = args.embedding_size
    if args.src_vocabulary is not None and not args.load_model: # make dictionary
        f = open(args.src_vocabulary, encoding=args.encoding, errors='surrogateescape')
        src_words = [line.strip() for line in f.readlines()]
        if args.cutoff > 0:
            src_words = src_words[:args.cutoff]
        src_dictionary = data.Dictionary(src_words)
    if args.trg_vocabulary is not None and not args.load_model: # make dictionary
        f = open(args.trg_vocabulary, encoding=args.encoding, errors='surrogateescape')
        trg_words = [line.strip() for line in f.readlines()]
        if args.cutoff > 0:
            trg_words = trg_words[:args.cutoff]
        trg_dictionary = data.Dictionary(trg_words)

    if not args.use_catembedds:
        if args.src_embeddings is not None and not args.load_model:
            f = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
            src_embeddings, src_dictionary = data.read_embeddings(f, args.cutoff, src_words)
            src_embeddings = src_embeddings.to(device)
            #printed srcembeddings
            # print("srcembeddings {}".format(src_embeddings.size()))
            src_embeddings.weight.requires_grad = False
            if embedding_size == 0:
                embedding_size = src_embeddings.weight.data.size()[1]
            if embedding_size != src_embeddings.weight.data.size()[1]:
                print('Embedding sizes do not match')
                sys.exit(-1)
        if args.trg_embeddings is not None and not args.load_model:
            # print('you are doing nice')
            trg_file = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
            trg_embeddings, trg_dictionary = data.read_embeddings(trg_file, args.cutoff, trg_words)
            trg_embeddings = trg_embeddings.to(device)
            # print(type(trg_embeddings))
            #printed trg_embeddings
            # print("trg_embeddings {}".format(trg_embeddings.size()))
            trg_embeddings.weight.requires_grad = False
            if embedding_size == 0:
                embedding_size = trg_embeddings.weight.data.size()[1]
            if embedding_size != trg_embeddings.weight.data.size()[1]:
                print('Embedding sizes do not match')
                sys.exit(-1)
    else:
        sys.stdout.flush()
        (src_embeddings, src_dictionary),(trg_embeddings, trg_dictionary) = torch.load(args.cat_embedds)
        src_embeddings = src_embeddings.to(device)
        trg_embeddings = trg_embeddings.to(device)
        src_embeddings.weight.requires_grad = False
        if embedding_size == 0:
            embedding_size = src_embeddings.weight.data.size()[1]
        if embedding_size != src_embeddings.weight.data.size()[1]:
            print('Embedding sizes do not match')
            sys.exit(-1)
        trg_embeddings.weight.requires_grad = False
        if embedding_size == 0:
            embedding_size = trg_embeddings.weight.data.size()[1]
        if embedding_size != trg_embeddings.weight.data.size()[1]:
            print('Embedding sizes do not match')
            sys.exit(-1)

    if args.learn_encoder_embeddings and not args.load_model:
        src_encoder_embeddings = data.random_embeddings(src_dictionary.size(), embedding_size).to(device)
        trg_encoder_embeddings = data.random_embeddings(trg_dictionary.size(), embedding_size).to(device) if not args.diff_embeddings \
                                    else src_encoder_embeddings
        if args.finetune_encembedds:
            src_encoder_embeddings.weight.data = src_embeddings.weight.data.clone()
            trg_encoder_embeddings.weight.data = trg_embeddings.weight.data.clone() if not args.diff_embeddings \
                                    else src_encoder_embeddings.weight.data
        add_optimizer(src_encoder_embeddings, (src2src_optimizers, src2trg_optimizers,enc_optimizers))
        # if not args.diff_embeddings:
        add_optimizer(trg_encoder_embeddings, (trg2trg_optimizers, trg2src_optimizers,enc_optimizers))
    elif not args.load_model:
        src_encoder_embeddings = src_embeddings
        trg_encoder_embeddings = trg_embeddings if not args.diff_embeddings else src_embeddings
    if args.fixed_decoder_embeddings and not args.load_model:
        src_decoder_embeddings = src_embeddings
        trg_decoder_embeddings = trg_embeddings
        src_decoder_embeddings.weight.requires_grad=False
        trg_decoder_embeddings.weight.requires_grad=False
    elif not args.load_model:
        src_decoder_embeddings = data.random_embeddings(src_dictionary.size(), embedding_size).to(device)
        trg_decoder_embeddings = data.random_embeddings(trg_dictionary.size(), embedding_size).to(device)
        if not args.learn_dec_scratch:
            src_decoder_embeddings.weight.data = src_embeddings.weight.data.clone()
            trg_decoder_embeddings.weight.data = trg_embeddings.weight.data.clone()
        src_decoder_embeddings.weight.requires_grad=True
        trg_decoder_embeddings.weight.requires_grad=True
        add_optimizer(src_decoder_embeddings, (src2src_optimizers, trg2src_optimizers, comdec_optimizers))
        add_optimizer(trg_decoder_embeddings, (trg2trg_optimizers, src2trg_optimizers, simdec_optimizers))
    if args.fixed_generator and not args.load_model:
        src_embedding_generator = multi_gpu(EmbeddingGenerator(hidden_size=args.hidden, embedding_size=embedding_size))
        trg_embedding_generator = multi_gpu(EmbeddingGenerator(hidden_size=args.hidden, embedding_size=embedding_size))
        src_genembedding = data.random_embeddings(src_dictionary.size(), embedding_size).to(device)
        trg_genembedding = data.random_embeddings(trg_dictionary.size(), embedding_size).to(device)
        src_genembedding.weight.data = src_embeddings.weight.data.clone()
        trg_genembedding.weight.data = trg_embeddings.weight.data.clone()
        src_genembedding.weight.requires_grad=args.learn_genembedds
        trg_genembedding.weight.requires_grad=args.learn_genembedds        
        add_optimizer(src_embedding_generator, (src2src_optimizers, trg2src_optimizers, comdec_optimizers))
        add_optimizer(trg_embedding_generator, (trg2trg_optimizers, src2trg_optimizers, simdec_optimizers))
        src_generator = multi_gpu(WrappedEmbeddingGenerator(src_embedding_generator, src_genembedding))
        trg_generator = multi_gpu(WrappedEmbeddingGenerator(trg_embedding_generator, trg_genembedding))
        if args.learn_genembedds:
            add_optimizer(src_genembedding, (src2src_optimizers, trg2src_optimizers, comdec_optimizers))
            add_optimizer(trg_genembedding, (trg2trg_optimizers, src2trg_optimizers, simdec_optimizers))
    elif not args.load_model:
        src_generator = multi_gpu(LinearGenerator(args.hidden, src_dictionary.size()))
        trg_generator = multi_gpu(LinearGenerator(args.hidden, trg_dictionary.size()))
        add_optimizer(src_generator, (src2src_optimizers, trg2src_optimizers, comdec_optimizers))
        add_optimizer(trg_generator, (trg2trg_optimizers, src2trg_optimizers, simdec_optimizers))
#------------------------------------------------------------------------------------------------------
    # Build encoder
    

# Build translators-----------------------------------------------------------------------------------
#     loadmodel(args, device, multi_gpu, Translator, add_optimizer, src2trg_optimizers, trg2trg_optimizers,
#               simdec_optimizers,src2src_optimizers, trg2src_optimizers, comdec_optimizers, disc_optimizers)
    if args.load_model is not None:
        loadmodel(args, device, multi_gpu, Translator, add_optimizer, src2trg_optimizers, trg2trg_optimizers,
                  simdec_optimizers, src2src_optimizers, trg2src_optimizers, comdec_optimizers, disc_optimizers)

    else:
        encoder = multi_gpu(RNNEncoder(embedding_size=embedding_size, hidden_size=args.hidden,
                                bidirectional=not args.disable_bidirectional, layers=args.layers, dropout=args.dropout))
        add_optimizer(encoder, (src2src_optimizers, trg2trg_optimizers, src2trg_optimizers, trg2src_optimizers,enc_optimizers))

        # Build decoders
        src_decoder = multi_gpu(RNNAttentionDecoder(embedding_size=embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
        trg_decoder = multi_gpu(RNNAttentionDecoder(embedding_size=embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
        add_optimizer(src_decoder, (src2src_optimizers, trg2src_optimizers,comdec_optimizers))
        add_optimizer(trg_decoder, (trg2trg_optimizers, src2trg_optimizers,simdec_optimizers))

        src2src_translator = Translator(encoder_embeddings=src_encoder_embeddings,
                                        decoder_embeddings=src_decoder_embeddings, generator=src_generator,
                                        src_dictionary=src_dictionary, trg_dictionary=src_dictionary, encoder=encoder,
                                        decoder=src_decoder, denoising=not args.disable_denoising, device=device,repeatnoise=False\
                                         if args.addn_noise else None, psencoder_embeddings=trg_encoder_embeddings)
        src2trg_translator = Translator(encoder_embeddings=src_encoder_embeddings,
                                        decoder_embeddings=trg_decoder_embeddings, generator=trg_generator,
                                        src_dictionary=src_dictionary, trg_dictionary=trg_dictionary, encoder=encoder,
                                        decoder=trg_decoder, denoising=not args.disable_denoising, device=device,repeatnoise=True\
                                         if args.addn_noise else None, psencoder_embeddings=trg_encoder_embeddings)
        trg2trg_translator = Translator(encoder_embeddings=trg_encoder_embeddings,
                                        decoder_embeddings=trg_decoder_embeddings, generator=trg_generator,
                                        src_dictionary=trg_dictionary, trg_dictionary=trg_dictionary, encoder=encoder,
                                        decoder=trg_decoder, denoising=not args.disable_denoising, device=device,repeatnoise=True\
                                         if args.addn_noise else None, psencoder_embeddings=src_encoder_embeddings)
        trg2src_translator = Translator(encoder_embeddings=trg_encoder_embeddings,
                                        decoder_embeddings=src_decoder_embeddings, generator=src_generator,
                                        src_dictionary=trg_dictionary, trg_dictionary=src_dictionary, encoder=encoder,
                                        decoder=src_decoder, denoising=not args.disable_denoising, device=device,repeatnoise=False\
                                         if args.addn_noise else None, psencoder_embeddings=src_encoder_embeddings)
        if args.enable_cross_alignment: # build discriminator
            src_discriminator = CNN(args)
            trg_discriminator = CNN(args)
        
        if args.enable_enc_alignment:
            enc_discriminator = multi_gpu(CNN(args))
            # enc_discriminator = multi_gpu(enc_discriminator)
            add_optimizer(enc_discriminator, (disc_optimizers,),lr=args.disclr)
        else:
            enc_discriminator=None
        if args.enable_mgan:
            mgan_disc = multi_gpu(CNN(args))
            # mgan_disc = multi_gpu(mgan_disc)
            add_optimizer(mgan_disc, (disc_optimizers,),lr=args.disclr)
        else:
            mgan_disc = None

    if args.unsup:
        args.src2trg=None
        args.trg2src=None
#---------------------------------------------------------------------------------------
    # Build trainers
    trainers = []
    src2src_trainer = trg2trg_trainer = src2trg_trainer = trg2src_trainer = None
    srcback2trg_trainer = trgback2src_trainer = None
    if args.src is not None:
        if not args.add_control:
            f = open(args.src, encoding=args.encoding, errors='surrogateescape')
            corpus = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        if not args.disable_autoencoder:
            if args.add_control:
                flist = [open(args.difficprefix+'-{}.lower'.format(i), encoding=args.encoding, errors='surrogateescape') for i in range(1,args.control_num+1)]
                corpus = [ data.CorpusReader(fl, max_sentence_length=args.max_sentence_length, cache_size=args.cache) for fl in flist]
            src2src_trainer = Trainer(translator=src2src_translator, optimizers=src2src_optimizers, corpus=corpus, batch_size=args.batch,add_control=args.add_control)
            trainers.append(src2src_trainer)
        # if not args.disable_backtranslation: # back-translation
        #     if args.add_control:
        #         flist = [open(args.difficprefix+'-{}.lower'.format(i), encoding=args.encoding, errors='surrogateescape') for i in range(1,args.control_num+1)]
        #         corpus = [data.CorpusReader(fl, max_sentence_length=args.max_sentence_length, cache_size=args.cache) for fl in flist]
        #         corpus = [[data.BacktranslatorCorpusReader(corpus=corp, translator=src2trg_translator,ncontrol=con)for con in range(1,args.control_num+1)] for corp in corpus]
        #         corpus = [ data.MulBackCorpusReader(*corp,control_num=args.control_num) for corp in corpus]
        #     else:
        #         corpus = data.BacktranslatorCorpusReader(corpus=corpus, translator=src2trg_translator)
        #     trgback2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers,
        #                                   corpus=corpus,\
        #                                    batch_size=args.batch, backbool=True, penalty_tuning=args.penalty_tuning\
        #                                    ,cosinealpha=args.cosinealpha,add_control=args.add_control)
        #     trainers.append(trgback2src_trainer)

    if args.trg is not None:
        if not args.add_control:
            f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
            corpus = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        if not args.disable_autoencoder:
            if args.add_control:
                flist = [open(args.easyprefix+'-{}.lower'.format(i), encoding=args.encoding, errors='surrogateescape') for i in range(1,args.control_num+1)]
                corpus = [data.CorpusReader(fl, max_sentence_length=args.max_sentence_length, cache_size=args.cache) for fl in flist]
            trg2trg_trainer = Trainer(translator=trg2trg_translator, optimizers=trg2trg_optimizers, corpus=corpus, batch_size=args.batch,add_control=args.add_control)
            trainers.append(trg2trg_trainer)
        # if not args.disable_backtranslation:
        #     if args.add_control:
        #         flist = [open(args.easyprefix+'-{}.lower'.format(i), encoding=args.encoding, errors='surrogateescape') for i in range(1,args.control_num+1)]
        #         corpus = [data.CorpusReader(fl, max_sentence_length=args.max_sentence_length, cache_size=args.cache) for fl in flist]
        #         corpus = [[data.BacktranslatorCorpusReader(corpus=corp, translator=trg2src_translator,ncontrol=con)for con in range(1,args.control_num+1)] for corp in corpus]
        #         corpus = [ data.MulBackCorpusReader(*corp,control_num=args.control_num) for corp in corpus]
        #     else:
        #         corpus = data.BacktranslatorCorpusReader(corpus=corpus, translator=trg2src_translator)
        #     srcback2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers,
        #                                   corpus=corpus\
        #                                   , batch_size=args.batch,\
        #                                    backbool=True, penalty_tuning=args.penalty_tuning, cosinealpha=args.cosinealpha,\
        #                                    add_control=args.add_control)
        #     trainers.append(srcback2trg_trainer)
    # if args.src2trg is not None:
    #     f1 = open(args.src2trg[0], encoding=args.encoding, errors='surrogateescape')
    #     f2 = open(args.src2trg[1], encoding=args.encoding, errors='surrogateescape')
    #     corpus = data.CorpusReader(f1, f2, max_sentence_length=args.max_sentence_length, cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
    #     src2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers, corpus=corpus, batch_size=args.batch)
    #     trainers.append(src2trg_trainer)
    # if args.trg2src is not None:
    #     f1 = open(args.trg2src[0], encoding=args.encoding, errors='surrogateescape')
    #     f2 = open(args.trg2src[1], encoding=args.encoding, errors='surrogateescape')
    #     corpus = data.CorpusReader(f1, f2, max_sentence_length=args.max_sentence_length, cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
    #     trg2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers, corpus=corpus, batch_size=args.batch)
    #     trainers.append(trg2src_trainer)
    if args.src is not None and args.trg is not None and args.enable_mgan:

        if not args.add_control:
            #instantiate Three corpuses and pass them to combined wrapper.
            f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
            corpustrg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
            f = open(args.src, encoding=args.encoding, errors='surrogateescape')
            corpussrc = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
            corpus = data.MganCorpusReader(corpustrg,corpussrc)
        else:
            flist = [open(args.difficprefix+'-{}.lower'.format(i), encoding=args.encoding, errors='surrogateescape') for i in range(1,args.control_num+1)]
            corpusdiffic = [data.CorpusReader(fl, max_sentence_length=args.max_sentence_length, cache_size=args.cache) for fl in flist]
            corpusdiffic = data.MulCorpusReader(*corpusdiffic, control_num=args.control_num)
            flist = [open(args.easyprefix+'-{}.lower'.format(i), encoding=args.encoding, errors='surrogateescape') for i in range(1,args.control_num+1)]
            corpuseasy = [data.CorpusReader(fl, max_sentence_length=args.max_sentence_length, cache_size=args.cache) for fl in flist]
            corpuseasy = data.MulCorpusReader(*corpuseasy, control_num=args.control_num) 
            corpus = data.MganCorpusReader(corpuseasy,corpusdiffic)
        #instantiate an Mgan Trainer
        disc_trainer = MganTrainer(src2src_translator=src2src_translator,src2trg_translator=src2trg_translator, trg2src_translator=trg2src_translator,\
         trg2trg_translator=trg2trg_translator, discriminator=mgan_disc\
        , optimizers=(disc_optimizers,enc_optimizers,simdec_optimizers,comdec_optimizers), corpus=corpus, gen_train=False\
        , add_control=args.add_control,control_num=args.control_num)

        #instantiate Three corpuses and pass them to combined wrapper.
        if not args.add_control:
            f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
            corpustrg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
            f = open(args.src, encoding=args.encoding, errors='surrogateescape')
            corpussrc = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
            corpus = data.MganCorpusReader(corpustrg,corpussrc)
        else:
            flist = [open(args.difficprefix+'-{}.lower'.format(i), encoding=args.encoding, errors='surrogateescape') for i in range(1,args.control_num+1)]
            corpusdiffic = [data.CorpusReader(fl, max_sentence_length=args.max_sentence_length, cache_size=args.cache) for fl in flist]
            corpusdiffic = data.MulCorpusReader(*corpusdiffic, control_num=args.control_num)
            flist = [open(args.easyprefix+'-{}.lower'.format(i), encoding=args.encoding, errors='surrogateescape') for i in range(1,args.control_num+1)]
            corpuseasy = [data.CorpusReader(fl, max_sentence_length=args.max_sentence_length, cache_size=args.cache) for fl in flist]
            corpuseasy = data.MulCorpusReader(*corpuseasy, control_num=args.control_num) 
            corpus = data.MganCorpusReader(corpuseasy,corpusdiffic)
        mgan_trainer = MganTrainer(src2src_translator=src2src_translator,src2trg_translator=src2trg_translator, trg2src_translator=trg2src_translator,\
         trg2trg_translator=trg2trg_translator, discriminator=mgan_disc\
        , optimizers=(disc_optimizers,enc_optimizers,simdec_optimizers,comdec_optimizers), corpus=corpus, gen_train=True\
        , add_control=args.add_control,control_num=args.control_num)
        trainers.append(disc_trainer)
        trainers.append(mgan_trainer)
        if args.sim_Loss: # similarity constraint
            sim_trainer = SimTrainer(src2src_translator=src2src_translator, \
                                       trg2trg_translator=trg2trg_translator, discriminator=mgan_disc \
                                       , optimizers=(disc_optimizers, enc_optimizers, simdec_optimizers, comdec_optimizers),
                                       corpus=corpus, gen_train=True \
                                       , add_control=args.add_control, control_num=args.control_num, src2trg_translator=src2trg_translator, trg2src_translator=trg2src_translator)
            trainers.append(sim_trainer)
    # if args.enable_cross_alignment and args.src is not None and args.trg is not None:
    #     #initializing src Cross Trainer
    #     f = open(args.src, encoding=args.encoding, errors='surrogateescape')
    #     corpus_src = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
    #     f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
    #     corpus_trg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
    #     src_cross_trainer = CrossTrainer(rec_translator=src2src_translator,cross_translator=trg2src_translator,discriminator=src_discriminator\
    #         ,corpus_rec=corpus_src,corpus_cross=corpus_trg)
    #     #initializing trg Cross Trainer
    #     f = open(args.src, encoding=args.encoding, errors='surrogateescape')
    #     corpus_src = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
    #     f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
    #     corpus_trg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
    #     src_cross_trainer = CrossTrainer(rec_translator=trg2trg_translator,cross_translator=src2trg_translator,discriminator=trg_discriminator\
    #         ,corpus_rec=corpus_trg,corpus_cross=corpus_src)

    if args.enable_enc_alignment:
        f = open(args.src, encoding=args.encoding, errors='surrogateescape')
        corpus_src = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
        corpus_trg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        src_align_trainer = EncTrainer(rec_translator=src2src_translator,discriminator=enc_discriminator\
            ,corpus_rec=corpus_src,corpus_cross=corpus_trg, optimizers=(src2src_optimizers,disc_optimizers),srcbool=True)
        trainers.append(src_align_trainer)
        f = open(args.src, encoding=args.encoding, errors='surrogateescape')
        corpus_src = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
        corpus_trg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        trg_align_trainer = EncTrainer(rec_translator=trg2trg_translator,discriminator=enc_discriminator\
            ,corpus_rec=corpus_trg,corpus_cross=corpus_src, optimizers=(trg2trg_optimizers,disc_optimizers),srcbool=False)
        trainers.append(trg_align_trainer)




    # Build validators
    src2src_validators = []
    trg2trg_validators = []
    src2trg_validators = []
    trg2src_validators = []
    for i in range(0, len(args.validation), 2):
        src_validation = open(args.validation[i],   encoding=args.encoding, errors='surrogateescape').readlines()
        trg_validation = open(args.validation[i+1], encoding=args.encoding, errors='surrogateescape').readlines()
        if len(src_validation) != len(trg_validation):
            print('Validation sizes do not match')
            sys.exit(-1)
        map(lambda x: x.strip(), src_validation)
        map(lambda x: x.strip(), trg_validation)
        if 'src2src' in args.validation_directions:
            src2src_validators.append(Validator(src2src_translator, src_validation, src_validation, args.batch, args.validation_beam_size))
        if 'trg2trg' in args.validation_directions:
            trg2trg_validators.append(Validator(trg2trg_translator, trg_validation, trg_validation, args.batch, args.validation_beam_size))
        if 'src2trg' in args.validation_directions:
            src2trg_validators.append(Validator(src2trg_translator, src_validation, trg_validation, args.batch, args.validation_beam_size))
        if 'trg2src' in args.validation_directions:
            trg2src_validators.append(Validator(trg2src_translator, trg_validation, src_validation, args.batch, args.validation_beam_size))

    # Build loggers
    loggers = []
    src2src_output = trg2trg_output = src2trg_output = trg2src_output = None
    if args.validation_output is not None:
        src2src_output = '{0}.src2src'.format(args.validation_output)
        trg2trg_output = '{0}.trg2trg'.format(args.validation_output)
        src2trg_output = '{0}.src2trg'.format(args.validation_output)
        trg2src_output = '{0}.trg2src'.format(args.validation_output)
    loggers.append(Logger('Source to target (backtranslation)', srcback2trg_trainer, [], None, args.encoding))
    loggers.append(Logger('Target to source (backtranslation)', trgback2src_trainer, [], None, args.encoding))
    loggers.append(Logger('Source to source', src2src_trainer, src2src_validators, src2src_output, args.encoding))
    loggers.append(Logger('Target to target', trg2trg_trainer, trg2trg_validators, trg2trg_output, args.encoding))
    loggers.append(Logger('Source to target', src2trg_trainer, src2trg_validators, src2trg_output, args.encoding))
    loggers.append(Logger('Target to source', trg2src_trainer, trg2src_validators, trg2src_output, args.encoding))

    # Method to save models
    def save_models(name):
        if args.save_small:
            torch.save(src2trg_translator, '{0}.{1}.src2trg.pth'.format(args.save, name))
        else:
            torch.save(src2src_translator, '{0}.{1}.src2src.pth'.format(args.save, name))
            torch.save(trg2trg_translator, '{0}.{1}.trg2trg.pth'.format(args.save, name))
            torch.save(src2trg_translator, '{0}.{1}.src2trg.pth'.format(args.save, name))
            torch.save(trg2src_translator, '{0}.{1}.trg2src.pth'.format(args.save, name))
            torch.save(enc_discriminator,'{0}.{1}.encdiscriminator.pth'.format(args.save, name))
            torch.save(mgan_disc,'{0}.{1}.mgandiscriminator.pth'.format(args.save, name))
    # Training
    total = args.denoising_steps+args.backtranslation_steps
    numdenbac = args.iterations//total
    rem = args.iterations%total
    den =False
    back=False
    allstep_t = 0
    for i in range(numdenbac+1):
        if i==numdenbac:
            nsteps = rem
        else:
            nsteps = total
        offset = (i)*total

        for step in range(1+offset, nsteps+offset + 1):
            verbose = (step % args.log_interval == 0)
            if nsteps==rem:
                back=True
                den=True
            if step-(offset+1)<args.denoising_steps and nsteps!=rem:
                den=True
                back=False
            if step-(offset+1)>=args.denoising_steps and nsteps!=rem:
                den=False
                back=True

            if i >=1 and args.immediate_consecutive:
                den=True
                back=True

            for idx,trainer in enumerate(trainers):
                # src2src, trg2src, trg2trg, src2trg, src2trgsup, trg2srcsup
                if not args.disable_backtranslation:
                    if den and idx%2 and not back and not idx>3:
                        continue
                    if back and idx%2==0 and not den and not idx>3:
                        continue
                    if idx%2==0 and idx<=3 and verbose:
                        print('denoising STEP')
                    if idx%2 and idx<=3 and verbose:
                        print('backtranslation STEP')
                    if idx==4 and back and not den:
                        continue
                    if idx==5 and back and not den:
                        continue
                    if idx==4 and verbose:
                        print('src2trg STEP')
                    if idx==5 and verbose:
                        print('trg2src STEP')
#                 try:
                if not args.add_control:
                    trainer.step(step,args.log_interval,device,args=args)
                    # mp.spawn(trainer.step(step,args.log_interval,device,args=args), args=(4,), nprocs=4, join=True)

                else:
                    for ncontrol in range(1,args.control_num+1):
                        # try:
                        trainer.step(step,args.log_interval,device,args=args,ncontrol=ncontrol)
                        # mp.spawn(trainer.step(step, args.log_interval, device, args=args, ncontrol=ncontrol), args=(4,), nprocs=4,
                        #          join=True)
                        # except Exception as e:
                        #     print('EXCEPTION OCCURED !!: {}'.format(e))
                        #     continue
#                 except Exception as e:
#                     print('EXCEPTION OCCURED !!: {}'.format(e))
#                     continue
            if args.save is not None and args.save_interval > 0 and step % args.save_interval == 0 and \
            step>args.startfrom and step >= args.start_save and step < args.stop_save:
                save_models('it{0}'.format(step))

            if step % args.log_interval == 0:
                print()
                print('STEP {0} x {1} no of loggers {2}'.format(step, args.batch,len(loggers)))
            #     for logger in loggers:
            #         print(logger.name,logger.trainer,logger.validators,logger.output_prefix,logger.encoding)
            #         sys.stdout.flush()
            #         logger.log(step)

            # step += 1
                print("time for {} steps : {} sec".format(args.log_interval,time.time()-allstep_t))
                allstep_t = time.time()
    save_models('final')




main_train()