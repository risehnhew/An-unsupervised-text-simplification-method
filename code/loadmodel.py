import torch

def loadmodel(args, device, multi_gpu, Translator,add_optimizer, src2trg_optimizers, trg2trg_optimizers, simdec_optimizers,
              src2src_optimizers, trg2src_optimizers, comdec_optimizers, disc_optimizers):
    if args.load_model is not None:
        print(args.src, args.src2trg)
        if args.src is not None:
            t = torch.load(args.load_model + '.src2src.pth')
            # t = torch.load(args.load_model)
            # Translate sentences
            src2src_denoising = not args.disable_denoising
            src2src_device = device

            src2src_encoder = multi_gpu(t.encoder)
            src2src_decoder = multi_gpu(t.decoder)
            src2src_encoder_embeddings = multi_gpu(t.encoder_embeddings)
            src2src_decoder_embeddings = multi_gpu(t.decoder_embeddings)
            src2src_generator = multi_gpu(t.generator)
            src2src_src_dictionary = t.src_dictionary
            src2src_trg_dictionary = t.trg_dictionary
            src2src_translator = Translator(src2src_encoder_embeddings, src2src_decoder_embeddings, src2src_generator,
                                            src2src_src_dictionary, \
                                            src2src_trg_dictionary, src2src_encoder, src2src_decoder, src2src_denoising,
                                            src2src_device, repeatnoise=True if args.addn_noise else None)
        else:
            src2src_translator = None

        if args.src2trg is not None:
            print(args.src2trg)
            t = torch.load(args.load_model + '.src2trg.pth')
            # t = torch.load(args.load_model)
            src2trg_denoising = not args.disable_denoising
            src2trg_device = device

            src2trg_encoder = multi_gpu(t.encoder)
            src2trg_decoder = multi_gpu(t.decoder)
            src2trg_encoder_embeddings = multi_gpu(t.encoder_embeddings)
            src2trg_decoder_embeddings = multi_gpu(t.decoder_embeddings)
            src2trg_generator = multi_gpu(t.generator)
            src2trg_src_dictionary = t.src_dictionary
            src2trg_trg_dictionary = t.trg_dictionary
            # Translate sentences
            src2trg_translator = Translator(src2trg_encoder_embeddings, src2trg_decoder_embeddings, src2trg_generator,
                                            src2trg_src_dictionary, \
                                            src2trg_trg_dictionary, src2trg_encoder, src2trg_decoder, src2trg_denoising,
                                            src2trg_device, repeatnoise=True if args.addn_noise else None)
            add_optimizer(src2trg_decoder, (src2trg_optimizers, trg2trg_optimizers, simdec_optimizers), no_init=True)
            add_optimizer(src2trg_decoder_embeddings, (src2trg_optimizers, trg2trg_optimizers, simdec_optimizers),
                          no_init=True)
            add_optimizer(src2trg_generator, (src2trg_optimizers, trg2trg_optimizers, simdec_optimizers), no_init=True)
            # if args.learn_encoder_embeddings:
            #     t.encoder_embeddings.requires_grad=True
            #     add_optimizer(src_encoder_embeddings, (src2src_optimizers, src2trg_optimizers,enc_optimizers),no_init=True)
        else:
            src2trg_translator = None

        if args.trg is not None:
            t = torch.load(args.load_model+'.trg2trg.pth')
            # t = torch.load(args.load_model)
            t.denoising=not args.disable_denoising
            t.device=device
            t.encoder=multi_gpu(t.encoder)
            t.decoder=multi_gpu(t.decoder)
            t.encoder_embeddings=multi_gpu(t.encoder_embeddings)
            t.decoder_embeddings=multi_gpu(t.decoder_embeddings)
            t.generator=multi_gpu(t.generator)
            # Translate sentences
            # trg2trg_translator = Translator(src2src_encoder_embeddings, src2trg_decoder_embeddings, src2trg_generator,
            #                                 src2src_src_dictionary, \
            #                                 src2trg_trg_dictionary, src2src_encoder, src2trg_decoder, src2trg_denoising,
            #                                 src2trg_device, repeatnoise=True if args.addn_noise else None)


        else:
            trg2trg_translator = None

        if args.trg2src is not None:
            t = torch.load(args.load_model+'.trg2src.pth')
            t.denoising=not args.disable_denoising
            # t.device=device
            # t.encoder=device(t.encoder)
            # t.decoder=device(t.decoder)
            # t.encoder_embeddings=device(t.encoder_embeddings)
            # t.decoder_embeddings=device(t.decoder_embeddings)
            # t.generator=device(t.generator)
            # Translate sentences
            # trg2src_translator = Translator(src2src_encoder_embeddings, src2src_decoder_embeddings, src2src_generator,
            #                                 src2src_src_dictionary, \
            #                                 src2src_trg_dictionary, src2src_encoder, src2src_decoder, src2src_denoising,
            #                                 src2src_device, repeatnoise=True if args.addn_noise else None)
            add_optimizer(src2src_decoder, (src2src_optimizers, trg2src_optimizers, comdec_optimizers), no_init=True)
            add_optimizer(src2src_decoder_embeddings, (src2src_optimizers, trg2src_optimizers, comdec_optimizers),
                          no_init=True)
            add_optimizer(src2src_generator, (src2src_optimizers, trg2src_optimizers, comdec_optimizers), no_init=True)
            add_optimizer(src2src_encoder,
                          (src2src_optimizers, src2trg_optimizers, trg2trg_optimizers, trg2src_optimizers, enc_optimizers),
                          no_init=True)
            # if args.learn_encoder_embeddings:
            #     t.encoder_embeddings.requires_grad=True
            #     add_optimizer(t.encoder_embeddings, (trg2trg_optimizers, trg2src_optimizers,enc_optimizers),no_init=True)

        else:
            trg2src_translator = None

        if args.enable_mgan:
            mgan_disc = torch.load(args.load_model + '.mgandiscriminator.pth')

            mgan_disc = multi_gpu(mgan_disc)
            enc_discriminator = None
            add_optimizer(mgan_disc, (disc_optimizers,), lr=args.disclr, no_init=True)
        else:
            mgan_disc = None
        enc_discriminator = None
        # print(src2src_translator,src2trg_translator,trg2trg_translator,trg2src_translator,mgan_disc,enc_discriminator)