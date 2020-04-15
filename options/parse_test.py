def options_test(parser):

    parser.add_argument('--channels',
                        help='number of channels for generated images',
                        type=int,
                        default=3)
    parser.add_argument('--outputs',
                        help='folder for outputs',
                        type=str,
                        default='outputs/')
    parser.add_argument('--number',
                        help='number of generated outputs',
                        type=int,
                        default=5)

    # Parameters for generator
    parser.add_argument('--gen_version',
                        help='version for discriminator',
                        type=int,
                        default=0)
    parser.add_argument('--gen_scale',
                        help='scale of hidden dimensions',
                        type=int,
                        default=64)
    parser.add_argument('--gen_checkpoint',
                        help='checkpoint for generator',
                        type=str,
                        default='dcgan/checkpoints/gen_checkpoint_v0.pt')
    parser.add_argument('--noise_size',
                        help='size of latent space',
                        type=int,
                        default=100)
