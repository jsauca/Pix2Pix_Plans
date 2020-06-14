def options_data(parser):
    # Parameters for data
    parser.add_argument('--channels',
                        help='number of channels for generated images',
                        type=int,
                        default=3)
    parser.add_argument('--data_folder',
                        help='folder for training data',
                        type=str,
                        default='dataset/images/data')
    parser.add_argument('--conditional',
                        help='conditional',
                        type=bool,
                        default=True)


def options_disc(parser):
    # Parameters for discriminator
    parser.add_argument('--disc_version',
                        help='version for discriminator',
                        type=int,
                        default=0)
    parser.add_argument('--disc_scale',
                        help='scale of hidden dimensions',
                        type=int,
                        default=64)
    parser.add_argument('--disc_checkpoint',
                        help='checkpoint for discriminator',
                        type=str,
                        default=None)


def options_gen(parser):
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
                        default=None)
    parser.add_argument('--noise_size',
                        help='size of latent space',
                        type=int,
                        default=100)


def options_trainer(parser):
    # Parameters for trainer
    parser.add_argument('--batch_size',
                        help='batch size for training',
                        type=int,
                        default=64)
    parser.add_argument('--gen_prob',
                        help='probability/frequency of training generator',
                        type=float,
                        default=1)
    parser.add_argument('--optimizer',
                        help='optimizer type : adam, sgd or rms',
                        type=str,
                        default='adam')
    parser.add_argument('--loss_type',
                        help='loss for model : wasserstein or minimax',
                        type=str,
                        default='minimax')
    parser.add_argument('--learning_rate',
                        help='learning rate for optimizer',
                        type=float,
                        default=2e-4)
    parser.add_argument('--learning_rate_decay',
                        help='learning rate decay for optimizer',
                        type=float,
                        default=0.999)  # 1
    parser.add_argument('--weight_decay',
                        help='weight decay for optimizer',
                        type=float,
                        default=1e-6)  # 0 #1e-6 #5e-6
    parser.add_argument('--num_samples',
                        help='number of samples for tester',
                        type=int,
                        default=8 * 8)
    parser.add_argument('--debug',
                        help='set to true for debug',
                        type=bool,
                        default=False)


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
                        default=2)

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
                        default='pix2pix/checkpoints/gen_checkpoint_pix.pt')
    parser.add_argument('--noise_size',
                        help='size of latent space',
                        type=int,
                        default=100)
    parser.add_argument('--shape_folder',
                        help='folder for input shapes',
                        type=str,
                        default='inputs/')
    parser.add_argument('--conditional',
                        help='conditional',
                        type=bool,
                        default=True)


def options_test_nrj(parser):

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
                        default=2)

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
                        default='pix2pix/checkpoints/gen_checkpoint_test-nrj.pt')
    parser.add_argument('--noise_size',
                        help='size of latent space',
                        type=int,
                        default=100)
    parser.add_argument('--gen_input',
                        help='path for test info',
                        type=str,
                        default='inputs/gen_eval.txt')
    parser.add_argument('--conditional',
                        help='conditional',
                        type=bool,
                        default=True)
