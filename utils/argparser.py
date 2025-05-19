import argparse

def get_train_val_args():
    parser = argparse.ArgumentParser(description="Training and Validation script for image completion model")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save trained models')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--print_every', type=int, default=5, help='Print metrics every N epochs')
    parser.add_argument('--data', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--upsampling_method', type=str, default='pixelshuffle', choices=['pixelshuffle', 'nearest'], help='Upsampling method for UNet')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_freq', type=int, default=5, help='Frequency of saving model checkpoints')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio for dataset split')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio for dataset split')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio for dataset split')
    parser.add_argument('--model', type=str, default='model_a', choices=['model_a', 'unet','model_b'], help='Which model architecture to use')



    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser(description="Testing script for image completion model")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save trained models')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--data', type=str, required=True, help='Path to testing dataset')
    parser.add_argument('--upsampling_method', type=str, default='pixelshuffle', choices=['pixelshuffle', 'nearest'], help='Upsampling method for UNet')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio for dataset split')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio for dataset split')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio for dataset split')
    parser.add_argument('--model', type=str, default='model_a', choices=['model_a', 'unet','model_b'], help='Which model architecture to use')


    return parser.parse_args()

