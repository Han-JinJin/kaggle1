"""
Quick script to verify MAE checkpoint can be loaded correctly
Run this before main training to ensure checkpoint loading works
"""

import torch
import sys

def verify_checkpoint_keys(ckpt_path):
    """Verify checkpoint has expected keys"""
    print(f"\n{'='*60}")
    print(f"Verifying: {ckpt_path}")
    print(f"{'='*60}\n")

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Find encoder-related keys
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k]

        print(f"✅ Checkpoint loaded successfully")
        print(f"   Total keys: {len(state_dict)}")
        print(f"   Encoder keys: {len(encoder_keys)}")
        print(f"\n   Sample encoder keys:")
        for key in encoder_keys[:5]:
            print(f"   - {key}")

        # Verify key mapping
        print(f"\n{'='*60}")
        print("Key Mapping Verification:")
        print(f"{'='*60}\n")

        for mae_key in encoder_keys[:3]:
            # MAE format: encoder.xxx
            # Expected: backbone.xxx
            swin_key = mae_key.replace('encoder.', 'backbone.')
            print(f"MAE key:    {mae_key}")
            print(f"Swin3D key: {swin_key}")
            print(f"Shape:      {state_dict[mae_key].shape}")
            print()

        return True

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(ckpt_path):
    """Test if SwinModel can load the checkpoint"""
    print(f"\n{'='*60}")
    print("Testing Model Loading:")
    print(f"{'='*60}\n")

    try:
        # Import model
        from swin import SwinModel

        # Create model instance
        model = SwinModel(
            pred_shape=(1000, 1000),  # Dummy shape
            size=224,
            lr=5e-5,
            checkpoint_path=ckpt_path,
            load_encoder_only=True
        )

        print("✅ Model created and checkpoint loaded successfully!")
        print(f"   Model type: {type(model.backbone).__name__}")
        print(f"   Encoder type: {type(model.backbone.encoder).__name__}")

        return True

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import glob
    import os

    # Find available checkpoints
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory '{checkpoint_dir}' not found!")
        print(f"   Please run MAE pretraining first: python mae-swin.py")
        sys.exit(1)

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

    if not ckpt_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}/")
        print(f"   Please run MAE pretraining first: python mae-swin.py")
        sys.exit(1)

    # Use the latest checkpoint
    latest_ckpt = sorted(ckpt_files)[-1]
    print(f"\n🔍 Using latest checkpoint: {latest_ckpt}\n")

    # Verify checkpoint keys
    if not verify_checkpoint_keys(latest_ckpt):
        sys.exit(1)

    # Test model loading
    if not test_model_loading(latest_ckpt):
        sys.exit(1)

    print(f"\n{'='*60}")
    print("✅ All checks passed! Ready for training.")
    print(f"{'='*60}\n")

    print("\n🚀 To start training with MAE weights, run:")
    print(f"\npython train.py \\")
    print(f"    --checkpoint_path {latest_ckpt} \\")
    print(f"    --load_encoder_only \\")
    print(f"    --segments remaining5 rect5 \\")
    print(f"    --valid_id rect5")
    print()
