"""
Main CLI Entrypoint for the Glaucoma Detection Research System.
Provides a unified interface to run training, evaluation, XAI, application, and test suites.
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Glaucoma Detection Research Pipeline CLI")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True,
        choices=['audit', 'split', 'train_seg', 'eval_seg', 'train_cls', 'eval_xai', 'cross_eval', 'app', 'test'],
        help="Pipeline stage to execute"
    )
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--dataset', type=str, default='acrima')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()

    if args.mode == 'audit':
        from src.data.dataset_analyzer import main as run_audit
        run_audit()
    elif args.mode == 'split':
        from src.data.data_split import main as run_split
        run_split()
    elif args.mode == 'train_seg':
        from src.segmentation.train import main as run_seg_train
        sys.argv = [sys.argv[0], '--arch', args.arch, '--epochs', str(args.epochs), '--batch_size', str(args.batch_size)]
        if args.smoke_test:
            sys.argv.append('--smoke_test')
        run_seg_train()
    elif args.mode == 'eval_seg':
        from src.segmentation.evaluate import run_evaluation as run_seg_eval
        run_seg_eval()
    elif args.mode == 'train_cls':
        from src.classification.train import main as run_cls_train
        sys.argv = [sys.argv[0], '--arch', args.arch, '--dataset', args.dataset]
        if args.smoke_test:
            sys.argv.append('--smoke_test')
        run_cls_train()
    elif args.mode == 'eval_xai':
        from src.explainability.visualization import run_explainability_pipeline
        run_explainability_pipeline(model_arch=args.arch, dataset=args.dataset)
    elif args.mode == 'cross_eval':
        from src.evaluation.cross_dataset import main as run_cross_eval
        run_cross_eval()
    elif args.mode == 'app':
        app_path = os.path.join(PROJECT_ROOT, 'app', 'app.py')
        os.system(f"streamlit run \"{app_path}\"")
    elif args.mode == 'test':
        import pytest
        tests_dir = os.path.join(PROJECT_ROOT, 'tests')
        sys.exit(pytest.main([tests_dir]))

if __name__ == '__main__':
    main()
