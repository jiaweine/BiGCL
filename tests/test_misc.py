"""Unit tests for utils/misc.py."""

import os
import tempfile

import torch
import numpy as np
import pytest

from utils.misc import (
    set_seed,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    cosine_scheduler,
    get_logger,
)


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_numpy_reproducibility(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        assert np.allclose(a, b)

    def test_different_seeds_different_results(self):
        set_seed(1)
        a = torch.randn(5)
        set_seed(2)
        b = torch.randn(5)
        assert not torch.allclose(a, b)


class TestAverageMeter:
    def test_initial_state(self):
        meter = AverageMeter("test")
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0

    def test_single_update(self):
        meter = AverageMeter()
        meter.update(5.0)
        assert meter.val == 5.0
        assert meter.avg == 5.0
        assert meter.sum == 5.0
        assert meter.count == 1

    def test_multiple_updates(self):
        meter = AverageMeter()
        meter.update(2.0)
        meter.update(4.0)
        meter.update(6.0)
        assert meter.avg == pytest.approx(4.0)
        assert meter.count == 3
        assert meter.sum == pytest.approx(12.0)

    def test_weighted_update(self):
        meter = AverageMeter()
        meter.update(3.0, n=2)
        meter.update(6.0, n=1)
        assert meter.sum == pytest.approx(12.0)
        assert meter.count == 3
        assert meter.avg == pytest.approx(4.0)

    def test_reset(self):
        meter = AverageMeter()
        meter.update(10.0)
        meter.reset()
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.count == 0

    def test_name(self):
        meter = AverageMeter("loss")
        assert meter.name == "loss"


class TestSaveLoadCheckpoint:
    def test_save_and_load(self):
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(state, tmpdir, filename="test.pth")
            assert os.path.exists(os.path.join(tmpdir, "test.pth"))

            new_model = torch.nn.Linear(10, 5)
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            ckpt = load_checkpoint(new_model, os.path.join(tmpdir, "test.pth"), new_optimizer)

            assert ckpt["epoch"] == 10
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_save_best(self):
        state = {"model_state_dict": {}, "epoch": 5}
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(state, tmpdir, is_best=True)
            assert os.path.exists(os.path.join(tmpdir, "best.pth"))
            assert os.path.exists(os.path.join(tmpdir, "checkpoint.pth"))

    def test_creates_directory(self):
        state = {"model_state_dict": {}, "epoch": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "nested", "dir")
            save_checkpoint(state, save_dir)
            assert os.path.exists(os.path.join(save_dir, "checkpoint.pth"))


class TestCosineScheduler:
    def test_output_length(self):
        schedule = cosine_scheduler(0.1, 0.001, epochs=100)
        assert len(schedule) == 100

    def test_starts_at_base(self):
        schedule = cosine_scheduler(0.1, 0.001, epochs=50)
        assert schedule[0] == pytest.approx(0.1, abs=1e-6)

    def test_ends_at_final(self):
        schedule = cosine_scheduler(0.1, 0.001, epochs=50)
        assert schedule[-1] == pytest.approx(0.001, abs=1e-3)

    def test_warmup(self):
        schedule = cosine_scheduler(0.1, 0.001, epochs=100, warmup_epochs=10, warmup_start_value=0.0)
        assert schedule[0] == pytest.approx(0.0)
        assert schedule[9] == pytest.approx(0.1, abs=0.02)
        assert schedule[10] == pytest.approx(0.1, abs=1e-6)

    def test_monotone_decay_after_warmup(self):
        schedule = cosine_scheduler(0.1, 0.001, epochs=50, warmup_epochs=5)
        # After warmup, schedule should be non-increasing
        cosine_part = schedule[5:]
        for i in range(len(cosine_part) - 1):
            assert cosine_part[i] >= cosine_part[i + 1] - 1e-10

    def test_no_warmup(self):
        schedule = cosine_scheduler(1.0, 0.0, epochs=10, warmup_epochs=0)
        assert len(schedule) == 10
        assert schedule[0] == pytest.approx(1.0)


class TestGetLogger:
    def test_basic_logger(self):
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level == 20  # INFO level

    def test_file_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = get_logger("file_test", log_file=log_file)
            logger.info("test message")
            assert os.path.exists(log_file)
