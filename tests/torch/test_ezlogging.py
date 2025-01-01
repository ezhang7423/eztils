import pytest
import inspect
from unittest.mock import patch
from eztils.ezlogging import print_creator


@pytest.mark.parametrize("color", ["red", "green", "blue"])
def test_print_creator_inner_calls_console_print(color):
    with patch("eztils.ezlogging.console.print") as mock_print:
        printer = print_creator(color)
        printer("Test message", extra_arg=True)
        mock_print.assert_called_once_with("Test message", extra_arg=True, style=color)


def test_print_creator_signature():
    printer = print_creator("red")
    sig = inspect.signature(printer)
    params = list(sig.parameters.keys())
    assert "args" in params
    assert "kwargs" in params


import os
import pytest
import torch
from unittest.mock import patch, MagicMock

from your_logger_file import (
    Logger,
    Config,
    tensor_to_list,
    requires_cfg,
)


@pytest.fixture
def logger_instance(tmp_path):
    """
    Pytest fixture that provides a fresh Logger instance per test.
    """
    # Create a fresh log file path in the temp directory
    log_file_path = tmp_path / "test_output.log"

    # Create a new instance of your Logger with default arguments
    logger = Logger()
    return logger, str(log_file_path)


@pytest.fixture
def mock_wandb():
    """
    Pytest fixture to mock out wandb.
    """
    with patch("your_logger_file.wandb") as wandb_mock:
        yield wandb_mock


# --------------------------------------------------
# Tests for tensor_to_list
# --------------------------------------------------

def test_tensor_to_list_cpu_tensor():
    tensor = torch.tensor([1, 2, 3])
    result = tensor_to_list(tensor)
    assert result == [1, 2, 3], "tensor_to_list should convert CPU tensor to a Python list"


def test_tensor_to_list_gpu_tensor():
    # This test will be skipped if no GPU is available. 
    # If your environment always has a GPU, remove the skip condition.
    if not torch.cuda.is_available():
        pytest.skip("GPU not available for this test.")
    tensor = torch.tensor([4, 5, 6], device="cuda")
    result = tensor_to_list(tensor)
    assert result == [4, 5, 6], "tensor_to_list should handle GPU tensors by moving them to CPU"


def test_tensor_to_list_non_tensor():
    with pytest.raises(TypeError):
        tensor_to_list([1, 2, 3])  # Not a torch.Tensor


def test_tensor_to_list_scalar():
    tensor = torch.tensor(42)
    result = tensor_to_list(tensor)
    assert result == 42, "Should handle zero-dimensional scalar tensors properly"


def test_tensor_to_list_empty_tensor():
    tensor = torch.tensor([])
    result = tensor_to_list(tensor)
    assert result == [], "tensor_to_list should convert an empty tensor to an empty list"


# --------------------------------------------------
# Tests for Logger.cfg
# --------------------------------------------------

def test_logger_cfg_basic(logger_instance):
    logger, _ = logger_instance
    cfg = Config(wandb=False, log_locally=True)
    logger.cfg(cfg)
    assert logger.configured is True, "Logger should be configured after calling cfg()"
    assert logger.log_wandb is False, "Expected wandb to be disabled in config"
    assert logger.log_locally is True, "Expected local logging to be enabled in config"


def test_logger_cfg_wandb_enabled(logger_instance, mock_wandb):
    logger, _ = logger_instance
    cfg = Config(wandb=True, wandb_project_name="test_project", wandb_entity="test_entity")
    logger.cfg(cfg)
    # Make sure wandb.init was called
    mock_wandb.init.assert_called_once_with(
        project="test_project", entity="test_entity", config=cfg
    )
    assert logger.log_wandb is True, "Expected wandb to be enabled in config"


def test_logger_cfg_wandb_missing_project_name(logger_instance, mock_wandb):
    logger, _ = logger_instance
    cfg = Config(wandb=True, wandb_project_name=None)
    with pytest.raises(ValueError):
        logger.cfg(cfg)


def test_logger_cfg_log_file_creates_file(logger_instance):
    logger, log_file_path = logger_instance
    cfg = Config(wandb=False, log_file=log_file_path, log_locally=False)
    logger.cfg(cfg)
    # Check that the file was created
    assert os.path.exists(log_file_path), "Log file should be created after calling cfg()"


def test_logger_cfg_no_local_logging(logger_instance):
    logger, _ = logger_instance
    cfg = Config(log_locally=False)
    logger.cfg(cfg)
    # The logger handlers should have been removed
    assert logger.log_locally is False, "Local logging should be disabled"
    # We can't easily test loguru's internal 'remove' side effects here,
    # but we confirm the config is set.


# --------------------------------------------------
# Tests for Logger.log_later
# --------------------------------------------------

def test_logger_log_later_before_cfg(logger_instance):
    logger, _ = logger_instance
    # Attempt to use log_later before cfg is called should raise ValueError
    with pytest.raises(ValueError):
        logger.log_later({"test": "data"})


def test_logger_log_later_accumulates_data(logger_instance):
    logger, _ = logger_instance
    logger.cfg(Config())
    logger.log_later({"key1": "value1"})
    logger.log_later({"key2": "value2"})
    # We won't flush yet, so check buffer
    assert len(logger.buffer) == 2, "Data should be accumulated in the buffer"


def test_logger_log_later_flush(logger_instance):
    logger, _ = logger_instance
    logger.cfg(Config())
    # We'll mock out the flush method to verify it's called
    with patch.object(logger, "flush") as mock_flush:
        logger.log_later({"key": "val"}, flush=True)
        mock_flush.assert_called_once(), "log_later should call flush if flush=True"


def test_logger_log_later_multiple_entries_overwrite(logger_instance):
    """
    When flush merges data with ChainMap, the later dictionary overwrites the earlier one.
    We'll verify that logic in flush() but also confirm log_later accumulates as expected.
    """
    logger, _ = logger_instance
    logger.cfg(Config())
    logger.log_later({"score": 100})
    logger.log_later({"score": 200})
    # After flush, 'score' should be 200
    assert len(logger.buffer) == 2, "Multiple entries in buffer are stored before flush"


def test_logger_log_later_none_data(logger_instance):
    """
    log_later can be called with None data (though not very useful).
    Make sure it just appends None to the buffer without crashing.
    """
    logger, _ = logger_instance
    logger.cfg(Config())
    logger.log_later(None)
    assert logger.buffer[-1] is None, "Should allow None data to be appended to buffer"


# --------------------------------------------------
# Tests for Logger.flush
# --------------------------------------------------

def test_logger_flush_merges_data(logger_instance, capsys):
    logger, _ = logger_instance
    logger.cfg(Config())
    logger.log_later({"a": 1, "b": 2})
    logger.log_later({"b": 3, "c": 4})
    logger.flush()

    # Check console output from flush (capfd or capsys)
    captured = capsys.readouterr()
    assert "Log Step 0" in captured.out, "Flush output should contain the flush step"
    # 'b' should be overwritten by 3
    assert "b | 3" in captured.out, "Flush should show updated value for 'b'"
    # 'a' and 'c' should also appear
    assert "a | 1" in captured.out
    assert "c | 4" in captured.out


def test_logger_flush_increments_step(logger_instance):
    logger, _ = logger_instance
    logger.cfg(Config())
    initial_step = logger._flush_step
    logger.log_later({"test": "step"})
    logger.flush()
    assert logger._flush_step == initial_step + 1, "Flush step should increment after flush"


def test_logger_flush_clears_buffer(logger_instance):
    logger, _ = logger_instance
    logger.cfg(Config())
    logger.log_later({"key": "val"})
    logger.flush()
    assert len(logger.buffer) == 0, "Buffer should be cleared after flush"


@patch("your_logger_file.tensor_to_list", return_value=[999])
def test_logger_flush_tensor_conversion(mock_tensor_to_list, logger_instance, capsys):
    """
    Ensures that flush calls tensor_to_list for torch.Tensor values.
    """
    logger, _ = logger_instance
    logger.cfg(Config())
    # Insert a dummy tensor
    logger.log_later({"tensor": torch.tensor([1, 2, 3])})
    logger.flush()
    mock_tensor_to_list.assert_called_once()
    captured = capsys.readouterr()
    # The displayed value should be [999], per our mock
    assert "[999]" in captured.out


def test_logger_flush_wandb_logging(logger_instance, mock_wandb):
    logger, _ = logger_instance
    cfg = Config(wandb=True, wandb_project_name="test_project")
    logger.cfg(cfg)
    logger.log_later({"metric": 123})
    logger.flush()
    # Confirm wandb.log was called with the merged data
    mock_wandb.log.assert_called_once_with({"metric": 123})


# --------------------------------------------------
# Tests for Logger.log_video
# --------------------------------------------------

def test_logger_log_video_without_cfg(logger_instance):
    """
    Should raise ValueError if we call log_video before logger.cfg().
    """
    logger, _ = logger_instance
    with pytest.raises(ValueError):
        logger.log_video("fake_path.mp4")


def test_logger_log_video_no_wandb(logger_instance):
    """
    If wandb is not enabled, calling log_video should do nothing special 
    (no error, but won't log anywhere).
    """
    logger, _ = logger_instance
    logger.cfg(Config(wandb=False))
    # Should not raise or call wandb
    logger.log_video("fake_path.mp4")


def test_logger_log_video_wandb_enabled(logger_instance, mock_wandb):
    logger, _ = logger_instance
    cfg = Config(wandb=True, wandb_project_name="test_project")
    logger.cfg(cfg)
    logger.log_video("fake_path.mp4")
    mock_wandb.log.assert_called_once()
    call_args = mock_wandb.log.call_args[0][0]
    assert "video" in call_args, "Should log a 'video' key to wandb"
    # The value should be wandb.Video(...)
    assert call_args["video"].__class__.__name__ == "Video", "Should create a wandb.Video object"


def test_logger_log_video_invalid_path(logger_instance, mock_wandb):
    """
    Just confirms it doesn't raise an error if the video path doesn't exist
    or is invalid. By default wandb.Video can handle it or throw its own error.
    """
    logger, _ = logger_instance
    cfg = Config(wandb=True, wandb_project_name="test_project")
    logger.cfg(cfg)
    # Shouldn't raise error, though wandb.Video might log a warning
    logger.log_video("nonexistent_path.mp4")
    mock_wandb.log.assert_called_once()


def test_logger_log_video_multiple_times(logger_instance, mock_wandb):
    """
    Log multiple videos in a row, ensure each call is separate.
    """
    logger, _ = logger_instance
    cfg = Config(wandb=True, wandb_project_name="test_project")
    logger.cfg(cfg)
    logger.log_video("video1.mp4")
    logger.log_video("video2.mp4")
    assert mock_wandb.log.call_count == 2, "Should log each video individually"


# --------------------------------------------------
# Optionally, you can also test requires_cfg decorator itself
# --------------------------------------------------

def test_requires_cfg_decorator_raises_error():
    """
    A small test to ensure that calling a requires_cfg method without
    self.configured = True raises a ValueError. We can use a minimal
    mock class or partial from the logger.
    """
    class TestClass:
        configured = False

        @requires_cfg
        def dummy_method(self):
            return "ok"

    t = TestClass()
    with pytest.raises(ValueError):
        t.dummy_method()
