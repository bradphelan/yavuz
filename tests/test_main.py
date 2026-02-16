"""Tests for the main module."""

from yavuz import main, __version__


def test_version():
    """Test that the version is defined."""
    assert __version__ == "0.1.0"


def test_main(capsys):
    """Test the main function."""
    main()
    captured = capsys.readouterr()
    assert "Hello from Yavuz!" in captured.out
