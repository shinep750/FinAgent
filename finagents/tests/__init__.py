"""Test package.

Made a package (rather than relying on rootdir insertion into ``sys.path``) so
`tests.fake_llm` is importable by conftest and by test modules alike, and so
nothing here can be imported as if it were product code.
"""
