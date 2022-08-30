import gzip
from pathlib import Path

from stopes.core.launcher import SubmititLauncher
from stopes.modules.bitext.mining.count_lines import CountLinesConfig, CountLinesModule

test_input_str = """This is a sentence.
This is another sentence.
There are 3 of them.
"""


async def test_line_count_with_launcher(tmp_path: Path):
    file_path = tmp_path / "lines.tsv.gz"
    with gzip.open(file_path, mode="wt") as f:
        f.write(test_input_str)
    launcher = SubmititLauncher(
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
    )
    counter = CountLinesModule(CountLinesConfig(shards=[file_path]))
    result = await launcher.schedule(counter)
    assert result == [3]


async def test_line_count(tmp_path: Path):
    file_path = tmp_path / "lines.tsv.gz"
    with gzip.open(file_path, mode="wt") as f:
        f.write(test_input_str)
    counter = CountLinesModule(CountLinesConfig(shards=[file_path]))
    result = await counter.run(file_path, 0)
    assert result == 3
